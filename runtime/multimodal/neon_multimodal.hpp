#pragma once
// =============================================================================
// neon_multimodal.hpp — glue that makes the vision encoder (rawllm_vit.hpp) and
// the Motion VAE (rawllm_mvae.hpp), both built on the shared kernels in
// nn/rawllm_nn.hpp, part of the NEON runtime instead of stand-alone libraries.
//
// What lives here
//   * VisionRuntime  — owns a nanity-vit GGUF + encoder, checks that the tower's
//                      output width equals the language model's n_embd, caches
//                      recent image encodings (re-sending an image each chat turn
//                      costs a hash, not a ViT pass).
//   * Prompt         — a token-id sequence in which image positions carry unique
//                      negative "sentinel" ids plus the embedding rows that must
//                      replace the token-table lookup at those positions.
//                      Sentinels are derived from the image content, so NEON's
//                      cross-turn KV-prefix reuse keeps working: the same image
//                      at the same place matches, a different image never does.
//   * MotionRuntime  — owns nanity-mvae GGUFs (pose VAE and/or controller MLP)
//                      and exposes them through a small JSON op protocol
//                      (info / reset / step / decode / encode), which is what
//                      NEON's --interactive loop and CLI call.
//
// Nothing in here knows about the Tokenizer or the KV cache; NEON-3.cpp passes
// an encode callback in and applies the embedding override itself.
// =============================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/rawllm_json.hpp"
#include "../core/rawllm_loader.hpp"
#include "../core/rawllm_util.hpp"
#include "../nn/rawllm_nn.hpp"
#include "../vision/rawllm_vit.hpp"
#include "../motion/rawllm_mvae.hpp"
#include "../motion/rawllm_cvqvae.hpp"
#include "../motion/rawllm_cvqprior.hpp"

namespace mm {

// ----------------------------------------------------------------------------
// small helpers
// ----------------------------------------------------------------------------
inline uint64_t fnv1a64(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

inline bool b64_decode(const std::string& in, std::string& out) {
    static int8_t T[256];
    static bool init = false;
    if (!init) {
        std::memset(T, -1, sizeof T);
        const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[(uint8_t)A[i]] = (int8_t)i;
        init = true;
    }
    out.clear();
    out.reserve(in.size() * 3 / 4);
    uint32_t acc = 0; int bits = 0;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        int8_t v = T[c];
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t)v; bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((char)((acc >> bits) & 0xFF)); }
    }
    return true;
}

inline void append_floats(std::ostringstream& o, const float* v, size_t n) {
    o << '[';
    char buf[32];
    for (size_t i = 0; i < n; ++i) {
        std::snprintf(buf, sizeof buf, "%.7g", (double)v[i]);
        if (i) o << ',';
        o << buf;
    }
    o << ']';
}

// Reads a JSON number array into out; throws with a useful message on size mismatch.
inline void read_floats(const json::Value& arr, size_t expect, const char* what, std::vector<float>& out) {
    if (!arr.is_array()) throw std::runtime_error(std::string("motion: '") + what + "' must be an array of numbers");
    if (arr.size() != expect)
        throw std::runtime_error(std::string("motion: '") + what + "' has " + std::to_string(arr.size()) +
                                 " values, expected " + std::to_string(expect));
    out.resize(expect);
    for (size_t i = 0; i < expect; ++i) out[i] = arr[i].get_float();
}

// ----------------------------------------------------------------------------
// Vision
// ----------------------------------------------------------------------------
struct EncodedImage {
    std::vector<float> emb;      // n_tokens * n_embd
    size_t             n_tokens = 0;
    uint64_t           hash = 0; // content hash of the source PPM
};

class VisionRuntime {
public:
    bool   ready() const { return enc_ != nullptr; }
    size_t tokens_per_image() const { return enc_ ? enc_->out_tokens() : 0; }
    size_t out_dim() const { return enc_ ? enc_->out_dim() : 0; }
    const std::string& path() const { return path_; }

    void load(const std::string& path, size_t llm_n_embd, util::ThreadPool* pool) {
        auto f = std::make_unique<loader::GGUFLoader>();
        f->open(path);
        vit::Options o; o.pool = pool;
        auto e = std::make_unique<vit::VisionEncoder>();
        e->load(*f, o);
        if (e->out_dim() != llm_n_embd)
            throw std::runtime_error(
                "vision: '" + path + "' outputs " + std::to_string(e->out_dim()) + "-wide tokens but the language model has n_embd=" +
                std::to_string(llm_n_embd) + ". Re-export the tower with a connector that projects to the LLM width "
                "(nanity_export_vit.py --preset llava|siglip ... with the matching projector).");
        file_ = std::move(f);      // keep the mmap alive: the encoder borrows weight pointers from it
        enc_  = std::move(e);
        path_ = path;
    }

    // PPM bytes (P6) -> image-token embeddings, with a small content-addressed cache.
    bool encode_ppm(const std::string& ppm, EncodedImage& out, std::string& err) {
        if (!enc_) { err = "no vision model loaded (start NEON with --vision-model <nanity-vit.gguf>)"; return false; }
        const uint64_t h = fnv1a64(ppm.data(), ppm.size());
        auto it = cache_.find(h);
        if (it != cache_.end()) { out = it->second; return true; }
        vit::Image img;
        if (!vit::parse_ppm(ppm, img)) { err = "image is not a binary PPM (P6, maxval 255); convert with e.g. `magick in.png out.ppm`"; return false; }
        EncodedImage e;
        e.n_tokens = enc_->out_tokens();
        e.hash = h;
        e.emb.assign(e.n_tokens * enc_->out_dim(), 0.f);
        enc_->encode(img, e.emb.data());
        cache_[h] = e;
        order_.push_back(h);
        while (order_.size() > kCacheMax) { cache_.erase(order_.front()); order_.pop_front(); }
        out = std::move(e);
        return true;
    }

private:
    static constexpr size_t kCacheMax = 8;
    std::unique_ptr<loader::GGUFLoader>   file_;
    std::unique_ptr<vit::VisionEncoder>   enc_;
    std::string                            path_;
    std::unordered_map<uint64_t, EncodedImage> cache_;
    std::deque<uint64_t>                   order_;
};

// ----------------------------------------------------------------------------
// Multimodal prompts
// ----------------------------------------------------------------------------
struct ImageSpan {
    size_t             start = 0, len = 0;   // positions in Prompt::ids
    std::vector<float> emb;                  // len * n_embd
};

struct Prompt {
    std::vector<int32_t> ids;                // image positions hold negative sentinel ids
    std::vector<ImageSpan> spans;
    bool has_images() const { return !spans.empty(); }
};

inline int32_t sentinel_id(uint64_t image_hash, size_t k) {
    uint64_t h = fnv1a64(&k, sizeof k, image_hash);
    return -(int32_t)(1 + (h & 0x3FFFFFFFull));
}

inline const char* kImageMarker() { return "<image>"; }

// Splits `text` at "<image>" markers and splices the given images in, in order.
//   * no marker in the text  -> images go right after BOS, before the text
//   * more images than markers -> the extras follow the last placed image
//   * more markers than images -> the surplus markers are dropped
// `encode(str, add_bos)` is the language model's tokenizer.
template <class EncodeFn>
Prompt build_prompt(const std::string& text, const std::vector<EncodedImage>& imgs, EncodeFn&& encode) {
    Prompt p;
    const std::string mk = kImageMarker();
    std::vector<std::string> segs;
    for (size_t pos = 0;;) {
        size_t f = text.find(mk, pos);
        if (f == std::string::npos) { segs.push_back(text.substr(pos)); break; }
        segs.push_back(text.substr(pos, f - pos));
        pos = f + mk.size();
    }
    auto put_text = [&](const std::string& s, bool bos) {
        if (s.empty() && !bos) return;
        std::vector<int32_t> t = encode(s, bos);
        p.ids.insert(p.ids.end(), t.begin(), t.end());
    };
    auto put_image = [&](const EncodedImage& im) {
        ImageSpan sp; sp.start = p.ids.size(); sp.len = im.n_tokens; sp.emb = im.emb;
        for (size_t k = 0; k < im.n_tokens; ++k) p.ids.push_back(sentinel_id(im.hash, k));
        p.spans.push_back(std::move(sp));
    };

    if (imgs.empty()) {                       // markers without images are simply removed
        std::string joined;
        for (auto& s : segs) joined += s;
        put_text(joined, true);
        return p;
    }
    if (segs.size() == 1) {                   // no markers: images first
        put_text("", true);
        for (auto& im : imgs) put_image(im);
        put_text(segs[0], false);
        return p;
    }
    put_text(segs[0], true);
    size_t next_img = 0;
    for (size_t i = 1; i < segs.size(); ++i) {
        if (next_img < imgs.size()) put_image(imgs[next_img++]);
        put_text(segs[i], false);
    }
    while (next_img < imgs.size()) put_image(imgs[next_img++]);
    return p;
}

// A reused KV prefix must not end inside an image: either the whole image is
// cached or none of it is (a half-cached image has no consistent embedding rows).
inline size_t align_reuse(const Prompt& p, size_t match_len) {
    for (const auto& s : p.spans)
        if (match_len > s.start && match_len < s.start + s.len) return s.start;
    return match_len;
}

// Rows/mask/tokens for prompt positions [from, ids.size()). `toks` gets 0 at image positions.
inline bool build_override(const Prompt& p, size_t from, size_t n_embd,
                           std::vector<int32_t>& toks, std::vector<float>& rows, std::vector<uint8_t>& mask) {
    const size_t n = p.ids.size() - from;
    toks.assign(p.ids.begin() + from, p.ids.end());
    mask.assign(n, 0);
    bool any = false;
    for (const auto& s : p.spans) {
        const size_t a = std::max(s.start, from), b = std::min(s.start + s.len, p.ids.size());
        if (a >= b) continue;
        if (!any) { rows.assign(n * n_embd, 0.f); any = true; }
        for (size_t pos = a; pos < b; ++pos) {
            std::memcpy(&rows[(pos - from) * n_embd], &s.emb[(pos - s.start) * n_embd], n_embd * sizeof(float));
            mask[pos - from] = 1; toks[pos - from] = 0;
        }
    }
    return any;
}

// ----------------------------------------------------------------------------
// Motion
// ----------------------------------------------------------------------------
class MotionRuntime {
public:
    bool ready() const { return vae_ != nullptr || ctrl_ != nullptr || cvq_ != nullptr || prior_ != nullptr; }
    bool has_cvqvae() const { return cvq_ != nullptr; }
    bool has_cvqprior() const { return prior_ != nullptr; }
    bool has_vae() const { return vae_ != nullptr; }
    bool has_controller() const { return ctrl_ != nullptr; }

    // Loads one nanity-mvae GGUF. A controller (kind=mlp) and a pose VAE can both be loaded.
    void load(const std::string& path, util::ThreadPool* pool = nullptr) {
        auto f = std::make_unique<loader::GGUFLoader>();
        f->open(path);
        if (f->meta_string("general.architecture", "") == "nanity-cvqvae") {   // NERVA conditional VQ-VAE
            cvqvae::Options co; co.pool = pool;
            auto c = std::make_unique<cvqvae::CVQVAE>();
            c->load(*f, co);
            cvq_file_ = std::move(f); cvq_ = std::move(c); cvq_path_ = path;
            return;
        }
        if (f->meta_string("general.architecture", "") == "nanity-cvqprior") {   // NERVA cVQ-VAE prior (samples the VAE's codes)
            cvqprior::Options po; po.pool = pool;
            auto p = std::make_unique<cvqprior::CodePrior>();
            p->load(*f, po);
            prior_file_ = std::move(f); prior_ = std::move(p); prior_path_ = path; prior_ready_ = false;
            return;
        }
        mvae::Options o; o.pool = pool;
        auto m = std::make_unique<mvae::MotionVAE>();
        m->load(*f, o);
        if (m->kind == mvae::Kind::MLP) { ctrl_file_ = std::move(f); ctrl_ = std::move(m); ctrl_path_ = path; }
        else                            { vae_file_  = std::move(f); vae_  = std::move(m); vae_path_  = path; have_hist_ = false; }
    }

    void set_seed(uint64_t s) { rng_.seed((uint32_t)(s ^ (s >> 32))); prior_rng_.seed(s); }

    // Mean pose as the neutral starting history (normalised 0 == the training mean/midpoint).
    void reset_to_mean() {
        need_vae();
        std::vector<float> z(vae_->frame_size * vae_->n_cond, 0.f), raw(z.size());
        vae_->denormalize(z.data(), raw.data(), vae_->n_cond);
        vae_->reset_history(raw.data());
        have_hist_ = true;
    }

    std::string info_json() const {
        std::ostringstream o;
        o << "{";
        if (vae_) {
            o << "\"vae\":{\"path\":\"" << json::escape(vae_path_) << "\",\"kind\":\""
              << (vae_->kind == mvae::Kind::PoseMixtureVAE ? "pose_mixture_vae" : "pose_vae")
              << "\",\"frame_size\":" << vae_->frame_size << ",\"latent_size\":" << vae_->latent_size
              << ",\"n_cond\":" << vae_->n_cond << ",\"n_future\":" << vae_->n_future
              << ",\"n_experts\":" << vae_->n_experts << ",\"has_encoder\":" << (vae_->has_encoder() ? "true" : "false") << "}";
        } else o << "\"vae\":null";
        if (cvq_) {
            o << ",\"cvqvae\":{\"path\":\"" << json::escape(cvq_path_) << "\",\"window\":" << cvq_->window
              << ",\"channels\":" << cvq_->channels << ",\"num_codes\":" << cvq_->num_codes
              << ",\"latent_steps\":" << (cvq_->window >> cvq_->downs) << ",\"frames_per_code\":" << (size_t(1) << cvq_->downs)
              << ",\"dt\":" << cvq_->dt << ",\"actions\":[";
            for (size_t i = 0; i < cvq_->action_names.size(); ++i) o << (i ? "," : "") << "\"" << json::escape(cvq_->action_names[i]) << "\"";
            o << "]}";
        } else o << ",\"cvqvae\":null";
        if (prior_) {
            const auto& P = *prior_;
            o << ",\"cvqprior\":{\"path\":\"" << json::escape(prior_path_) << "\",\"num_codes\":" << P.K << ",\"codes_per_window\":" << P.n
              << ",\"context_windows\":" << P.windows << ",\"d\":" << P.d << ",\"layers\":" << P.layers << ",\"heads\":" << P.heads
              << ",\"params\":" << P.params() << ",\"stride\":" << P.stride << ",\"dt\":" << P.dt;
            if (!std::isnan(P.val_nll)) o << ",\"val_nll\":" << P.val_nll;
            const std::string why = prior_vs_vae();
            o << ",\"matches_vae\":" << (cvq_ ? (why.empty() ? "true" : "false") : "null");
            if (!why.empty()) o << ",\"mismatch\":\"" << json::escape(why) << "\"";
            o << ",\"actions\":[";
            for (size_t i = 0; i < P.action_names.size(); ++i) o << (i ? "," : "") << "\"" << json::escape(P.action_names[i]) << "\"";
            o << "]";
            if (P.has_goal()) {                                        // what a planner needs: the numbers' names + what the training windows really contain
                o << ",\"goal\":{\"names\":[";
                for (size_t i = 0; i < P.goal_dim; ++i) o << (i ? "," : "") << "\"" << json::escape(i < P.goal_names.size() ? P.goal_names[i] : "g" + std::to_string(i)) << "\"";
                o << "],\"mean\":"; append_floats(o, P.goal_mean.data(), P.goal_dim);
                o << ",\"std\":"; append_floats(o, P.goal_std.data(), P.goal_dim);
                o << ",\"turn_max\":" << P.turn_max << ",\"speed_cruise\":" << P.speed_cruise << ",\"speed_max\":" << P.speed_max << "}";
            } else o << ",\"goal\":null";
            o << ",\"styles\":[";
            for (size_t i = 0; i < P.style_names.size(); ++i) o << (i ? "," : "") << "\"" << json::escape(P.style_names[i]) << "\"";
            o << "]}";
        } else o << ",\"cvqprior\":null";
        if (ctrl_) {
            o << ",\"controller\":{\"path\":\"" << json::escape(ctrl_path_) << "\",\"obs_dim\":" << ctrl_->mlp_in()
              << ",\"action_dim\":" << ctrl_->mlp_out() << "}";
        } else o << ",\"controller\":null";
        o << "}";
        return o.str();
    }

    // One JSON op in, one JSON object (as text) out. Throws std::runtime_error on bad input.
    //   {"op":"info"}
    //   {"op":"reset"}                                   -> mean pose
    //   {"op":"reset","frame":[...]}                     -> that raw frame repeated n_cond times
    //   {"op":"reset","frames":[[...],...]}              -> n_cond raw frames, newest first
    //   {"op":"step","n":30}                             -> n frames sampled from the prior
    //   {"op":"step","n":30,"temperature":0.7,"seed":1}  -> prior scaled by temperature
    //   {"op":"step","z":[...]}  /  {"op":"step","z_seq":[[...],...]}
    //   {"op":"step","obs_seq":[[...],...]}              -> controller MLP picks z per frame
    //   {"op":"decode","z":[...]}                        -> preview next frame, history untouched
    //   {"op":"encode","target":[...],"cond":[[...],...]} -> mu/logvar (cond defaults to current history)
    std::string handle(const json::Value& req) {
        const std::string op = req["op"].get_str("info");
        if (op == "info") return info_json();
        if (op == "cvq_encode" || op == "cvq_decode" || op == "cvq_recon") return handle_cvq(op, req);
        if (op == "cvq_sample" || op == "cvq_prior_reset" || op == "cvq_prior_next") return handle_cvq_prior(op, req);

        if (op == "reset") {
            need_vae();
            const size_t F = vae_->frame_size, C = vae_->n_cond;
            if (req.contains("frames")) {
                const auto& fr = req["frames"];
                if (!fr.is_array() || fr.size() != C)
                    throw std::runtime_error("motion: 'frames' must hold n_cond=" + std::to_string(C) + " frames (newest first)");
                std::vector<float> all(F * C), one;
                for (size_t c = 0; c < C; ++c) { read_floats(fr[c], F, "frames[i]", one); std::copy(one.begin(), one.end(), all.begin() + c * F); }
                vae_->reset_history(all.data());
                have_hist_ = true;
            } else if (req.contains("frame")) {
                std::vector<float> one, all(F * C);
                read_floats(req["frame"], F, "frame", one);
                for (size_t c = 0; c < C; ++c) std::copy(one.begin(), one.end(), all.begin() + c * F);
                vae_->reset_history(all.data());
                have_hist_ = true;
            } else reset_to_mean();
            return "{\"ok\":true}";
        }

        if (op == "step") {
            need_vae();
            if (!have_hist_) reset_to_mean();
            const size_t F = vae_->frame_size, L = vae_->latent_size;
            int n = req["n"].get_int(1);
            n = std::max(1, std::min(n, 4096));
            if (req.contains("seed")) set_seed((uint64_t)req["seed"].get_int64(0));
            const float temp    = req["temperature"].get_float(1.0f);
            const float z_scale = req["z_scale"].get_float(1.0f);
            const bool have_z = req.contains("z"), have_zs = req.contains("z_seq"), have_obs = req.contains("obs_seq");
            std::vector<float> zc, z(L), obs, next(F * vae_->n_future);
            if (have_z) read_floats(req["z"], L, "z", zc);
            if (have_zs && (!req["z_seq"].is_array() || req["z_seq"].size() == 0)) throw std::runtime_error("motion: 'z_seq' must be a non-empty array");
            if (have_obs) {
                if (!ctrl_) throw std::runtime_error("motion: 'obs_seq' needs a controller (load a kind=mlp model with --motion-model)");
                if (ctrl_->mlp_out() != L) throw std::runtime_error("motion: controller action_dim " + std::to_string(ctrl_->mlp_out()) + " != VAE latent " + std::to_string(L));
                if (!req["obs_seq"].is_array() || req["obs_seq"].size() == 0) throw std::runtime_error("motion: 'obs_seq' must be a non-empty array");
            }
            std::ostringstream o; o << "{\"frames\":[";
            std::normal_distribution<float> nd(0.f, 1.f);
            for (int i = 0; i < n; ++i) {
                if (have_obs) {
                    const auto& ov = req["obs_seq"][(size_t)i % req["obs_seq"].size()];
                    read_floats(ov, ctrl_->mlp_in(), "obs_seq[i]", obs);
                    ctrl_->mlp_forward(obs.data(), z.data());
                    for (auto& v : z) v *= z_scale;
                } else if (have_zs) {
                    read_floats(req["z_seq"][(size_t)i % req["z_seq"].size()], L, "z_seq[i]", z);
                } else if (have_z) {
                    z = zc;
                } else {
                    for (size_t k = 0; k < L; ++k) z[k] = nd(rng_) * temp;
                }
                vae_->step(z.data(), next.data());
                if (i) o << ',';
                append_floats(o, next.data(), F);
            }
            o << "],\"frame_size\":" << F << "}";
            return o.str();
        }

        if (op == "decode") {
            need_vae();
            if (!have_hist_) reset_to_mean();
            const size_t F = vae_->frame_size, L = vae_->latent_size, C = vae_->n_cond;
            std::vector<float> z, cond(F * C), out(F * vae_->n_future), raw(F * vae_->n_future);
            read_floats(req["z"], L, "z", z);
            vae_->normalize(vae_->history(), cond.data(), C);
            vae_->decode(z.data(), cond.data(), out.data());
            vae_->denormalize(out.data(), raw.data(), vae_->n_future);
            std::ostringstream o; o << "{\"frame\":"; append_floats(o, raw.data(), F); o << "}";
            return o.str();
        }

        if (op == "encode") {
            need_vae();
            if (!vae_->has_encoder()) throw std::runtime_error("motion: this file has no encoder tensors");
            const size_t F = vae_->frame_size, L = vae_->latent_size, C = vae_->n_cond, T = F * vae_->n_future;
            std::vector<float> target, tn(T), cond_raw(F * C), cond(F * C), mu(L), lv(L), one;
            read_floats(req["target"], T, "target", target);
            if (req.contains("cond")) {
                const auto& cr = req["cond"];
                if (!cr.is_array() || cr.size() != C) throw std::runtime_error("motion: 'cond' must hold n_cond=" + std::to_string(C) + " frames (newest first)");
                for (size_t c = 0; c < C; ++c) { read_floats(cr[c], F, "cond[i]", one); std::copy(one.begin(), one.end(), cond_raw.begin() + c * F); }
            } else {
                if (!have_hist_) reset_to_mean();
                std::copy(vae_->history(), vae_->history() + F * C, cond_raw.begin());
            }
            vae_->normalize(target.data(), tn.data(), vae_->n_future);
            vae_->normalize(cond_raw.data(), cond.data(), C);
            vae_->encode(tn.data(), cond.data(), mu.data(), lv.data());
            std::ostringstream o; o << "{\"mu\":"; append_floats(o, mu.data(), L);
            o << ",\"logvar\":"; append_floats(o, lv.data(), L); o << "}";
            return o.str();
        }

        throw std::runtime_error("motion: unknown op '" + op + "' (info|reset|step|decode|encode|cvq_encode|cvq_decode|cvq_recon|cvq_sample|cvq_prior_reset|cvq_prior_next)");
    }

private:
    // action: absent/null -> unconditional label; string -> name from the model; number -> index
    int cvq_action(const json::Value& req) const {
        const auto& a = req["action"];
        if (a.is_string()) {
            if (a.get_str() == "null") return -1;
            int i = cvq_->action_index(a.get_str());
            if (i < 0) throw std::runtime_error("cvqvae: unknown action '" + a.get_str() + "' (see info for the list)");
            return i;
        }
        if (a.is_number()) {
            int i = a.get_int(-1);
            if (i < -1 || i >= (int)cvq_->n_actions) throw std::runtime_error("cvqvae: action index out of range");
            return i;
        }
        return -1;
    }

    //   {"op":"cvq_encode","frames":[[30]...],"action":"walk","normalized":false}  -> {"codes":[...]}
    //   {"op":"cvq_decode","codes":[...],"action":"walk"}                          -> {"frames":[[30]...]}
    //   {"op":"cvq_recon","frames":[...]}                                          -> encode -> quantise -> decode
    // Frames are in train_cvqvae.py's canonicalised physical units (flags 0/1); count must be a multiple of 2^downs.
    // "normalized":true means input/output are model space (decode then returns raw logits for flag channels).
    std::string handle_cvq(const std::string& op, const json::Value& req) {
        if (!cvq_) throw std::runtime_error("cvqvae: no cVQ-VAE loaded (start NEON with --motion-model <nanity-cvqvae.gguf>)");
        const size_t C = cvq_->channels, G = size_t(1) << cvq_->downs;
        const int act = cvq_action(req);
        const bool normd = req["normalized"].get_bool(false);
        std::vector<int32_t> codes;
        size_t T = 0;
        if (op != "cvq_decode") {
            const auto& fr = req["frames"];
            if (!fr.is_array() || fr.size() == 0 || fr.size() % G)
                throw std::runtime_error("cvqvae: 'frames' must hold a positive multiple of " + std::to_string(G) + " frames");
            T = fr.size();
            std::vector<float> x(T * C), one;
            for (size_t t = 0; t < T; ++t) { read_floats(fr[t], C, "frames[i]", one); std::copy(one.begin(), one.end(), x.begin() + t * C); }
            if (!normd) cvq_->normalize(x.data(), x.data(), T);
            codes.resize(T / G);
            cvq_->tokens(x.data(), T, act, codes.data());
            if (op == "cvq_encode") {
                std::ostringstream o; o << "{\"codes\":[";
                for (size_t i = 0; i < codes.size(); ++i) o << (i ? "," : "") << codes[i];
                o << "]}";
                return o.str();
            }
        } else {
            const auto& cj = req["codes"];
            if (!cj.is_array() || cj.size() == 0) throw std::runtime_error("cvqvae: 'codes' must be a non-empty array of indices");
            for (size_t i = 0; i < cj.size(); ++i) codes.push_back(cj[i].get_int(-1));
            T = codes.size() * G;
        }
        std::vector<float> y(T * C), phys(T * C);
        cvq_->decode_codes(codes.data(), codes.size(), act, y.data());
        if (!normd) cvq_->to_motion(y.data(), phys.data(), T);
        const float* out = normd ? y.data() : phys.data();
        std::ostringstream o; o << "{";
        if (op == "cvq_recon") {
            o << "\"codes\":[";
            for (size_t i = 0; i < codes.size(); ++i) o << (i ? "," : "") << codes[i];
            o << "],";
        }
        o << "\"frames\":[";
        for (size_t t = 0; t < T; ++t) { if (t) o << ','; append_floats(o, out + t * C, C); }
        o << "]}";
        return o.str();
    }

    // Empty string when the loaded prior and VAE belong together (or one of them is absent).
    std::string prior_vs_vae() const {
        if (!prior_ || !cvq_) return "";
        return cvqprior::check_compatible(prior_->K, prior_->n, prior_->n_actions, prior_->action_names,
                                          cvq_->num_codes, cvq_->window >> cvq_->downs, cvq_->n_actions, cvq_->action_names);
    }
    int prior_action(const json::Value& req) const {
        const auto& a = req["action"];
        if (a.is_string()) {
            if (a.get_str() == "null") return -1;
            int i = prior_->action_index(a.get_str());
            if (i < 0) throw std::runtime_error("cvqprior: unknown action '" + a.get_str() + "' (see info for the list)");
            return i;
        }
        if (a.is_number()) {
            int i = a.get_int(-1);
            if (i < -1 || i >= (int)prior_->n_actions) throw std::runtime_error("cvqprior: action index out of range");
            return i;
        }
        return -1;
    }
    // style: a name from info.styles, an index, "null"/absent = none (-1)
    int prior_style(const json::Value& req) const {
        const auto& a = req["style"];
        if (a.is_null()) return -1;
        if (!prior_->has_style()) throw std::runtime_error("cvqprior: this prior was trained without style ids");
        if (a.is_string()) {
            if (a.get_str() == "null") return -1;
            int i = prior_->style_index(a.get_str());
            if (i < 0) throw std::runtime_error("cvqprior: unknown style '" + a.get_str() + "' (see info.styles)");
            return i;
        }
        if (a.is_number()) {
            int i = a.get_int(-1);
            if (i < -1 || i >= (int)prior_->n_styles) throw std::runtime_error("cvqprior: style index out of range");
            return i;
        }
        throw std::runtime_error("cvqprior: \"style\" must be a name, an index or null");
    }
    // goal: goal_dim RAW numbers in info.goal.names order (the prior normalises them); null/absent -> {} (no goal for that window)
    std::vector<float> prior_goal(const json::Value& g) const {
        std::vector<float> out;
        if (g.is_null()) return out;
        if (!prior_->has_goal()) throw std::runtime_error("cvqprior: this prior was trained without goal conditioning (re-train without --no-prior-goal)");
        if (!g.is_array() || g.size() != prior_->goal_dim)
            throw std::runtime_error("cvqprior: a goal is an array of " + std::to_string(prior_->goal_dim) + " numbers (see info.goal.names)");
        for (size_t i = 0; i < prior_->goal_dim; ++i) {
            if (!g[i].is_number()) throw std::runtime_error("cvqprior: goal entries must be numbers");
            const float v = g[i].get_float(0.f);
            if (!std::isfinite(v)) throw std::runtime_error("cvqprior: goal numbers must be finite");
            out.push_back(v);
        }
        return out;
    }
    float prior_cfg(const json::Value& req, float def) const {
        const float s = req["cfg_scale"].get_float(def);
        if (!std::isfinite(s) || s < 0.f || s > 20.f) throw std::runtime_error("cvqprior: cfg_scale must be in [0, 20] (1 = off)");
        if (s != 1.f && !prior_->has_goal()) throw std::runtime_error("cvqprior: cfg_scale needs a prior trained with goal conditioning");
        return s;
    }
    static const float* nullptr_or(const std::vector<float>& v) { return v.empty() ? nullptr : v.data(); }
    cvqprior::SampleParams prior_params(const json::Value& req) const {
        cvqprior::SampleParams sp;
        sp.temperature = req["temperature"].get_float(1.0f);
        sp.top_k       = req["top_k"].get_int(0);
        sp.top_p       = req["top_p"].get_float(1.0f);
        if (!(sp.temperature > 0.f) || sp.top_k < 0 || !(sp.top_p > 0.f))
            throw std::runtime_error("cvqprior: need temperature > 0, top_k >= 0, top_p > 0");
        return sp;
    }
    // One window of codes -> VAE -> "[ [C]*T ]" (T = n << downs frames; physical units unless normd, then raw model output).
    void append_window_frames(std::ostringstream& o, const int32_t* codes, int act, bool normd) {
        const size_t n = prior_->n, C = cvq_->channels, T = n << cvq_->downs;
        std::vector<float> y(T * C), phys(T * C);
        cvq_->decode_codes(codes, n, act, y.data());
        if (!normd) cvq_->to_motion(y.data(), phys.data(), T);
        const float* out = normd ? y.data() : phys.data();
        o << '[';
        for (size_t t = 0; t < T; ++t) { if (t) o << ','; append_floats(o, out + t * C, C); }
        o << ']';
    }

    // The prior samples the VAE's discrete codes, so motion can be generated without an input clip.
    //   {"op":"cvq_sample","action":"walk","windows":6,"temperature":0.9,"top_k":40,"top_p":1.0,"seed":1,"decode":true,"normalized":false}
    //        -> {"codes":[[n]*windows],"probs":[[n]*windows],"stride":S,"dt":dt[,"frames":[[T][C] per window]]}
    //   {"op":"cvq_prior_reset","action":"walk","seed":1}          start an endless session (keeps the KV cache between calls)
    //   {"op":"cvq_prior_next","temperature":0.9,"top_k":40,"decode":true}   -> the next window: {"window":i,"codes":[n],"probs":[n],"entropy":[n][,"frames":[T][C]]}
    // Goal-conditioned priors (info.goal != null) also take, all optional: "style" (name from info.styles | index | null), "cfg_scale" (1 = off; reset / cvq_sample),
    // "goal": [dyaw,dx,dz,speed] RAW numbers for THIS window (cvq_prior_next), or "goal" (one for every window) / "goals" (one per window, the last repeats) in cvq_sample.
    // No "action" (or "null") + a goal = movement described by numbers only, no label. A window without a goal uses the null goal.
    // "probs" = the prior's own probability (temperature 1, unfiltered) of each code it picked. Windows are NOT stitched here:
    // consecutive windows overlap by (window - stride) frames and are cross-faded by the caller (NERVA's stitch), `stride` is reported.
    // "decode":true needs a matching VAE loaded too; without one only codes come back.
    std::string handle_cvq_prior(const std::string& op, const json::Value& req) {
        if (!prior_) throw std::runtime_error("cvqprior: no prior loaded (start NEON with --motion-model <nanity-cvqprior.gguf>)");
        const std::string why = prior_vs_vae();
        const bool dec = req["decode"].get_bool(false);
        if (dec) {
            if (!cvq_) throw std::runtime_error("cvqprior: \"decode\":true needs the cVQ-VAE too (--motion-model <nanity-cvqvae.gguf>)");
            if (!why.empty()) throw std::runtime_error("cvqprior: prior and VAE do not match: " + why);
        }
        const bool normd = req["normalized"].get_bool(false);
        if (req.contains("seed")) prior_rng_.seed((uint64_t)req["seed"].get_int64(0));
        const size_t n = prior_->n;

        if (op == "cvq_sample") {
            const int act = prior_action(req);
            int nw = std::max(1, std::min(req["windows"].get_int(1), 256));
            const auto sp = prior_params(req);
            const int sty = prior_style(req);
            std::vector<float> gflat;
            if (req["goals"].is_array() && req["goals"].size() > 0) {
                for (int w = 0; w < nw; ++w) {
                    auto g = prior_goal(req["goals"][std::min<size_t>((size_t)w, req["goals"].size() - 1)]);
                    if (g.empty()) throw std::runtime_error("cvqprior: every entry of \"goals\" must be an array of numbers");
                    gflat.insert(gflat.end(), g.begin(), g.end());
                }
            } else {
                auto g = prior_goal(req["goal"]);
                if (!g.empty()) for (int w = 0; w < nw; ++w) gflat.insert(gflat.end(), g.begin(), g.end());
            }
            prior_->set_cfg_scale(prior_cfg(req, 1.f));
            std::vector<int32_t> codes((size_t)nw * n); std::vector<float> pr((size_t)nw * n);
            prior_->generate(act, (size_t)nw, sp, prior_rng_, codes.data(), pr.data(), gflat.empty() ? nullptr : gflat.data(), sty);
            std::ostringstream o; o << "{\"codes\":[";
            for (int w = 0; w < nw; ++w) { if (w) o << ','; o << '['; for (size_t i = 0; i < n; ++i) o << (i ? "," : "") << codes[(size_t)w * n + i]; o << ']'; }
            o << "],\"probs\":[";
            for (int w = 0; w < nw; ++w) { if (w) o << ','; append_floats(o, pr.data() + (size_t)w * n, n); }
            o << "],\"stride\":" << prior_->stride << ",\"dt\":" << prior_->dt;
            if (dec) {
                o << ",\"frames\":[";
                for (int w = 0; w < nw; ++w) { if (w) o << ','; append_window_frames(o, codes.data() + (size_t)w * n, act, normd); }
                o << ']';
            }
            o << '}';
            return o.str();
        }

        if (op == "cvq_prior_reset") {
            prior_act_ = prior_action(req);
            const int sty = prior_style(req); const float cfg = prior_cfg(req, 1.f);
            prior_->reset(prior_act_, sty); prior_->set_cfg_scale(cfg); prior_windows_done_ = 0; prior_ready_ = true;
            return "{\"ok\":true,\"context_windows\":" + std::to_string(prior_->windows) + ",\"codes_per_window\":" + std::to_string(n) + "}";
        }

        // cvq_prior_next
        if (!prior_ready_) { prior_act_ = -1; prior_->reset(-1); prior_windows_done_ = 0; prior_ready_ = true; }   // no reset yet -> unconditional
        const auto sp = prior_params(req);
        std::vector<int32_t> codes(n); std::vector<float> pr(n), ent(n);
        const auto goal = prior_goal(req["goal"]);
        if (req.contains("cfg_scale")) prior_->set_cfg_scale(prior_cfg(req, 1.f));
        prior_->next_window(prior_windows_done_, prior_rng_, sp, codes.data(), pr.data(), ent.data(), nullptr_or(goal));
        std::ostringstream o; o << "{\"window\":" << prior_windows_done_ << ",\"codes\":[";
        for (size_t i = 0; i < n; ++i) o << (i ? "," : "") << codes[i];
        o << "],\"probs\":"; append_floats(o, pr.data(), n);
        o << ",\"entropy\":"; append_floats(o, ent.data(), n);
        o << ",\"stride\":" << prior_->stride << ",\"dt\":" << prior_->dt;
        if (dec) { o << ",\"frames\":"; append_window_frames(o, codes.data(), prior_act_, normd); }
        o << '}';
        ++prior_windows_done_;
        return o.str();
    }

    void need_vae() const {
        if (!vae_) throw std::runtime_error("motion: no pose VAE loaded (start NEON with --motion-model <nanity-mvae.gguf>)");
    }
    std::unique_ptr<loader::GGUFLoader> vae_file_, ctrl_file_, cvq_file_, prior_file_;
    std::unique_ptr<mvae::MotionVAE>    vae_, ctrl_;
    std::unique_ptr<cvqvae::CVQVAE>     cvq_;
    std::unique_ptr<cvqprior::CodePrior> prior_;
    std::string                          prior_path_;
    std::mt19937_64                      prior_rng_{std::random_device{}()};
    bool                                 prior_ready_ = false;
    int                                  prior_act_ = -1;
    size_t                               prior_windows_done_ = 0;
    std::string                          vae_path_, ctrl_path_, cvq_path_;
    bool                                 have_hist_ = false;
    std::mt19937                         rng_{std::random_device{}()};
};

} // namespace mm
