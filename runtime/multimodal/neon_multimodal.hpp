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
    bool ready() const { return vae_ != nullptr || ctrl_ != nullptr; }
    bool has_vae() const { return vae_ != nullptr; }
    bool has_controller() const { return ctrl_ != nullptr; }

    // Loads one nanity-mvae GGUF. A controller (kind=mlp) and a pose VAE can both be loaded.
    void load(const std::string& path, util::ThreadPool* pool = nullptr) {
        auto f = std::make_unique<loader::GGUFLoader>();
        f->open(path);
        mvae::Options o; o.pool = pool;
        auto m = std::make_unique<mvae::MotionVAE>();
        m->load(*f, o);
        if (m->kind == mvae::Kind::MLP) { ctrl_file_ = std::move(f); ctrl_ = std::move(m); ctrl_path_ = path; }
        else                            { vae_file_  = std::move(f); vae_  = std::move(m); vae_path_  = path; have_hist_ = false; }
    }

    void set_seed(uint64_t s) { rng_.seed((uint32_t)(s ^ (s >> 32))); }

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

        throw std::runtime_error("motion: unknown op '" + op + "' (info|reset|step|decode|encode)");
    }

private:
    void need_vae() const {
        if (!vae_) throw std::runtime_error("motion: no pose VAE loaded (start NEON with --motion-model <nanity-mvae.gguf>)");
    }
    std::unique_ptr<loader::GGUFLoader> vae_file_, ctrl_file_;
    std::unique_ptr<mvae::MotionVAE>    vae_, ctrl_;
    std::string                          vae_path_, ctrl_path_;
    bool                                 have_hist_ = false;
    std::mt19937                         rng_{std::random_device{}()};
};

} // namespace mm
