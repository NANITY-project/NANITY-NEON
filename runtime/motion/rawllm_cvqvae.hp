#pragma once
// =============================================================================
// rawllm_cvqvae.hpp — conditional VQ-VAE (cVQ-VAE) motion model inference
//
// Implements the inference graph of tools/train_cvqvae.py (NERVA repo):
//
//   cond      = cond_mlp(act_emb[action])                 Linear -> SiLU -> Linear
//   encoder   = Conv1d(k3) -> [FiLMRes x n_res -> Conv1d(k4, stride 2, pad 1)] x downs
//               -> GroupNorm -> SiLU -> Conv1d(k3)                   [T,30] -> [T/2^downs, D]
//   quantizer = exact nearest codebook row (L2); EMA state is training-only
//   decoder   = Conv1d(k3) -> [FiLMRes x n_res -> nearest x2 -> Conv1d(k3)] x downs
//               -> GroupNorm -> SiLU -> Conv1d(k3)                   [N, D] -> [N*2^downs, 30]
//   FiLMRes   : y = GroupNorm8(x)*(1+s)+b ; x + conv1x1(SiLU(conv3_dil(SiLU(y))))   (s,b = Linear(cond))
//
// Time layout everywhere is [T][C] (time-major). Because the PyTorch graph is
// channel-first, weights are exported tap-major ([out, k*in], column =
// tap*in + channel; see conversion/nanity_export_cvqvae.py) and every
// convolution is run as im2col + ONE batched nn::linear over all time steps
// (GEMM path), with bias and the following SiLU fused where the graph has one.
//
// Exactness: everything is F32 and matches the reference up to float
// summation order. GroupNorm statistics accumulate in double. The only lossy
// knob is Options::convert (F32 -> F16/BF16 weight re-encode at load),
// off by default per NEON's convention. Q8_0 is rejected: hidden*3 is not a
// multiple of 32 for the default sizes.
//
// Not thread-safe (scratch buffers are members). No heap allocation after the
// first call at a given length.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../nn/rawllm_nn.hpp"
#include "../core/rawllm_loader.hpp"

namespace cvqvae {

using nn::Act;
using nn::Linear;
using nn::WType;

namespace detail {
inline WType wtype_of(loader::GGMLType t) {
    switch (t) {
        case loader::GGMLType::F32:  return WType::F32;
        case loader::GGMLType::F16:  return WType::F16;
        case loader::GGMLType::BF16: return WType::BF16;
        default: throw std::runtime_error("cvqvae: unsupported tensor type (use F32/F16/BF16)");
    }
}
struct TensorFile {
    const loader::GGUFLoader& g;
    explicit TensorFile(const loader::GGUFLoader& gg) : g(gg) {}
    const loader::TensorInfo* find(const std::string& n) const {
        for (auto& t : g.tensors) if (t.name == n) return &t;
        return nullptr;
    }
    const loader::TensorInfo& need(const std::string& n) const {
        auto* t = find(n); if (!t) throw std::runtime_error("cvqvae: missing tensor '" + n + "'"); return *t;
    }
    std::vector<float> floats(const std::string& n, size_t expect = 0) const {
        const auto& t = need(n);
        size_t cnt = 1; for (auto d : t.shape) cnt *= (size_t)d;
        if (expect && cnt != expect)
            throw std::runtime_error("cvqvae: tensor '" + n + "' has " + std::to_string(cnt) + " values, expected " + std::to_string(expect));
        Linear tmp; tmp.borrow(wtype_of(t.type), t.data_ptr, 1, cnt);
        std::vector<float> out(cnt); tmp.dequant_row(0, out.data()); return out;
    }
    // GGUF 2-D shape is [K(in), N(out)]
    Linear linear(const std::string& stem, size_t expect_k, size_t expect_n) const {
        const auto& w = need(stem + ".weight");
        if (w.shape.size() != 2) throw std::runtime_error("cvqvae: " + stem + ".weight must be 2-D");
        Linear L; L.borrow(wtype_of(w.type), w.data_ptr, (size_t)w.shape[1], (size_t)w.shape[0]);
        if (L.K != expect_k || L.N != expect_n)
            throw std::runtime_error("cvqvae: " + stem + ".weight has shape [K=" + std::to_string(L.K) + ",N=" + std::to_string(L.N) +
                                     "], expected K=" + std::to_string(expect_k) + " N=" + std::to_string(expect_n));
        auto bv = floats(stem + ".bias", L.N);
        L.set_bias_copy(bv.data(), bv.size());
        return L;
    }
};

// action_names is stored as a JSON array of plain strings
inline std::vector<std::string> parse_json_strings(const std::string& s) {
    std::vector<std::string> out; std::string cur; bool in = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char ch = s[i];
        if (!in) { if (ch == '"') { in = true; cur.clear(); } continue; }
        if (ch == '\\' && i + 1 < s.size()) { cur += s[++i]; continue; }
        if (ch == '"') { out.push_back(cur); in = false; continue; }
        cur += ch;
    }
    return out;
}
}  // namespace detail

struct Options {
    bool   convert = false;                 // opt-in lossy re-encode of weights
    WType  weights = WType::F16;            // target type when convert==true (F16 or BF16)
    util::ThreadPool* pool = nullptr;       // helps the big GEMMs; nullptr = single thread
};

// the 30-channel NERVA action vector: these are 0/1 flags, predicted as logits
inline constexpr int kFlagChannels[6] = {8, 13, 17, 21, 25, 29};

class CVQVAE {
public:
    size_t window = 64, hidden = 528, emb = 128, code_dim = 256, num_codes = 512;
    size_t downs = 2, n_res = 3, n_actions = 0, channels = 30, groups = 8;
    bool   cond_enc = true;
    float  dt = 1.f / 30.f;
    std::vector<std::string> action_names;

    void load(const loader::GGUFLoader& g, const Options& opt = {}) {
        opt_ = opt;
        if (g.meta_string("general.architecture", "") != "nanity-cvqvae")
            throw std::runtime_error("cvqvae: general.architecture != 'nanity-cvqvae'");
        if (opt.convert && opt.weights != WType::F16 && opt.weights != WType::BF16)
            throw std::runtime_error("cvqvae: --convert supports F16/BF16 only (Q8_0 needs K%32==0; hidden*3 is not)");
        auto num = [&](const char* k, double d) { return g.meta_number(std::string("nanity.cvqvae.") + k, d); };
        window = (size_t)num("window", 64);   hidden = (size_t)num("hidden", 528);  emb = (size_t)num("emb", 128);
        code_dim = (size_t)num("code_dim", 256); num_codes = (size_t)num("num_codes", 512);
        downs = (size_t)num("downs", 2); n_res = (size_t)num("n_res", 3); n_actions = (size_t)num("n_actions", 0);
        channels = (size_t)num("channels", 30); groups = (size_t)num("groups", 8);
        cond_enc = num("cond_enc", 1) != 0; dt = (float)num("dt", 1.0 / 30.0);
        if (!n_actions || !hidden || hidden % groups) throw std::runtime_error("cvqvae: bad metadata (n_actions/hidden/groups)");
        action_names = detail::parse_json_strings(g.meta_string("nanity.cvqvae.action_names", ""));

        detail::TensorFile tf(g);
        act_emb_ = tf.floats("act_emb.weight", (n_actions + 1) * emb);
        cm0_ = tf.linear("cond_mlp.0", emb, emb); cm1_ = tf.linear("cond_mlp.2", emb, emb);
        in_scale_ = tf.floats("in_scale", channels); in_shift_ = tf.floats("in_shift", channels);
        mean_ = tf.floats("norm.mean", channels); std_ = tf.floats("norm.std", channels);

        auto mk = [&](const std::string& stem, size_t cin, size_t cout, size_t k, size_t dil, size_t stride, size_t pad) {
            Conv c; c.L = tf.linear(stem, k * cin, cout);
            c.cin = cin; c.cout = cout; c.k = k; c.dil = dil; c.stride = stride; c.pad = pad; return c;
        };
        auto mkblocks = [&](const std::string& pre) {
            std::vector<Block> v;
            for (size_t b = 0; b < n_res; ++b) {
                const std::string p = pre + "." + std::to_string(b);
                size_t dil = 1; for (size_t i = 0; i < b % 3; ++i) dil *= 3;        // 3 ** (i % 3)
                Block B; B.gw = tf.floats(p + ".norm.weight", hidden); B.gb = tf.floats(p + ".norm.bias", hidden);
                B.film = tf.linear(p + ".film", emb, 2 * hidden);
                B.c1 = mk(p + ".conv1", hidden, hidden, 3, dil, 1, dil);
                B.c2 = mk(p + ".conv2", hidden, hidden, 1, 1, 1, 0);
                v.push_back(std::move(B));
            }
            return v;
        };
        enc_in_ = mk("enc.inp", channels, hidden, 3, 1, 1, 1);
        dec_in_ = mk("dec.inp", code_dim, hidden, 3, 1, 1, 1);
        for (size_t s = 0; s < downs; ++s) {
            Stage e; e.blocks = mkblocks("enc.stages." + std::to_string(s) + ".0");
            e.resample = mk("enc.stages." + std::to_string(s) + ".1", hidden, hidden, 4, 1, 2, 1);
            enc_.push_back(std::move(e));
            Stage d; d.blocks = mkblocks("dec.stages." + std::to_string(s) + ".0");
            d.resample = mk("dec.stages." + std::to_string(s) + ".1", hidden, hidden, 3, 1, 1, 1);
            dec_.push_back(std::move(d));
        }
        enc_nw_ = tf.floats("enc.norm.weight", hidden); enc_nb_ = tf.floats("enc.norm.bias", hidden);
        dec_nw_ = tf.floats("dec.norm.weight", hidden); dec_nb_ = tf.floats("dec.norm.bias", hidden);
        enc_out_ = mk("enc.out", hidden, code_dim, 3, 1, 1, 1);
        dec_out_ = mk("dec.out", hidden, channels, 3, 1, 1, 1);

        // codebook: kept F32 always (nearest-code lookup must not be perturbed)
        const auto& et = tf.need("vq.embed");
        if (et.type != loader::GGMLType::F32 || et.shape.size() != 2 || et.shape[0] != code_dim || et.shape[1] != num_codes)
            throw std::runtime_error("cvqvae: vq.embed must be F32 [code_dim, num_codes]");
        embed_ = reinterpret_cast<const float*>(et.data_ptr);
        embed_lin_.borrow(WType::F32, et.data_ptr, num_codes, code_dim);
        esq_.assign(num_codes, 0.f);
        for (size_t j = 0; j < num_codes; ++j) { float s = 0.f; for (size_t d = 0; d < code_dim; ++d) s += embed_[j * code_dim + d] * embed_[j * code_dim + d]; esq_[j] = s; }

        if (opt_.convert) {
            auto cv = [&](Linear& l) { l.convert_to(opt_.weights); };
            cv(cm0_); cv(cm1_);
            for (auto* c : {&enc_in_, &dec_in_, &enc_out_, &dec_out_}) cv(c->L);
            for (auto* st : {&enc_, &dec_}) for (auto& s : *st) {
                cv(s.resample.L);
                for (auto& b : s.blocks) { cv(b.film); cv(b.c1.L); cv(b.c2.L); }
            }
        }
        cond_.assign(emb, 0.f); cond_tmp_.assign(emb, 0.f); film_.assign(2 * hidden, 0.f);
    }

    // ---- conditioning ----
    // action: index in [0,n_actions) or -1 for the learned "null" label (unconditional)
    int action_index(const std::string& name) const {
        for (size_t i = 0; i < action_names.size(); ++i) if (action_names[i] == name) return (int)i;
        return -2;   // not found
    }
    size_t null_index() const { return n_actions; }

    // ---- normalisation (physical -> model space) ----
    // x_phys [T][channels] in canonicalised physical units (flags 0/1) -> (x-mean)/std
    void normalize(const float* phys, float* out, size_t T) const {
        for (size_t t = 0; t < T; ++t)
            for (size_t c = 0; c < channels; ++c)
                out[t * channels + c] = (phys[t * channels + c] - mean_[c]) / std_[c];
    }
    // model output [T][channels] (continuous = normalised, flags = logits) -> physical motion, flags thresholded to 0/1
    void to_motion(const float* y, float* phys, size_t T) const {
        for (size_t t = 0; t < T; ++t)
            for (size_t c = 0; c < channels; ++c)
                phys[t * channels + c] = y[t * channels + c] * std_[c] + mean_[c];
        if (channels == 30)
            for (size_t t = 0; t < T; ++t)
                for (int f : kFlagChannels) phys[t * channels + (size_t)f] = y[t * channels + (size_t)f] > 0.f ? 1.f : 0.f;
    }

    // ---- encoder: x_norm [T][channels] -> pre-quantisation latents z [N][code_dim], N = T >> downs ----
    void encode(const float* x_norm, size_t T, int action, float* z) {
        check_len(T);
        set_cond(cond_enc ? action : -1);
        ensure(T);
        in_.resize(T * channels);
        for (size_t t = 0; t < T; ++t)
            for (size_t c = 0; c < channels; ++c)
                in_[t * channels + c] = x_norm[t * channels + c] * in_scale_[c] + in_shift_[c];
        conv1d(enc_in_, in_.data(), T, cur_.data(), Act::None);
        size_t len = T;
        for (auto& st : enc_) {
            for (auto& b : st.blocks) film_res(b, cur_.data(), len);
            conv1d(st.resample, cur_.data(), len, nxt_.data(), Act::None);
            len /= 2; std::swap(cur_, nxt_);
        }
        group_norm(cur_.data(), nxt_.data(), len, enc_nw_.data(), enc_nb_.data(), nullptr, nullptr);
        nn::apply_act(Act::SiLU, nxt_.data(), len * hidden);
        conv1d(enc_out_, nxt_.data(), len, z, Act::None);
    }

    // exact nearest codebook row (squared L2; first index wins ties, like argmin)
    void quantize(const float* z, size_t N, int32_t* idx) {
        score_.resize(num_codes);
        for (size_t n = 0; n < N; ++n) {
            nn::linear(embed_lin_, z + n * code_dim, 1, code_dim, score_.data(), num_codes, Act::None, nullptr);
            size_t best = 0; float bv = std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < num_codes; ++j) {
                float d = esq_[j] - 2.f * score_[j];
                if (d < bv) { bv = d; best = j; }
            }
            idx[n] = (int32_t)best;
        }
    }

    void tokens(const float* x_norm, size_t T, int action, int32_t* idx) {
        const size_t N = T >> downs;
        zbuf_.resize(N * code_dim);
        encode(x_norm, T, action, zbuf_.data());
        quantize(zbuf_.data(), N, idx);
    }

    // ---- decoder: code indices [N] -> model output y [N<<downs][channels] ----
    void decode_codes(const int32_t* idx, size_t N, int action, float* y) {
        zbuf_.resize(N * code_dim);
        for (size_t n = 0; n < N; ++n) {
            if (idx[n] < 0 || (size_t)idx[n] >= num_codes) throw std::runtime_error("cvqvae: code index out of range");
            std::memcpy(zbuf_.data() + n * code_dim, embed_ + (size_t)idx[n] * code_dim, code_dim * sizeof(float));
        }
        decode_latents(zbuf_.data(), N, action, y);
    }

    // decoder on arbitrary latents z [N][code_dim] (e.g. straight from encode(), or a prior's samples)
    void decode_latents(const float* z, size_t N, int action, float* y) {
        set_cond(action);
        const size_t Tmax = N << downs;
        ensure(Tmax);
        conv1d(dec_in_, z, N, cur_.data(), Act::None);
        size_t len = N;
        for (auto& st : dec_) {
            for (auto& b : st.blocks) film_res(b, cur_.data(), len);
            for (size_t t = 0; t < len; ++t) {                         // nearest-neighbour x2
                std::memcpy(nxt_.data() + (2 * t) * hidden,     cur_.data() + t * hidden, hidden * sizeof(float));
                std::memcpy(nxt_.data() + (2 * t + 1) * hidden, cur_.data() + t * hidden, hidden * sizeof(float));
            }
            len *= 2;
            conv1d(st.resample, nxt_.data(), len, cur_.data(), Act::None);
        }
        group_norm(cur_.data(), nxt_.data(), len, dec_nw_.data(), dec_nb_.data(), nullptr, nullptr);
        nn::apply_act(Act::SiLU, nxt_.data(), len * hidden);
        conv1d(dec_out_, nxt_.data(), len, y, Act::None);
    }

    const float* codebook() const { return embed_; }
    const std::vector<float>& mean() const { return mean_; }
    const std::vector<float>& stdev() const { return std_; }

private:
    struct Conv  { Linear L; size_t cin = 0, cout = 0, k = 0, dil = 1, stride = 1, pad = 0; };
    struct Block { std::vector<float> gw, gb; Linear film; Conv c1, c2; };
    struct Stage { std::vector<Block> blocks; Conv resample; };

    Options opt_;
    std::vector<float> act_emb_, in_scale_, in_shift_, mean_, std_;
    Linear cm0_, cm1_;
    Conv enc_in_, dec_in_, enc_out_, dec_out_;
    std::vector<Stage> enc_, dec_;
    std::vector<float> enc_nw_, enc_nb_, dec_nw_, dec_nb_;
    const float* embed_ = nullptr; Linear embed_lin_; std::vector<float> esq_;
    // scratch
    std::vector<float> cond_, cond_tmp_, film_, in_, cur_, nxt_, a_, b_, col_, zbuf_, score_;

    void check_len(size_t T) const {
        if (T == 0 || (T & ((size_t(1) << downs) - 1)))
            throw std::runtime_error("cvqvae: frame count must be a positive multiple of " + std::to_string(size_t(1) << downs));
    }
    void ensure(size_t T) {                                            // sized for the longest activation: T frames x hidden
        const size_t n = T * hidden;
        if (cur_.size() < n) { cur_.assign(n, 0.f); nxt_.assign(n, 0.f); a_.assign(n, 0.f); b_.assign(n, 0.f); }
    }
    void set_cond(int action) {
        size_t i = (action < 0 || (size_t)action >= n_actions) ? n_actions : (size_t)action;
        nn::linear(cm0_, act_emb_.data() + i * emb, 1, emb, cond_tmp_.data(), emb, Act::SiLU, nullptr);
        nn::linear(cm1_, cond_tmp_.data(), 1, emb, cond_.data(), emb, Act::None, nullptr);
    }

    // y[Tout][cout] = act(conv(x[Tin][cin])) ; zero padding, dilation, stride as in nn.Conv1d
    void conv1d(const Conv& c, const float* x, size_t Tin, float* y, Act act) {
        const long span = (long)c.dil * (long)(c.k - 1) + 1;
        const size_t Tout = (size_t)(((long)Tin + 2 * (long)c.pad - span) / (long)c.stride + 1);
        if (c.k == 1 && c.stride == 1 && c.pad == 0) { nn::linear(c.L, x, Tin, c.cin, y, c.cout, act, opt_.pool); return; }
        const size_t K = c.k * c.cin;
        if (col_.size() < Tout * K) col_.assign(Tout * K, 0.f);
        for (size_t t = 0; t < Tout; ++t)
            for (size_t j = 0; j < c.k; ++j) {
                const long s = (long)t * (long)c.stride - (long)c.pad + (long)j * (long)c.dil;
                float* dst = col_.data() + t * K + j * c.cin;
                if (s < 0 || s >= (long)Tin) std::memset(dst, 0, c.cin * sizeof(float));
                else std::memcpy(dst, x + (size_t)s * c.cin, c.cin * sizeof(float));
            }
        nn::linear(c.L, col_.data(), Tout, K, y, c.cout, act, opt_.pool);
    }

    // GroupNorm(groups) over [T][hidden] with optional FiLM: out = norm(x) * (1 + s) + b
    void group_norm(const float* x, float* out, size_t T, const float* gw, const float* gb, const float* s, const float* b) const {
        const size_t C = hidden, cpg = C / groups;
        for (size_t g = 0; g < groups; ++g) {
            const size_t c0 = g * cpg;
            double sum = 0.0, sq = 0.0;
            for (size_t t = 0; t < T; ++t) {
                const float* p = x + t * C + c0;
                for (size_t i = 0; i < cpg; ++i) { double v = p[i]; sum += v; sq += v * v; }
            }
            const double n = (double)(T * cpg), mean = sum / n;
            const double var = std::max(0.0, sq / n - mean * mean);
            const float m = (float)mean, rstd = (float)(1.0 / std::sqrt(var + 1e-5));
            for (size_t t = 0; t < T; ++t) {
                const float* p = x + t * C + c0; float* o = out + t * C + c0;
                for (size_t i = 0; i < cpg; ++i) {
                    float v = (p[i] - m) * rstd * gw[c0 + i] + gb[c0 + i];
                    if (s) v = v * (1.f + s[c0 + i]) + b[c0 + i];
                    o[i] = v;
                }
            }
        }
    }

    // x <- x + conv2(SiLU(conv1(SiLU(GN(x)*(1+s)+b))))   (dropout is inference-off)
    void film_res(const Block& B, float* x, size_t T) {
        nn::linear(B.film, cond_.data(), 1, emb, film_.data(), 2 * hidden, Act::None, nullptr);
        group_norm(x, a_.data(), T, B.gw.data(), B.gb.data(), film_.data(), film_.data() + hidden);
        nn::apply_act(Act::SiLU, a_.data(), T * hidden);
        conv1d(B.c1, a_.data(), T, b_.data(), Act::SiLU);
        conv1d(B.c2, b_.data(), T, a_.data(), Act::None);
        nn::add_inplace(x, a_.data(), T * hidden);
    }
};

}  // namespace cvqvae
