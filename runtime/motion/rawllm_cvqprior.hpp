#pragma once
// =============================================================================
// rawllm_cvqprior.hpp — autoregressive PRIOR over the cVQ-VAE's codes (inference)
//
// Implements the inference graph of CodePrior in tools/train_cvqvae.py (NERVA repo):
//
//   token ids  codes 0..K-1 | SEP = K | action tokens K+1..K+n_actions | NULL = K+1+n_actions | PAD
//   sequence   [ACTION] c1..cn [SEP] c1..cn [SEP] ...     (up to `windows` consecutive windows)
//   x          = tok_emb[id] + pos_emb[position]
//   block      : x += proj(attn(LN1(x)))          causal multi-head self attention, qkv = Linear(d, 3d)
//                x += ff2(GELU_erf(ff0(LN2(x))))
//   logits     = head(LN(x))                      [K+1] (codes + SEP; no bias)
//
// The class owns a KV cache, so sampling a window costs n single-token steps instead of n
// full forward passes. Positions are absolute inside the context (learned embeddings), so when the
// context is full the OLDEST window is dropped (the action prefix stays) and the remaining tokens are
// re-fed in one batch — exactly the trainer's generate_codes(), just without recomputing every step.
//
// Sampling mirrors train_cvqvae.py: logits / max(temperature, 1e-4), then top-k (ties at the k-th
// value are kept, like torch.topk's threshold), then top-p ("cumsum - p > top_p" is removed, the first
// token always survives), softmax, draw. Only the K code logits are used; SEP is appended by force.
//
// Exactness: F32 everywhere and matches PyTorch up to float summation order. Options::convert (F32 ->
// F16/BF16 re-encode of the big matrices at load) is the only lossy knob, off by default.
// Not thread-safe (state and scratch are members). The RNG is the caller's, so a seed reproduces a run.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../nn/rawllm_nn.hpp"
#include "../core/rawllm_loader.hpp"

namespace cvqprior {

using nn::Act;
using nn::Linear;
using nn::WType;

namespace detail {
inline WType wtype_of(loader::GGMLType t) {
    switch (t) {
        case loader::GGMLType::F32:  return WType::F32;
        case loader::GGMLType::F16:  return WType::F16;
        case loader::GGMLType::BF16: return WType::BF16;
        default: throw std::runtime_error("cvqprior: unsupported tensor type (use F32/F16/BF16)");
    }
}
struct TensorFile {
    const loader::GGUFLoader& g;
    explicit TensorFile(const loader::GGUFLoader& gg) : g(gg) {}
    const loader::TensorInfo& need(const std::string& n) const {
        for (auto& t : g.tensors) if (t.name == n) return t;
        throw std::runtime_error("cvqprior: missing tensor '" + n + "'");
    }
    std::vector<float> floats(const std::string& n, size_t expect) const {
        const auto& t = need(n);
        size_t cnt = 1; for (auto d : t.shape) cnt *= (size_t)d;
        if (cnt != expect)
            throw std::runtime_error("cvqprior: tensor '" + n + "' has " + std::to_string(cnt) + " values, expected " + std::to_string(expect));
        Linear tmp; tmp.borrow(wtype_of(t.type), t.data_ptr, 1, cnt);
        std::vector<float> out(cnt); tmp.dequant_row(0, out.data()); return out;
    }
    // GGUF 2-D shape is [K(in), N(out)]
    Linear linear(const std::string& stem, size_t expect_k, size_t expect_n, bool bias) const {
        const auto& w = need(stem + ".weight");
        if (w.shape.size() != 2) throw std::runtime_error("cvqprior: " + stem + ".weight must be 2-D");
        Linear L; L.borrow(wtype_of(w.type), w.data_ptr, (size_t)w.shape[1], (size_t)w.shape[0]);
        if (L.K != expect_k || L.N != expect_n)
            throw std::runtime_error("cvqprior: " + stem + ".weight has shape [K=" + std::to_string(L.K) + ",N=" + std::to_string(L.N) +
                                     "], expected K=" + std::to_string(expect_k) + " N=" + std::to_string(expect_n));
        if (bias) { auto bv = floats(stem + ".bias", L.N); L.set_bias_copy(bv.data(), bv.size()); }
        return L;
    }
};
inline std::vector<std::string> parse_json_strings(const std::string& s) {      // ["a","b"] -> {a,b}
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
    bool   convert = false;                 // opt-in lossy re-encode of the big matrices
    WType  weights = WType::F16;            // F16 or BF16 when convert==true
    util::ThreadPool* pool = nullptr;       // only used for prefill batches; single-token steps are too small to split
};

struct SampleParams {
    float temperature = 1.0f;
    int   top_k       = 0;                  // 0 = off
    float top_p       = 1.0f;               // >= 1 = off
};

// Returns "" when a prior and a VAE belong together, else a human-readable reason.
inline std::string check_compatible(size_t prior_K, size_t prior_n, size_t prior_actions, const std::vector<std::string>& prior_names,
                                    size_t vae_codes, size_t vae_latent_steps, size_t vae_actions, const std::vector<std::string>& vae_names) {
    if (prior_K != vae_codes)
        return "prior has " + std::to_string(prior_K) + " codes, the VAE codebook has " + std::to_string(vae_codes) + " (not trained together)";
    if (prior_n != vae_latent_steps)
        return "prior makes " + std::to_string(prior_n) + " codes per window, the VAE needs " + std::to_string(vae_latent_steps);
    if (prior_actions != vae_actions)
        return "prior knows " + std::to_string(prior_actions) + " actions, the VAE knows " + std::to_string(vae_actions);
    if (!prior_names.empty() && !vae_names.empty() && prior_names != vae_names)
        return "prior and VAE action lists differ (same count, different names/order)";
    return "";
}

class CodePrior {
public:
    size_t K = 512, n_actions = 0, n = 16, windows = 4, d = 256, layers = 6, heads = 4, stride = 0;
    float  dt = 1.f / 30.f;
    double val_nll = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::string> action_names;

    size_t vocab()   const { return K + 3 + n_actions; }
    size_t seq_len() const { return windows * (n + 1); }
    size_t max_ctx() const { return seq_len() - 1; }                        // rows of the position table
    int32_t sep_token()  const { return (int32_t)K; }
    int32_t null_token() const { return (int32_t)(K + 1 + n_actions); }
    // action: index in [0,n_actions) or -1 / out of range = the unconditional NULL token
    int32_t action_token(int action) const { return (action < 0 || (size_t)action >= n_actions) ? null_token() : (int32_t)(K + 1 + (size_t)action); }
    int action_index(const std::string& name) const {
        for (size_t i = 0; i < action_names.size(); ++i) if (action_names[i] == name) return (int)i;
        return -2;
    }
    size_t params() const { return nparams_; }

    void load(const loader::GGUFLoader& g, const Options& opt = {}) {
        opt_ = opt;
        if (g.meta_string("general.architecture", "") != "nanity-cvqprior")
            throw std::runtime_error("cvqprior: general.architecture != 'nanity-cvqprior'");
        if (opt.convert && opt.weights != WType::F16 && opt.weights != WType::BF16)
            throw std::runtime_error("cvqprior: --convert supports F16/BF16 only");
        auto num = [&](const char* k, double def) { return g.meta_number(std::string("nanity.cvqprior.") + k, def); };
        K = (size_t)num("K", 512); n_actions = (size_t)num("n_actions", 0); n = (size_t)num("n", 16); windows = (size_t)num("windows", 4);
        d = (size_t)num("d", 256); layers = (size_t)num("layers", 6); heads = (size_t)num("heads", 4); stride = (size_t)num("stride", 0);
        dt = (float)num("dt", 1.0 / 30.0); val_nll = num("val_nll", std::numeric_limits<double>::quiet_NaN());
        if (!K || !n || !windows || !d || !layers || !heads || d % heads)
            throw std::runtime_error("cvqprior: bad metadata (K/n/windows/d/layers/heads)");
        hd_ = d / heads;
        action_names = detail::parse_json_strings(g.meta_string("nanity.cvqprior.action_names", ""));
        if (!action_names.empty() && action_names.size() != n_actions)
            throw std::runtime_error("cvqprior: action_names has " + std::to_string(action_names.size()) + " entries, n_actions = " + std::to_string(n_actions));

        detail::TensorFile tf(g);
        tok_ = tf.floats("tok.weight", vocab() * d);
        pos_ = tf.floats("pos.weight", max_ctx() * d);
        blocks_.clear(); blocks_.reserve(layers);
        for (size_t i = 0; i < layers; ++i) {
            const std::string p = "blocks." + std::to_string(i);
            Block b;
            b.n1w = tf.floats(p + ".n1.weight", d); b.n1b = tf.floats(p + ".n1.bias", d);
            b.n2w = tf.floats(p + ".n2.weight", d); b.n2b = tf.floats(p + ".n2.bias", d);
            b.qkv = tf.linear(p + ".qkv", d, 3 * d, true); b.proj = tf.linear(p + ".proj", d, d, true);
            b.ff0 = tf.linear(p + ".ff.0", d, 4 * d, true); b.ff2 = tf.linear(p + ".ff.2", 4 * d, d, true);
            blocks_.push_back(std::move(b));
        }
        lnw_ = tf.floats("ln.weight", d); lnb_ = tf.floats("ln.bias", d);
        head_ = tf.linear("head", d, K + 1, false);

        if (opt_.convert)
            for (auto& b : blocks_) for (auto* l : {&b.qkv, &b.proj, &b.ff0, &b.ff2}) l->convert_to(opt_.weights);
        if (opt_.convert) head_.convert_to(opt_.weights);

        nparams_ = tok_.size() + pos_.size() + lnw_.size() + lnb_.size() + head_.N * head_.K;
        for (auto& b : blocks_) nparams_ += 4 * d + b.qkv.N * b.qkv.K + 3 * d + b.proj.N * b.proj.K + d + b.ff0.N * b.ff0.K + 4 * d + b.ff2.N * b.ff2.K + d;
        kc_.assign(layers, std::vector<float>(max_ctx() * d, 0.f));
        vc_.assign(layers, std::vector<float>(max_ctx() * d, 0.f));
        ctx_.clear(); fed_ = 0; last_.assign(K + 1, 0.f);
    }

    // ---- incremental context -------------------------------------------------------------------------------
    // Start a new sequence: ctx = [ACTION]. Cheap: the cache is simply considered empty.
    void reset(int action) { ctx_.assign(1, action_token(action)); fed_ = 0; }
    size_t length() const { return ctx_.size(); }
    const std::vector<int32_t>& context() const { return ctx_; }
    void push(int32_t tok) {
        if (tok < 0 || (size_t)tok >= vocab()) throw std::runtime_error("cvqprior: token id out of range");
        ctx_.push_back(tok);
    }
    // ctx = [ACTION] + ctx[1+n+1:]   (the oldest window and its SEP go; positions shift, so the cache is rebuilt on demand)
    void drop_oldest_window() {
        if (ctx_.size() < 1 + n + 1) throw std::runtime_error("cvqprior: no complete window to drop");
        ctx_.erase(ctx_.begin() + 1, ctx_.begin() + 1 + (long)(n + 1)); fed_ = 0;
    }
    // Logits [K+1] (codes 0..K-1, then SEP) for the token that follows the current context. Feeds whatever is pending.
    const std::vector<float>& logits() {
        if (ctx_.empty()) throw std::runtime_error("cvqprior: call reset(action) first");
        if (ctx_.size() > max_ctx()) throw std::runtime_error("cvqprior: context longer than the model's position table");
        if (fed_ < ctx_.size()) { forward_chunk(ctx_.data() + fed_, ctx_.size() - fed_, fed_, nullptr, last_.data()); fed_ = ctx_.size(); }
        return last_;
    }
    // Reference path for tests: logits [T][K+1] at EVERY position of an arbitrary token sequence (T <= max_ctx), no cache reuse.
    void forward_all(const int32_t* toks, size_t T, float* out) {
        if (!T || T > max_ctx()) throw std::runtime_error("cvqprior: sequence length must be in [1, " + std::to_string(max_ctx()) + "]");
        for (size_t i = 0; i < T; ++i) if (toks[i] < 0 || (size_t)toks[i] >= vocab()) throw std::runtime_error("cvqprior: token id out of range");
        ctx_.assign(toks, toks + T); fed_ = 0;
        forward_chunk(ctx_.data(), T, 0, out, last_.data()); fed_ = T;
    }

    // ---- sampling -------------------------------------------------------------------------------------------
    // The distribution actually sampled from: temperature, then top-k, then top-p, softmax. probs has K entries.
    void filtered_probs(const float* code_logits, const SampleParams& sp, std::vector<double>& probs) const {
        std::vector<double> lg(K);
        const double inv_t = 1.0 / std::max((double)sp.temperature, 1e-4);
        for (size_t j = 0; j < K; ++j) lg[j] = (double)code_logits[j] * inv_t;
        const double NEG = -std::numeric_limits<double>::infinity();
        if (sp.top_k > 0 && (size_t)sp.top_k < K) {
            std::vector<double> tmp(lg);
            std::nth_element(tmp.begin(), tmp.begin() + (sp.top_k - 1), tmp.end(), std::greater<double>());
            const double kth = tmp[(size_t)sp.top_k - 1];
            for (auto& v : lg) if (v < kth) v = NEG;
        }
        if (sp.top_p < 1.0f) {
            std::vector<size_t> order(K); std::iota(order.begin(), order.end(), (size_t)0);
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lg[a] > lg[b]; });
            double mx = lg[order[0]], sum = 0.0; std::vector<double> pr(K);
            for (size_t i = 0; i < K; ++i) { pr[i] = std::exp(lg[order[i]] - mx); sum += pr[i]; }
            double cum = 0.0; std::vector<double> kept(K, NEG);
            for (size_t i = 0; i < K; ++i) {
                pr[i] /= sum;
                const bool removed = (cum > (double)sp.top_p) && i > 0;       // cumsum - p > top_p  (cum is the sum before this token)
                if (!removed) kept[order[i]] = lg[order[i]];
                cum += pr[i];
            }
            lg.swap(kept);
        }
        double mx = NEG; for (double v : lg) mx = std::max(mx, v);
        probs.assign(K, 0.0); double sum = 0.0;
        for (size_t j = 0; j < K; ++j) { probs[j] = std::isinf(lg[j]) ? 0.0 : std::exp(lg[j] - mx); sum += probs[j]; }
        for (auto& p : probs) p /= sum;
    }

    // Samples ONE code for the current context (does not append it; call push()).
    // p_raw = the prior's own probability (temperature 1, unfiltered) of the chosen code; entropy in nats of that raw distribution.
    int32_t sample_code(std::mt19937_64& rng, const SampleParams& sp, float* p_raw = nullptr, float* entropy = nullptr) {
        const std::vector<float>& lg = logits();
        std::vector<double> probs; filtered_probs(lg.data(), sp, probs);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const double u = U(rng); double c = 0.0; size_t pick = K - 1;
        for (size_t j = 0; j < K; ++j) { c += probs[j]; if (u < c) { pick = j; break; } }
        if (probs[pick] == 0.0) { pick = (size_t)(std::max_element(probs.begin(), probs.end()) - probs.begin()); }   // rounding at the tail of the cdf
        if (p_raw || entropy) {
            double mx = -std::numeric_limits<double>::infinity(); for (size_t j = 0; j < K; ++j) mx = std::max(mx, (double)lg[j]);
            double z = 0.0; for (size_t j = 0; j < K; ++j) z += std::exp((double)lg[j] - mx);
            double ent = 0.0; for (size_t j = 0; j < K; ++j) { double p = std::exp((double)lg[j] - mx) / z; ent -= p * std::log(p + 1e-12); }
            if (p_raw) *p_raw = (float)(std::exp((double)lg[pick] - mx) / z);
            if (entropy) *entropy = (float)ent;
        }
        return (int32_t)pick;
    }

    // Continues the current sequence by one window: n codes, each pushed to the context. Handles the SEP between windows and
    // the sliding context: call it repeatedly after reset(); `windows_done` is how many windows this sequence already has.
    void next_window(size_t windows_done, std::mt19937_64& rng, const SampleParams& sp, int32_t* codes_out, float* p_raw_out = nullptr, float* entropy_out = nullptr) {
        if (windows_done > 0) push(sep_token());
        if (windows_done >= windows) drop_oldest_window();
        for (size_t i = 0; i < n; ++i) {
            const int32_t c = sample_code(rng, sp, p_raw_out ? p_raw_out + i : nullptr, entropy_out ? entropy_out + i : nullptr);
            push(c); codes_out[i] = c;
        }
    }

    // Whole sequence in one call: n_windows * n codes (row-major [window][code]).
    void generate(int action, size_t n_windows, const SampleParams& sp, std::mt19937_64& rng, int32_t* out, float* p_raw = nullptr) {
        reset(action);
        for (size_t w = 0; w < n_windows; ++w) next_window(w, rng, sp, out + w * n, p_raw ? p_raw + w * n : nullptr, nullptr);
    }

private:
    struct Block { std::vector<float> n1w, n1b, n2w, n2b; Linear qkv, proj, ff0, ff2; };

    Options opt_;
    size_t hd_ = 64, nparams_ = 0, fed_ = 0;
    std::vector<float> tok_, pos_, lnw_, lnb_;
    std::vector<Block> blocks_;
    Linear head_;
    std::vector<std::vector<float>> kc_, vc_;      // [layer][position][d]  (head h lives at columns h*hd .. h*hd+hd)
    std::vector<int32_t> ctx_;
    std::vector<float> last_, x_, h_, qkv_, att_, tmp_, ff_, sc_, rows_;

    // Runs M new tokens at positions t0..t0+M-1 through every block (their K/V are appended to the cache and they attend to the cache
    // plus each other, causally). Writes the logits of ALL M rows to out_all ([M][K+1]) if given, and the last row's to out_last.
    void forward_chunk(const int32_t* toks, size_t M, size_t t0, float* out_all, float* out_last) {
        if (t0 + M > max_ctx()) throw std::runtime_error("cvqprior: context overflow");
        const size_t D = d;
        x_.resize(M * D); h_.resize(M * D); qkv_.resize(M * 3 * D); att_.resize(M * D); tmp_.resize(M * D); ff_.resize(M * 4 * D); sc_.resize(max_ctx());
        for (size_t i = 0; i < M; ++i) {
            const float* te = tok_.data() + (size_t)toks[i] * D; const float* pe = pos_.data() + (t0 + i) * D;
            for (size_t j = 0; j < D; ++j) x_[i * D + j] = te[j] + pe[j];
        }
        const float scale = 1.f / std::sqrt((float)hd_);
        for (size_t l = 0; l < layers; ++l) {
            Block& B = blocks_[l];
            std::memcpy(h_.data(), x_.data(), M * D * sizeof(float));
            for (size_t i = 0; i < M; ++i) nn::layernorm_row(h_.data() + i * D, D, B.n1w.data(), B.n1b.data(), 1e-5f);
            nn::linear(B.qkv, h_.data(), M, D, qkv_.data(), 3 * D, Act::None, opt_.pool);
            for (size_t i = 0; i < M; ++i) {
                std::memcpy(kc_[l].data() + (t0 + i) * D, qkv_.data() + i * 3 * D + D,     D * sizeof(float));
                std::memcpy(vc_[l].data() + (t0 + i) * D, qkv_.data() + i * 3 * D + 2 * D, D * sizeof(float));
            }
            for (size_t i = 0; i < M; ++i) {
                const size_t len = t0 + i + 1;                                // causal: positions 0..t0+i
                for (size_t h = 0; h < heads; ++h) {
                    const float* q = qkv_.data() + i * 3 * D + h * hd_;
                    for (size_t j = 0; j < len; ++j) sc_[j] = nn::dot_f32(q, kc_[l].data() + j * D + h * hd_, hd_);
                    nn::softmax_scaled(sc_.data(), len, scale);
                    float* o = att_.data() + i * D + h * hd_;
                    for (size_t e = 0; e < hd_; ++e) o[e] = 0.f;
                    for (size_t j = 0; j < len; ++j) {
                        const float p = sc_[j]; const float* v = vc_[l].data() + j * D + h * hd_;
                        for (size_t e = 0; e < hd_; ++e) o[e] += p * v[e];
                    }
                }
            }
            nn::linear(B.proj, att_.data(), M, D, tmp_.data(), D, Act::None, opt_.pool);
            nn::add_inplace(x_.data(), tmp_.data(), M * D);
            std::memcpy(h_.data(), x_.data(), M * D * sizeof(float));
            for (size_t i = 0; i < M; ++i) nn::layernorm_row(h_.data() + i * D, D, B.n2w.data(), B.n2b.data(), 1e-5f);
            nn::linear(B.ff0, h_.data(), M, D, ff_.data(), 4 * D, Act::GeluErf, opt_.pool);
            nn::linear(B.ff2, ff_.data(), M, 4 * D, tmp_.data(), D, Act::None, opt_.pool);
            nn::add_inplace(x_.data(), tmp_.data(), M * D);
        }
        const size_t V = K + 1;
        if (out_all) {
            std::memcpy(h_.data(), x_.data(), M * D * sizeof(float));
            for (size_t i = 0; i < M; ++i) nn::layernorm_row(h_.data() + i * D, D, lnw_.data(), lnb_.data(), 1e-5f);
            nn::linear(head_, h_.data(), M, D, out_all, V, Act::None, opt_.pool);
            if (out_last) std::memcpy(out_last, out_all + (M - 1) * V, V * sizeof(float));
        } else {
            std::memcpy(h_.data(), x_.data() + (M - 1) * D, D * sizeof(float));
            nn::layernorm_row(h_.data(), D, lnw_.data(), lnb_.data(), 1e-5f);
            nn::linear(head_, h_.data(), 1, D, out_last, V, Act::None, nullptr);
        }
    }
};

}  // namespace cvqprior
