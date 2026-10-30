// cvqprior_cli.cpp — standalone driver for the NEON cVQ-VAE prior (rawllm_cvqprior.hpp).
//
// Build (from the repo root):
//   g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c runtime/motion/cvqprior_cli.cpp -o cvqprior
//
// Usage:
//   cvqprior PRIOR.gguf info
//   cvqprior PRIOR.gguf sample [--vae VAE.gguf] [--action NAME|IDX|null] [--windows N] [--temperature T] [--top-k K] [--top-p P]
//                              [--seed S] [--out codes.txt] [--motion-out motion.csv]
//        -> one line of codes per window ("c c c ..."); with --vae the windows are also decoded (normalised -> physical units)
//           and written to --motion-out as CSV, windows back to back (stitching/cross-fading is NERVA's job: see its `stride`)
//   cvqprior PRIOR.gguf logits --tokens 3,7,... [--action ...]     all-position logits of a raw token sequence, CSV [T][K+1]
//        (token ids are the trainer's: codes < K, SEP = K, actions K+1.., NULL = K+1+n_actions; --tokens is the FULL sequence incl. the action token)
//   cvqprior PRIOR.gguf cfglogits --tokens 3,7,... [--goals "dyaw,dx,dz,speed;..."] [--style S] [--cfg X]   next-token logits of the FULL incremental path (KV cache, null/conditioned
//        branches, guidance): one CSV row [K+1] for the position after the last token; --goals = RAW numbers, one entry per window the tokens touch
//   cvqprior PRIOR.gguf probs  --tokens 3,7,... [--temperature T] [--top-k K] [--top-p P]      the filtered next-token distribution, one line
//   cvqprior PRIOR.gguf selftest                                   KV-cache path vs full forward on random tokens (exit 0 = identical to ~1e-5)
//   cvqprior PRIOR.gguf bench [--iters N]
// Goal-conditioned priors: sample / logits / probs / cfglogits also take  --goal "a,b,c,d" (one raw goal for every window) or --goals "a,b,c,d;a,b,c,d;..." (one per window,
//   the last repeats; for logits/probs the unspecified windows are zeros in normalised space like the trainer),  --style NAME|IDX|null,  --cfg SCALE.
// Options anywhere after MODEL: --threads N   --convert f16|bf16 (lossy, opt-in)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>

#include "rawllm_cvqprior.hpp"
#include "rawllm_cvqvae.hpp"

using namespace cvqprior;

static std::vector<int32_t> parse_ints(std::string s) {
    for (char& c : s) if (c == ',' || c == ';' || c == '\n' || c == '\t') c = ' ';
    std::istringstream ss(s); std::vector<int32_t> v; long x;
    while (ss >> x) v.push_back((int32_t)x);
    return v;
}
static std::vector<std::vector<float>> parse_goals(const std::string& s) {     // "a,b,c,d;a,b,c,d" -> rows
    std::vector<std::vector<float>> rows; std::string cur;
    auto flush = [&]() { if (cur.find_first_not_of(" \t,") == std::string::npos) { cur.clear(); return; }
        std::string t = cur; for (char& c : t) if (c == ',') c = ' ';
        std::istringstream ss(t); std::vector<float> r; double x; while (ss >> x) r.push_back((float)x); rows.push_back(r); cur.clear(); };
    for (char c : s) { if (c == ';') flush(); else cur += c; }
    flush(); return rows;
}
static void write_csv(std::ostream& o, const float* v, size_t rows, size_t cols) {
    char buf[32];
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) { std::snprintf(buf, sizeof buf, "%.9g", v[r * cols + c]); o << (c ? "," : "") << buf; }
        o << "\n";
    }
}

int main(int argc, char** argv) try {
    if (argc < 3) { std::fprintf(stderr, "usage: %s PRIOR.gguf (info|sample|logits|probs|selftest|bench) [options]   (see header of cvqprior_cli.cpp)\n", argv[0]); return 2; }
    const std::string model_path = argv[1], cmd = argv[2];
    std::string vae_path, out, motion_out, tokens, action_s = "null", convert, goals_s, goal_s, style_s = "null"; float cfg = 1.f;
    SampleParams sp; size_t windows = 4; long long seed = 1; int threads = 0, iters = 5;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--vae") vae_path = val(); else if (a == "--out") out = val(); else if (a == "--motion-out") motion_out = val();
        else if (a == "--tokens") tokens = val(); else if (a == "--action") action_s = val();
        else if (a == "--windows") windows = (size_t)std::atol(val().c_str()); else if (a == "--temperature") sp.temperature = (float)std::atof(val().c_str());
        else if (a == "--top-k") sp.top_k = std::atoi(val().c_str()); else if (a == "--top-p") sp.top_p = (float)std::atof(val().c_str());
        else if (a == "--seed") seed = std::atoll(val().c_str()); else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--convert") convert = val(); else if (a == "--iters") iters = std::atoi(val().c_str());
        else if (a == "--goals") goals_s = val(); else if (a == "--goal") goal_s = val(); else if (a == "--style") style_s = val(); else if (a == "--cfg") cfg = (float)std::atof(val().c_str());
        else throw std::runtime_error("unknown option " + a);
    }

    loader::GGUFLoader g; g.open(model_path);
    Options opt;
    if (!convert.empty()) {
        opt.convert = true;
        if (convert == "f16") opt.weights = WType::F16; else if (convert == "bf16") opt.weights = WType::BF16;
        else throw std::runtime_error("--convert must be f16 or bf16");
    }
    std::unique_ptr<util::ThreadPool> pool;
    if (threads > 1) { pool = std::make_unique<util::ThreadPool>(threads - 1); opt.pool = pool.get(); }
    CodePrior P; P.load(g, opt);

    int action = -1;
    if (action_s != "null") {
        char* end = nullptr; long x = std::strtol(action_s.c_str(), &end, 10);
        if (end && *end == 0 && !action_s.empty()) action = (int)x;
        else { action = P.action_index(action_s); if (action < 0) throw std::runtime_error("unknown action '" + action_s + "' (run `info` for the list)"); }
        if (action >= (int)P.n_actions) throw std::runtime_error("action index out of range");
    }
    int style = -1;
    if (style_s != "null") {
        if (!P.has_style()) throw std::runtime_error("this prior was trained without style ids");
        char* end = nullptr; long x = std::strtol(style_s.c_str(), &end, 10);
        if (end && *end == 0 && !style_s.empty()) style = (int)x; else { style = P.style_index(style_s); if (style < 0) throw std::runtime_error("unknown style '" + style_s + "'"); }
        if (style >= (int)P.n_styles) throw std::runtime_error("style index out of range");
    }
    std::vector<std::vector<float>> graw = parse_goals(goals_s.empty() ? goal_s : goals_s);
    for (auto& r : graw) if (r.size() != P.goal_dim) throw std::runtime_error("a goal has " + std::to_string(P.goal_dim) + " numbers (a,b,c,d); got " + std::to_string(r.size()));
    if (!graw.empty() && !P.has_goal()) throw std::runtime_error("this prior was trained without goal conditioning");
    const bool one_goal = goals_s.empty() && !goal_s.empty();
    auto sink = [&](const std::string& path, const std::function<void(std::ostream&)>& fn) {
        if (path.empty()) fn(std::cout); else { std::ofstream f(path); if (!f) throw std::runtime_error("cannot write " + path); fn(f); }
    };

    if (cmd == "info") {
        std::printf("cVQ-VAE prior  codes=%zu  per-window=%zu  context=%zu windows (%zu tokens)  d=%zu  layers=%zu  heads=%zu  params=%.2fM  stride=%zu  dt=%.6f",
                    P.K, P.n, P.windows, P.max_ctx(), P.d, P.layers, P.heads, P.params() / 1e6, P.stride, P.dt);
        if (!std::isnan(P.val_nll)) std::printf("  val_nll=%.4f", P.val_nll);
        std::printf("\nactions (%zu + null):", P.n_actions);
        for (size_t i = 0; i < P.action_names.size(); ++i) std::printf(" %zu=%s", i, P.action_names[i].c_str());
        std::printf("\n");
        if (P.has_goal()) {
            std::printf("goal (%zu numbers, raw -> normalised by mean/std):", P.goal_dim);
            for (size_t i = 0; i < P.goal_dim; ++i) std::printf(" %s(mean %.4g std %.4g)", i < P.goal_names.size() ? P.goal_names[i].c_str() : "?", P.goal_mean[i], P.goal_std[i]);
            std::printf("\n  data range: turn_max %.3f rad/window, cruise speed %.3f m/s, max speed %.3f m/s\n", P.turn_max, P.speed_cruise, P.speed_max);
        } else std::printf("goal: none (trained without goal conditioning)\n");
        if (P.has_style()) { std::printf("styles (%zu + null):", P.n_styles); for (size_t i = 0; i < P.style_names.size(); ++i) std::printf(" %zu=%s", i, P.style_names[i].c_str()); std::printf("\n"); }
        else std::printf("styles: none\n");
        return 0;
    }

    if (cmd == "sample") {
        std::mt19937_64 rng((uint64_t)seed);
        std::vector<int32_t> codes(windows * P.n);
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> gflat;
        for (size_t w = 0; w < windows && !graw.empty(); ++w) { const auto& r = graw[one_goal ? 0 : std::min(w, graw.size() - 1)]; gflat.insert(gflat.end(), r.begin(), r.end()); }
        P.set_cfg_scale(cfg);
        P.generate(action, windows, sp, rng, codes.data(), nullptr, gflat.empty() ? nullptr : gflat.data(), style);
        auto t1 = std::chrono::steady_clock::now();
        sink(out, [&](std::ostream& o) { for (size_t w = 0; w < windows; ++w) { for (size_t i = 0; i < P.n; ++i) o << (i ? " " : "") << codes[w * P.n + i]; o << "\n"; } });
        std::fprintf(stderr, "sampled %zu windows x %zu codes in %.1f ms\n", windows, P.n, std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (!vae_path.empty()) {
            loader::GGUFLoader vg; vg.open(vae_path);
            cvqvae::Options vo; vo.pool = opt.pool;
            cvqvae::CVQVAE V; V.load(vg, vo);
            const std::string why = check_compatible(P.K, P.n, P.n_actions, P.action_names, V.num_codes, V.window >> V.downs, V.n_actions, V.action_names);
            if (!why.empty()) throw std::runtime_error("prior and VAE do not match: " + why);
            const size_t T = P.n << V.downs;
            std::vector<float> y(T * V.channels), phys(T * V.channels), all;
            for (size_t w = 0; w < windows; ++w) {
                V.decode_codes(codes.data() + w * P.n, P.n, action, y.data());
                V.to_motion(y.data(), phys.data(), T);
                all.insert(all.end(), phys.begin(), phys.end());
            }
            sink(motion_out, [&](std::ostream& o) { write_csv(o, all.data(), windows * T, V.channels); });
        }
        return 0;
    }

    if (cmd == "logits" || cmd == "probs") {
        auto tk = parse_ints(tokens);
        if (tk.empty()) throw std::runtime_error("--tokens required");
        std::vector<float> lg(tk.size() * (P.K + 1));
        std::vector<float> slots;                                              // normalised per-window goals like the trainer's `goal` input; unspecified windows = zeros
        if (!graw.empty()) { slots.assign(P.windows * P.goal_dim, 0.f); for (size_t j = 0; j < P.windows && j < graw.size(); ++j) P.normalise_goal(graw[one_goal ? 0 : j].data(), slots.data() + j * P.goal_dim);
                             if (one_goal) for (size_t j = 1; j < P.windows; ++j) std::memcpy(slots.data() + j * P.goal_dim, slots.data(), P.goal_dim * sizeof(float)); }
        P.forward_all(tk.data(), tk.size(), lg.data(), slots.empty() ? nullptr : slots.data(), style);
        if (cmd == "logits") { sink(out, [&](std::ostream& o) { write_csv(o, lg.data(), tk.size(), P.K + 1); }); return 0; }
        std::vector<double> pr; P.filtered_probs(lg.data() + (tk.size() - 1) * (P.K + 1), sp, pr);
        std::vector<float> pf(pr.begin(), pr.end());
        sink(out, [&](std::ostream& o) { write_csv(o, pf.data(), 1, P.K); });
        return 0;
    }

    if (cmd == "cfglogits") {
        auto tk = parse_ints(tokens);
        if (tk.empty()) throw std::runtime_error("--tokens required");
        const size_t nslots = (tk.size() - 1) / (P.n + 1) + 1;
        if (!graw.empty() && !one_goal && graw.size() < nslots) throw std::runtime_error("these tokens touch " + std::to_string(nslots) + " windows: give that many goals");
        P.reset(tk[0] - (int)P.K - 1, style); P.set_cfg_scale(cfg);
        for (size_t j = 0; j < nslots; ++j) P.add_window_goal(graw.empty() ? nullptr : graw[one_goal ? 0 : j].data());
        for (size_t i = 1; i < tk.size(); ++i) P.push(tk[i]);
        const std::vector<float> l = P.logits();
        sink(out, [&](std::ostream& o) { write_csv(o, l.data(), 1, P.K + 1); });
        return 0;
    }

    if (cmd == "selftest") {
        std::mt19937_64 rng(7); const size_t T = P.max_ctx();
        std::vector<int32_t> tk(T);
        tk[0] = P.action_token(action);
        for (size_t i = 1; i < T; ++i) tk[i] = (int32_t)(rng() % (P.K + 1));            // codes and SEPs, like a real sequence
        std::vector<float> full(T * (P.K + 1));
        P.forward_all(tk.data(), T, full.data());
        double worst = 0, mag = 0;
        P.reset(action);
        for (size_t i = 0; i < T; ++i) {                                                  // one token at a time through the KV cache
            const auto& l = P.logits();
            for (size_t j = 0; j <= P.K; ++j) { worst = std::max(worst, (double)std::fabs(l[j] - full[i * (P.K + 1) + j])); mag = std::max(mag, (double)std::fabs(full[i * (P.K + 1) + j])); }
            if (i + 1 < T) P.push(tk[i + 1]);
        }
        // chunked: first half as one batch, rest one by one
        P.reset(action); for (size_t i = 1; i <= T / 2; ++i) P.push(tk[i]);
        for (size_t i = T / 2; i < T; ++i) {
            const auto& l = P.logits();
            for (size_t j = 0; j <= P.K; ++j) worst = std::max(worst, (double)std::fabs(l[j] - full[i * (P.K + 1) + j]));
            if (i + 1 < T) P.push(tk[i + 1]);
        }
        const double rel = worst / (mag + 1e-12);
        std::printf("KV cache vs full forward over %zu positions: max abs diff %.3e (relative %.3e)  %s\n", T, worst, rel, rel < 1e-4 ? "OK" : "FAIL");

        // sliding context: sample windows+2 windows (the oldest window is dropped and the cache rebuilt twice) and, at EVERY step,
        // compare the incremental logits with a full forward of exactly the tokens the context holds
        CodePrior R; R.load(g, opt);                                   // second instance = the reference
        std::mt19937_64 srng(11); SampleParams ssp; ssp.top_k = 50;
        std::vector<float> ref(P.K + 1); double worst2 = 0, mag2 = 0; size_t steps = 0;
        P.reset(action);
        std::vector<int32_t> unused(P.n);
        for (size_t w = 0; w < P.windows + 2; ++w) {
            if (w > 0) P.push(P.sep_token());
            if (w >= P.windows) P.drop_oldest_window();
            for (size_t i = 0; i < P.n; ++i) {
                const std::vector<float> l = P.logits();
                std::vector<float> all(P.length() * (P.K + 1));
                R.forward_all(P.context().data(), P.length(), all.data());
                const float* last = all.data() + (P.length() - 1) * (P.K + 1);
                for (size_t j = 0; j <= P.K; ++j) { worst2 = std::max(worst2, (double)std::fabs(l[j] - last[j])); mag2 = std::max(mag2, (double)std::fabs(last[j])); }
                P.push(P.sample_code(srng, ssp)); ++steps;
            }
        }
        const double rel2 = worst2 / (mag2 + 1e-12);
        std::printf("sliding context (%zu windows, %zu steps, 2 cache rebuilds): max abs diff %.3e (relative %.3e)  %s\n", P.windows + 2, steps, worst2, rel2, rel2 < 1e-4 ? "OK" : "FAIL");
        bool ok3 = true;
        if (P.has_goal()) {
            // goal / style / guidance through the sliding context: at every step the incremental logits (conditioned cache, nulled-goal cache, mix) must equal a
            // full forward of exactly the tokens the context holds, with the goals of exactly the windows it holds.
            const int sty = P.has_style() ? (int)(P.n_styles / 2) : -1;
            std::mt19937_64 grng(5); std::normal_distribution<float> nd(0.f, 1.f);
            const size_t NW = P.windows + 2, G = P.goal_dim;
            std::vector<std::vector<float>> gr(NW, std::vector<float>(G));
            for (auto& r : gr) for (size_t i = 0; i < G; ++i) r[i] = P.goal_mean[i] + P.goal_std[i] * nd(grng);
            for (float scale : {1.0f, 1.7f}) {
                std::mt19937_64 s3(13); SampleParams sp3; sp3.top_k = 50; double w3 = 0, m3 = 0; size_t st3 = 0;
                P.reset(action, sty); P.set_cfg_scale(scale);
                std::vector<float> slots(P.windows * G), condall, uncall, mix(P.K + 1);
                for (size_t w = 0; w < NW; ++w) {
                    if (w > 0) P.push(P.sep_token());
                    if (w >= P.windows) P.drop_oldest_window();
                    P.add_window_goal(gr[w].data());
                    const size_t lo = w + 1 > P.windows ? w + 1 - P.windows : 0;
                    std::fill(slots.begin(), slots.end(), 0.f);
                    for (size_t j = 0; j + lo <= w; ++j) P.normalise_goal(gr[lo + j].data(), slots.data() + j * G);
                    for (size_t i = 0; i < P.n; ++i) {
                        const std::vector<float> l = P.logits();
                        const size_t L = P.length(), V = P.K + 1;
                        condall.assign(L * V, 0.f); uncall.assign(L * V, 0.f);
                        R.forward_all(P.context().data(), L, condall.data(), slots.data(), sty);
                        R.forward_all(P.context().data(), L, uncall.data(), nullptr, sty);
                        for (size_t j = 0; j < V; ++j) {
                            const float c = condall[(L - 1) * V + j], u = uncall[(L - 1) * V + j];
                            mix[j] = scale != 1.f ? u + scale * (c - u) : c;
                            w3 = std::max(w3, (double)std::fabs(l[j] - mix[j])); m3 = std::max(m3, (double)std::fabs(mix[j]));
                        }
                        P.push(P.sample_code(s3, sp3)); ++st3;
                    }
                }
                const double r3 = w3 / (m3 + 1e-12);
                std::printf("goal%s, cfg %.1f, sliding context (%zu steps): max abs diff %.3e (relative %.3e)  %s\n", sty >= 0 ? "+style" : "", scale, st3, w3, r3, r3 < 1e-4 ? "OK" : "FAIL");
                ok3 = ok3 && r3 < 1e-4;
            }
            P.set_cfg_scale(1.f);
        }
        return (rel < 1e-4 && rel2 < 1e-4 && ok3) ? 0 : 1;
    }

    if (cmd == "bench") {
        std::mt19937_64 rng(1);
        const size_t nw = P.windows + 2;                       // exercises the sliding-context rebuild too
        std::vector<int32_t> codes(nw * P.n);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) P.generate(action, nw, sp, rng, codes.data());
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        std::printf("%zu windows x %zu codes: %.2f ms  (%.3f ms / code, %.2f ms / window; avg of %d, threads=%d)\n", nw, P.n, ms, ms / (double)(nw * P.n), ms / (double)nw, iters, std::max(1, threads));
        return 0;
    }
    throw std::runtime_error("unknown command '" + cmd + "'");
} catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
