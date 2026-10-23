// cvqvae_cli.cpp — standalone driver for the NEON conditional VQ-VAE (rawllm_cvqvae.hpp).
//
// Build (from the repo root):
//   g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c runtime/motion/cvqvae_cli.cpp -o cvqvae
//
// Usage:
//   cvqvae MODEL.gguf info
//   cvqvae MODEL.gguf encode --in motion.csv [--action NAME|IDX|null] [--normalized] [--latents-out z.csv]   -> prints codes
//   cvqvae MODEL.gguf decode (--codes 1,2,3,... | --codes-file f.txt) [--action NAME|IDX|null] [--out motion.csv] [--normalized]
//   cvqvae MODEL.gguf recon  --in motion.csv [--action ...] [--out recon.csv] [--normalized]   (encode -> quantise -> decode)
//   cvqvae MODEL.gguf bench  [--iters N]
// Options anywhere after MODEL: --threads N   --convert f16|bf16 (lossy, opt-in)
//
// motion.csv: one frame per line, 30 comma/space separated floats, in the CANONICALISED physical units
// train_cvqvae.py trains on (frame 0: pelvis at x=z=0 facing +z; flags 0/1). Lines starting with '#' are skipped.
// Frame count must be a multiple of 4 (model window = 64). With --normalized the CSV is already in
// model space ((x-mean)/std) on input, and `decode`/`recon` write the raw model output (flags = logits).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <iostream>
#include <sstream>

#include "rawllm_cvqvae.hpp"

using namespace cvqvae;

static std::vector<float> read_csv(const std::string& path, size_t cols, size_t& rows) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<float> v; std::string line; rows = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(line); float x; size_t n = 0;
        while (ss >> x) { v.push_back(x); ++n; }
        if (n == 0) continue;
        if (n != cols) throw std::runtime_error(path + ": line " + std::to_string(rows + 1) + " has " + std::to_string(n) + " values, expected " + std::to_string(cols));
        ++rows;
    }
    return v;
}
static void write_csv(std::ostream& o, const float* v, size_t rows, size_t cols) {
    char buf[32];
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) { std::snprintf(buf, sizeof buf, "%.9g", v[r * cols + c]); o << (c ? "," : "") << buf; }
        o << "\n";
    }
}
static std::vector<int32_t> parse_codes(std::string s) {
    for (char& c : s) if (c == ',' || c == ';' || c == '\n' || c == '\t') c = ' ';
    std::istringstream ss(s); std::vector<int32_t> v; long x;
    while (ss >> x) v.push_back((int32_t)x);
    return v;
}

int main(int argc, char** argv) try {
    if (argc < 3) { std::fprintf(stderr, "usage: %s MODEL.gguf (info|encode|decode|recon|bench) [options]   (see header of cvqvae_cli.cpp)\n", argv[0]); return 2; }
    const std::string model_path = argv[1], cmd = argv[2];
    std::string in, out, codes, codes_file, lat_out, action_s = "null", convert;
    bool normalized = false; int threads = 0, iters = 20;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--in") in = val(); else if (a == "--out") out = val(); else if (a == "--codes") codes = val();
        else if (a == "--codes-file") codes_file = val(); else if (a == "--latents-out") lat_out = val();
        else if (a == "--action") action_s = val(); else if (a == "--normalized") normalized = true;
        else if (a == "--threads") threads = std::atoi(val().c_str()); else if (a == "--convert") convert = val();
        else if (a == "--iters") iters = std::atoi(val().c_str());
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
    CVQVAE m; m.load(g, opt);

    int action = -1;
    if (action_s != "null") {
        char* end = nullptr; long x = std::strtol(action_s.c_str(), &end, 10);
        if (end && *end == 0 && !action_s.empty()) action = (int)x;
        else { action = m.action_index(action_s); if (action < 0) throw std::runtime_error("unknown action '" + action_s + "' (run `info` for the list)"); }
        if (action >= (int)m.n_actions) throw std::runtime_error("action index out of range");
    }

    auto sink = [&](const std::function<void(std::ostream&)>& fn) {
        if (out.empty()) fn(std::cout); else { std::ofstream f(out); if (!f) throw std::runtime_error("cannot write " + out); fn(f); }
    };

    if (cmd == "info") {
        std::printf("cVQ-VAE  window=%zu  hidden=%zu  emb=%zu  code_dim=%zu  codes=%zu  downs=%zu  n_res=%zu  channels=%zu  cond_enc=%d  dt=%.6f\n",
                    m.window, m.hidden, m.emb, m.code_dim, m.num_codes, m.downs, m.n_res, m.channels, (int)m.cond_enc, m.dt);
        std::printf("latent steps per window: %zu   actions (%zu + null):", m.window >> m.downs, m.n_actions);
        for (size_t i = 0; i < m.action_names.size(); ++i) std::printf(" %zu=%s", i, m.action_names[i].c_str());
        std::printf("\n"); return 0;
    }

    if (cmd == "encode" || cmd == "recon") {
        if (in.empty()) throw std::runtime_error("--in motion.csv required");
        size_t T = 0; auto x = read_csv(in, m.channels, T);
        if (!T) throw std::runtime_error("no frames in " + in);
        if (!normalized) m.normalize(x.data(), x.data(), T);          // in place is safe: elementwise
        const size_t N = T >> m.downs;
        std::vector<float> z(N * m.code_dim); std::vector<int32_t> idx(N);
        m.encode(x.data(), T, action, z.data());
        m.quantize(z.data(), N, idx.data());
        if (!lat_out.empty()) { std::ofstream f(lat_out); write_csv(f, z.data(), N, m.code_dim); }
        if (cmd == "encode") {
            sink([&](std::ostream& o) { for (size_t n = 0; n < N; ++n) o << (n ? " " : "") << idx[n]; o << "\n"; });
            return 0;
        }
        std::vector<float> y(T * m.channels), phys(T * m.channels);
        m.decode_codes(idx.data(), N, action, y.data());
        if (!normalized) m.to_motion(y.data(), phys.data(), T);
        sink([&](std::ostream& o) { write_csv(o, normalized ? y.data() : phys.data(), T, m.channels); });
        return 0;
    }

    if (cmd == "decode") {
        std::vector<int32_t> idx;
        if (!codes.empty()) idx = parse_codes(codes);
        else if (!codes_file.empty()) { std::ifstream f(codes_file); std::stringstream b; b << f.rdbuf(); idx = parse_codes(b.str()); }
        else throw std::runtime_error("--codes or --codes-file required");
        if (idx.empty()) throw std::runtime_error("no codes given");
        const size_t N = idx.size(), T = N << m.downs;
        std::vector<float> y(T * m.channels), phys(T * m.channels);
        m.decode_codes(idx.data(), N, action, y.data());
        if (!normalized) m.to_motion(y.data(), phys.data(), T);
        sink([&](std::ostream& o) { write_csv(o, normalized ? y.data() : phys.data(), T, m.channels); });
        return 0;
    }

    if (cmd == "bench") {
        const size_t T = m.window, N = T >> m.downs;
        std::vector<float> x(T * m.channels), z(N * m.code_dim), y(T * m.channels); std::vector<int32_t> idx(N);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * std::sin(0.37f * (float)i);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { m.encode(x.data(), T, action, z.data()); m.quantize(z.data(), N, idx.data()); }
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) m.decode_codes(idx.data(), N, action, y.data());
        auto t2 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::printf("window=%zu frames: encode+quantise %.2f ms   decode %.2f ms   (avg of %d, threads=%d)\n", T, ms(t0, t1) / iters, ms(t1, t2) / iters, iters, std::max(1, threads));
        return 0;
    }
    throw std::runtime_error("unknown command '" + cmd + "'");
} catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
