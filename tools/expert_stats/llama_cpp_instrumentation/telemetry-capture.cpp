// telemetry-capture.cpp — offline-oracle instrumentation for DS5 WP-1 (M1 hour-zero
// task 3: routing-frequency telemetry over the Qwen3-235B-A22B router-calibration
// corpus). Drop into a llama.cpp checkout's examples/telemetry-capture/ and build
// as an ordinary example (see README.md in this directory for the one-line
// examples/CMakeLists.txt registration and build/run commands).
//
// Per ADR-002 this is an offline oracle only: it links against llama.cpp to
// produce a data file and is never invoked by, or linked into, the DS5 runtime.
//
// Uses llama.cpp's existing ggml_backend_sched_eval_callback hook (the same
// mechanism as examples/eval-callback and tools/imatrix) — no llama.cpp core
// patch is required. The callback watches for the "ffn_moe_topk-<il>" and
// "ffn_moe_weights_norm-<il>" tensors that llm_graph_context::build_moe_ffn
// names on every MoE forward pass (src/llama-graph.cpp) and accumulates
// per-(layer, expert) activation counts and gate-weight sums across the
// corpus. Qwen3-MoE always runs with norm_w=true (src/models/qwen3moe.cpp),
// so "ffn_moe_weights_norm" (the post-normalization gate weight actually used
// to combine expert outputs) is always emitted for this architecture.
//
// Output is the routing-telemetry JSON contract documented in
// docs/runbooks/expert-stats-capture.md and consumed by tools/expert_stats/merge_stats.py
// unmodified.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    std::string corpus_path;
    std::string output_path;
    std::string source_id;
    std::string model_id;
    std::string model_hash;
    std::string corpus_id;
    std::string corpus_hash;
    int32_t n_layer  = 94;
    int32_t n_expert = 128;
    int32_t top_k    = 8;
    int32_t n_ctx    = 4096;
    int32_t n_batch  = 4096;
    int32_t n_gpu_layers = 99;
    int32_t chunks   = -1; // -1 = whole corpus
};

// Accumulator, keyed by [layer][expert]. Also holds the scratch needed to pair
// up the "ffn_moe_topk-<il>" tensor (selected expert ids) with the
// "ffn_moe_weights_norm-<il>" tensor (gate weight per selected expert) that the
// graph builder emits immediately afterward for the same layer and ubatch.
struct TelemetryState {
    int32_t n_layer  = 0;
    int32_t n_expert = 0;

    std::vector<std::vector<int64_t>> activation_counts; // [layer][expert]
    std::vector<std::vector<double>>  gate_weight_sums;  // [layer][expert]

    std::map<int, std::vector<int32_t>> pending_topk_ids;   // layer -> [n_tokens * n_used]
    std::map<int, int64_t>              pending_topk_ntok;  // layer -> n_tokens
    std::map<int, int64_t>              pending_topk_nused;  // layer -> n_used
};

// llm_graph_context::cb() formats tensor names as "%s-%d" (src/llama-context.cpp,
// ggml_format_name(cur, "%s-%d", name, il)). Match "<prefix>-<digits>" exactly so
// e.g. "ffn_moe_weights-3" (a different, shorter base name) is never mistaken
// for a match against prefix "ffn_moe_weights_norm".
bool parse_layer_suffix(const char * name, const char * prefix, int * il) {
    const size_t plen = strlen(prefix);
    if (strncmp(name, prefix, plen) != 0 || name[plen] != '-') {
        return false;
    }
    char * end = nullptr;
    const long v = strtol(name + plen + 1, &end, 10);
    if (end == name + plen + 1 || *end != '\0') {
        return false;
    }
    *il = (int) v;
    return true;
}

bool telemetry_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (TelemetryState *) user_data;

    int il = -1;
    const bool is_topk    = parse_layer_suffix(t->name, "ffn_moe_topk", &il);
    // Qwen3-MoE always sets norm_w=true (src/models/qwen3moe.cpp), so
    // build_moe_ffn always emits the post-normalization "ffn_moe_weights_norm"
    // tensor — the gate weight actually used to combine expert outputs — right
    // after "ffn_moe_topk" for the same layer. Track that one, not the
    // pre-normalization "ffn_moe_weights".
    const bool is_weights = !is_topk && parse_layer_suffix(t->name, "ffn_moe_weights_norm", &il);

    if (!is_topk && !is_weights) {
        return false; // not a tensor we care about; skip materializing its data
    }
    if (ask) {
        return true; // yes, follow up with the actual data
    }
    if (il < 0 || il >= st->n_layer) {
        return true; // out-of-range layer index; ignore rather than corrupt state
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    std::vector<uint8_t> tmp;
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
        data = tmp.data();
    }

    if (is_topk) {
        // "ffn_moe_topk": I32, shape [n_expert_used, n_tokens]
        const int64_t n_used   = t->ne[0];
        const int64_t n_tokens = t->ne[1];
        std::vector<int32_t> ids((size_t) (n_used * n_tokens));
        for (int64_t j = 0; j < n_tokens; ++j) {
            for (int64_t k = 0; k < n_used; ++k) {
                const int32_t e = *(const int32_t *) (data + j * t->nb[1] + k * t->nb[0]);
                ids[(size_t) (j * n_used + k)] = e;
                if (e >= 0 && e < st->n_expert) {
                    st->activation_counts[il][e] += 1;
                }
            }
        }
        st->pending_topk_ids[il]  = std::move(ids);
        st->pending_topk_ntok[il] = n_tokens;
        st->pending_topk_nused[il] = n_used;
        return true;
    }

    // "ffn_moe_weights_norm": F32, shape [n_expert_used, n_tokens] (2D — this
    // tensor is reshaped to 2D right before the cb() call in build_moe_ffn,
    // unlike the 3D [1, n_expert_used, n_tokens] "ffn_moe_weights").
    auto it = st->pending_topk_ids.find(il);
    if (it == st->pending_topk_ids.end()) {
        return true; // no matching topk buffered for this layer; skip this pass
    }
    const int64_t n_used   = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (st->pending_topk_ntok[il] != n_tokens || st->pending_topk_nused[il] != n_used) {
        st->pending_topk_ids.erase(it); // shape mismatch guard; drop rather than misattribute
        return true;
    }
    const auto & ids = it->second;
    for (int64_t j = 0; j < n_tokens; ++j) {
        for (int64_t k = 0; k < n_used; ++k) {
            const float w = *(const float *) (data + j * t->nb[1] + k * t->nb[0]);
            const int32_t e = ids[(size_t) (j * n_used + k)];
            if (e >= 0 && e < st->n_expert) {
                st->gate_weight_sums[il][e] += (double) w;
            }
        }
    }
    st->pending_topk_ids.erase(it);
    return true;
}

void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s -m MODEL.gguf -f corpus.txt -o telemetry.json \\\n"
        "          --source-id ID --model-id ID --model-hash sha256:HEX \\\n"
        "          --corpus-id ID --corpus-hash sha256:HEX \\\n"
        "          [--n-layer 94] [--n-expert 128] [--top-k 8] \\\n"
        "          [--ctx 4096] [--batch 4096] [--ngl 99] [--chunks N]\n"
        "\n"
        "Runs the router-calibration corpus through the model as a sequence of\n"
        "independent n_ctx-token windows (same chunking model as tools/imatrix),\n"
        "recording per-(layer, expert) activation counts and gate-weight sums via\n"
        "llama.cpp's eval callback. Writes the telemetry JSON contract consumed by\n"
        "tools/expert_stats/merge_stats.py.\n",
        prog);
}

bool parse_args(int argc, char ** argv, Args * args) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")        args->model_path  = next();
        else if (a == "-f" || a == "--corpus")       args->corpus_path = next();
        else if (a == "-o" || a == "--output")       args->output_path = next();
        else if (a == "--source-id")                 args->source_id   = next();
        else if (a == "--model-id")                  args->model_id    = next();
        else if (a == "--model-hash")                args->model_hash  = next();
        else if (a == "--corpus-id")                  args->corpus_id   = next();
        else if (a == "--corpus-hash")                args->corpus_hash = next();
        else if (a == "--n-layer")                    args->n_layer     = std::stoi(next());
        else if (a == "--n-expert")                   args->n_expert    = std::stoi(next());
        else if (a == "--top-k")                      args->top_k       = std::stoi(next());
        else if (a == "--ctx")                        args->n_ctx       = std::stoi(next());
        else if (a == "--batch")                      args->n_batch     = std::stoi(next());
        else if (a == "--ngl")                        args->n_gpu_layers = std::stoi(next());
        else if (a == "--chunks")                     args->chunks      = std::stoi(next());
        else if (a == "-h" || a == "--help")         { print_usage(argv[0]); exit(0); }
        else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (args->model_path.empty() || args->corpus_path.empty() || args->output_path.empty() ||
        args->source_id.empty() || args->model_id.empty() || args->model_hash.empty() ||
        args->corpus_id.empty() || args->corpus_hash.empty()) {
        fprintf(stderr, "missing required argument\n");
        print_usage(argv[0]);
        return false;
    }
    return true;
}

std::string read_file(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "failed to open file: %s\n", path.c_str());
        exit(1);
    }
    std::ostringstream ss;
    ss << fin.rdbuf();
    return ss.str();
}

void write_telemetry_json(const Args & args, const TelemetryState & st, int64_t total_tokens) {
    std::ofstream fout(args.output_path);
    if (!fout) {
        fprintf(stderr, "failed to open output file: %s\n", args.output_path.c_str());
        exit(1);
    }
    fout << "{\n";
    fout << "  \"source_id\": \"" << args.source_id << "\",\n";
    fout << "  \"model_id\": \"" << args.model_id << "\",\n";
    fout << "  \"model_hash\": \"" << args.model_hash << "\",\n";
    fout << "  \"corpus_id\": \"" << args.corpus_id << "\",\n";
    fout << "  \"corpus_hash\": \"" << args.corpus_hash << "\",\n";
    fout << "  \"num_layers\": " << args.n_layer << ",\n";
    fout << "  \"num_experts_per_layer\": " << args.n_expert << ",\n";
    fout << "  \"top_k\": " << args.top_k << ",\n";
    fout << "  \"total_tokens\": " << total_tokens << ",\n";
    fout << "  \"layers\": [\n";
    for (int32_t l = 0; l < args.n_layer; ++l) {
        fout << "    {\n";
        fout << "      \"layer\": " << l << ",\n";
        fout << "      \"activation_counts\": [";
        for (int32_t e = 0; e < args.n_expert; ++e) {
            fout << st.activation_counts[l][e];
            if (e + 1 < args.n_expert) fout << ", ";
        }
        fout << "],\n";
        fout << "      \"gate_weight_sums\": [";
        for (int32_t e = 0; e < args.n_expert; ++e) {
            fout << st.gate_weight_sums[l][e];
            if (e + 1 < args.n_expert) fout << ", ";
        }
        fout << "]\n";
        fout << "    }" << (l + 1 < args.n_layer ? "," : "") << "\n";
    }
    fout << "  ]\n";
    fout << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return 1;
    }
    if (args.n_batch < args.n_ctx) {
        args.n_batch = args.n_ctx; // decode is called with up to n_ctx tokens at once
    }

    const std::string corpus_text = read_file(args.corpus_path);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", args.model_path.c_str());
        return 1;
    }

    TelemetryState state;
    state.n_layer  = args.n_layer;
    state.n_expert = args.n_expert;
    state.activation_counts.assign(args.n_layer, std::vector<int64_t>(args.n_expert, 0));
    state.gate_weight_sums.assign(args.n_layer, std::vector<double>(args.n_expert, 0.0));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = args.n_ctx;
    cparams.n_batch           = args.n_batch;
    cparams.n_ubatch          = args.n_batch;
    cparams.cb_eval           = telemetry_cb_eval;
    cparams.cb_eval_user_data = &state;
    cparams.no_perf           = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    int32_t n_tokens_max = (int32_t) corpus_text.size() + 16;
    std::vector<llama_token> tokens(n_tokens_max);
    int32_t n_tokens = llama_tokenize(vocab, corpus_text.c_str(), (int32_t) corpus_text.size(),
                                       tokens.data(), n_tokens_max, add_bos, false);
    if (n_tokens < 0) {
        n_tokens_max = -n_tokens + 16;
        tokens.resize(n_tokens_max);
        n_tokens = llama_tokenize(vocab, corpus_text.c_str(), (int32_t) corpus_text.size(),
                                   tokens.data(), n_tokens_max, add_bos, false);
    }
    if (n_tokens < 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    tokens.resize(n_tokens);

    const int32_t n_ctx_run = (int32_t) llama_n_ctx(ctx);
    const int64_t budget = args.chunks > 0 ? (int64_t) args.chunks * n_ctx_run : (int64_t) n_tokens;
    const int64_t limit  = std::min<int64_t>(n_tokens, budget);

    fprintf(stderr,
        "telemetry-capture: %lld of %d corpus tokens, %d-token windows, "
        "%d layers x %d experts (top-%d)\n",
        (long long) limit, n_tokens, n_ctx_run, args.n_layer, args.n_expert, args.top_k);

    llama_batch batch = llama_batch_init(n_ctx_run, 0, 1);
    int64_t total_tokens = 0;
    int64_t pos = 0;
    while (pos < limit) {
        const int32_t n_this = (int32_t) std::min<int64_t>(n_ctx_run, limit - pos);
        batch.n_tokens = n_this;
        for (int32_t i = 0; i < n_this; ++i) {
            batch.token[i]     = tokens[pos + i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = false; // prefill-only: we read MoE routing tensors
                                        // via cb_eval, not final logits
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed at corpus offset %lld\n", (long long) pos);
            return 1;
        }
        total_tokens += n_this;
        // Each window is an independent context, matching tools/perplexity's/
        // tools/imatrix's fixed-size chunking (documented compromise: routing
        // telemetry does not carry cross-window attention context).
        llama_memory_clear(llama_get_memory(ctx), true);
        pos += n_this;
    }
    llama_batch_free(batch);

    write_telemetry_json(args, state, total_tokens);
    fprintf(stderr, "wrote %s (%lld tokens processed)\n",
            args.output_path.c_str(), (long long) total_tokens);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
