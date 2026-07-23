// MiniLM embedder via ONNX Runtime. Tokenizes with WordPiece, runs the
// transformer forward pass, then mean-pools the token embeddings over the
// attention mask and L2-normalizes -- the canonical all-MiniLM-L6-v2 sentence
// embedding. Compiled only when -DCHAOS_ONNX=ON.
#include "chaos/embedder.hpp"

#if defined(CHAOS_ONNX)
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "chaos/distance.hpp"
#include "chaos/tokenizer.hpp"

namespace chaos {
namespace {

class OnnxEmbedder : public Embedder {
 public:
  OnnxEmbedder(const char* model_path, const char* vocab_path, int intra_threads)
      : env_(ORT_LOGGING_LEVEL_WARNING, "chaos") {
    if (!tok_.load(vocab_path)) {
      std::fprintf(stderr, "chaos: failed to load vocab %s\n", vocab_path);
      std::abort();
    }
    Ort::SessionOptions opts;
    // Latency-optimized: parallelize one query's GEMM across a few pinned
    // cores (intra-op), single inter-op thread. Full graph optimization.
    opts.SetIntraOpNumThreads(intra_threads);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, model_path, opts);

    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = session_->GetInputCount();
    for (size_t i = 0; i < n_in; ++i)
      input_names_.push_back(session_->GetInputNameAllocated(i, alloc).get());
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

    // Hidden size from the output shape's last dim (384 for MiniLM-L6).
    auto shape = session_->GetOutputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();
    hidden_ = shape.empty() ? 384 : static_cast<size_t>(shape.back());
  }

  size_t dim() const override { return hidden_; }

  void embed(const char* text, float* out) const override {
    std::vector<int64_t> ids, mask;
    tok_.encode(text, ids, mask);
    const int64_t seq = static_cast<int64_t>(ids.size());
    std::vector<int64_t> type_ids(ids.size(), 0);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t shape[2] = {1, seq};

    // Bind by the model's declared input names so exports with/without
    // token_type_ids both work.
    std::vector<Ort::Value> inputs;
    std::vector<const char*> in_names;
    for (const auto& name : input_names_) {
      int64_t* buf = nullptr;
      if (name == "input_ids") buf = ids.data();
      else if (name == "attention_mask") buf = mask.data();
      else if (name == "token_type_ids") buf = type_ids.data();
      else continue;
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, buf, ids.size(), shape, 2));
      in_names.push_back(name.c_str());
    }

    const char* out_names[1] = {output_name_.c_str()};
    auto res = session_->Run(Ort::RunOptions{nullptr}, in_names.data(),
                             inputs.data(), inputs.size(), out_names, 1);

    // last_hidden_state: [1, seq, hidden]. Mean-pool over valid tokens.
    const float* hs = res[0].GetTensorData<float>();
    for (size_t d = 0; d < hidden_; ++d) out[d] = 0.f;
    int64_t valid = 0;
    for (int64_t t = 0; t < seq; ++t) {
      if (!mask[t]) continue;
      const float* row = hs + t * hidden_;
      for (size_t d = 0; d < hidden_; ++d) out[d] += row[d];
      ++valid;
    }
    if (valid > 0) {
      float inv = 1.f / static_cast<float>(valid);
      for (size_t d = 0; d < hidden_; ++d) out[d] *= inv;
    }
    l2_normalize(out, hidden_);
  }

 private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  WordPieceTokenizer tok_;
  std::vector<std::string> input_names_;
  std::string output_name_;
  size_t hidden_ = 384;
};

}  // namespace

std::unique_ptr<Embedder> make_onnx_embedder(const char* model_path,
                                             const char* vocab_path,
                                             int intra_op_threads) {
  return std::unique_ptr<Embedder>(
      new OnnxEmbedder(model_path, vocab_path, intra_op_threads));
}

}  // namespace chaos
#endif  // CHAOS_ONNX
