#pragma once
#include <cstddef>
#include <memory>

namespace chaos {

class Embedder {
public:
  virtual ~Embedder() = default;
  virtual size_t dim() const = 0;
  // Writes dim() floats into `out`. Not required to be normalized; the index
  // normalizes on insert and the caller should normalize queries.
  virtual void embed(const char *text, float *out) const = 0;
};

#if defined(CHAOS_ONNX)
// Real MiniLM path. model_path = *.onnx, vocab_path = WordPiece vocab.txt.
std::unique_ptr<Embedder> make_onnx_embedder(const char *model_path,
                                             const char *vocab_path,
                                             int intra_op_threads = 4);
#endif

} // namespace chaos
