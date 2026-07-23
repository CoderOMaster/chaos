// Verifies the WordPiece tokenizer against known bert-base-uncased ids.
//   ./test_tokenizer <vocab.txt>
#include <cstdio>
#include <vector>

#include "chaos/tokenizer.hpp"

using namespace chaos;

static int failures = 0;
static void check(bool ok, const char* msg) {
  if (!ok) { std::printf("FAIL: %s\n", msg); ++failures; }
}

static bool eq(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <vocab.txt>\n", argv[0]); return 2; }
  WordPieceTokenizer tok;
  check(tok.load(argv[1]), "load vocab");
  check(tok.vocab_size() == 30522, "vocab size 30522");

  std::vector<int64_t> ids, mask;

  // "hello world" -> [CLS] hello world [SEP] = 101 7592 2088 102
  tok.encode("hello world", ids, mask);
  check(eq(ids, {101, 7592, 2088, 102}), "hello world ids");
  check(mask.size() == ids.size(), "mask length matches ids");

  // Lowercasing + punctuation split: "Hello, World!"
  tok.encode("Hello, World!", ids, mask);
  check(eq(ids, {101, 7592, 1010, 2088, 999, 102}), "punctuation + lowercase");

  // WordPiece continuation: "embeddings" -> em ##bed ##ding ##s = 7861 8270 4667 2015
  tok.encode("embeddings", ids, mask);
  check(eq(ids, {101, 7861, 8270, 4667, 2015, 102}), "wordpiece embeddings");

  // Truncation respects max_len (incl. [CLS]/[SEP]).
  tok.set_max_len(5);
  tok.encode("one two three four five six seven", ids, mask);
  check(ids.size() == 5, "truncated to max_len");
  check(ids.front() == 101 && ids.back() == 102, "framing preserved after truncation");

  if (failures == 0) std::printf("all tokenizer tests passed\n");
  return failures ? 1 : 0;
}
