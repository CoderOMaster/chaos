// BERT WordPiece tokenizer (uncased) for MiniLM. Loads a HuggingFace
// `vocab.txt` (one token per line, line index == id) and encodes text into
// input ids + attention mask with [CLS]/[SEP] framing, matching the
// preprocessing all-MiniLM-L6-v2 was trained with.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace chaos {

class WordPieceTokenizer {
 public:
  bool load(const char* vocab_path);

  // Encodes `text` into `ids` (incl. [CLS] and [SEP]) and a matching all-ones
  // `mask`, truncated to max_len tokens total. token_type_ids are all zero for
  // single-sequence input, so the caller can pass a zero buffer of len ids.
  void encode(const std::string& text, std::vector<int64_t>& ids,
              std::vector<int64_t>& mask) const;

  size_t max_len() const { return max_len_; }
  void set_max_len(size_t m) { max_len_ = m; }
  size_t vocab_size() const { return vocab_.size(); }

 private:
  // Splits into "basic" tokens: lowercased whitespace-delimited words with
  // punctuation broken out as separate tokens (BERT BasicTokenizer, uncased).
  void basic_tokenize(const std::string& text, std::vector<std::string>& out) const;
  // Greedy longest-match WordPiece with "##" continuation.
  void wordpiece(const std::string& word, std::vector<int32_t>& out) const;

  std::unordered_map<std::string, int32_t> vocab_;
  int32_t cls_ = 101, sep_ = 102, unk_ = 100, pad_ = 0;
  size_t max_len_ = 128;
};

}  // namespace chaos
