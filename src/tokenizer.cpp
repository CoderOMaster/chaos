#include "chaos/tokenizer.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>

namespace chaos {

bool WordPieceTokenizer::load(const char* vocab_path) {
  std::ifstream f(vocab_path);
  if (!f) return false;
  vocab_.clear();
  std::string line;
  int32_t id = 0;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF safety
    vocab_.emplace(line, id++);
  }
  auto lookup = [&](const char* t, int32_t& slot) {
    auto it = vocab_.find(t);
    if (it != vocab_.end()) slot = it->second;
  };
  lookup("[CLS]", cls_);
  lookup("[SEP]", sep_);
  lookup("[UNK]", unk_);
  lookup("[PAD]", pad_);
  return !vocab_.empty();
}

// ASCII-oriented basic tokenizer: lowercase, split on whitespace, and peel
// punctuation into standalone tokens. (Full Unicode normalization/CJK handling
// is a known limitation, fine for English edge workloads.)
void WordPieceTokenizer::basic_tokenize(const std::string& text,
                                        std::vector<std::string>& out) const {
  std::string cur;
  auto flush = [&] {
    if (!cur.empty()) { out.push_back(cur); cur.clear(); }
  };
  for (unsigned char c : text) {
    if (std::isspace(c)) {
      flush();
    } else if (std::ispunct(c)) {
      flush();
      out.emplace_back(1, static_cast<char>(c));
    } else {
      cur.push_back(static_cast<char>(std::tolower(c)));
    }
  }
  flush();
}

// Greedy longest-match-first WordPiece. Emits [UNK] for a whole word if any
// piece fails to match, per the reference implementation.
void WordPieceTokenizer::wordpiece(const std::string& word,
                                   std::vector<int32_t>& out) const {
  const size_t n = word.size();
  if (n == 0) return;
  if (n > 100) { out.push_back(unk_); return; }  // reference caps word length

  size_t start = 0;
  std::vector<int32_t> pieces;
  while (start < n) {
    size_t end = n;
    int32_t found = -1;
    while (end > start) {
      std::string sub = word.substr(start, end - start);
      if (start > 0) sub = "##" + sub;
      auto it = vocab_.find(sub);
      if (it != vocab_.end()) { found = it->second; break; }
      --end;
    }
    if (found < 0) { out.push_back(unk_); return; }  // unmatchable -> whole word UNK
    pieces.push_back(found);
    start = end;
  }
  out.insert(out.end(), pieces.begin(), pieces.end());
}

void WordPieceTokenizer::encode(const std::string& text, std::vector<int64_t>& ids,
                                std::vector<int64_t>& mask) const {
  ids.clear();
  mask.clear();
  ids.push_back(cls_);

  std::vector<std::string> basic;
  basic_tokenize(text, basic);

  std::vector<int32_t> wp;
  for (const auto& w : basic) {
    wp.clear();
    wordpiece(w, wp);
    for (int32_t id : wp) {
      if (ids.size() + 1 >= max_len_) break;  // leave room for [SEP]
      ids.push_back(id);
    }
    if (ids.size() + 1 >= max_len_) break;
  }

  ids.push_back(sep_);
  mask.assign(ids.size(), 1);
}

}  // namespace chaos
