// Procedurally composed but genuine English sentences, grouped by topic.
//
// This is a *text* source for benchmarks and the recall test -- the embeddings
// are always produced by the real MiniLM model. Because sentences within a
// topic share vocabulary, their real embeddings cluster (as a real corpus
// would), giving well-defined nearest neighbors to measure recall against.
//
// sentence(i) is deterministic in i. Combinations yield ~thousands of distinct
// sentences per topic before repeating; for very large N some repeats occur,
// which is harmless for latency measurement.
#pragma once
#include <cstdio>
#include <string>
#include <vector>

namespace chaos {

class CorpusGen {
 public:
  CorpusGen() {
    topics_ = {
        {"cooking",   {"the chef", "a baker", "the cook", "my grandmother"},
                      {"simmered", "roasted", "seasoned", "kneaded", "braised"},
                      {"the fresh dough", "a rich broth", "the tender meat", "a savory sauce"}},
        {"astronomy", {"the telescope", "an astronomer", "the probe", "the observatory"},
                      {"observed", "tracked", "photographed", "measured", "detected"},
                      {"a distant galaxy", "the gas giant", "a passing comet", "the neutron star"}},
        {"finance",   {"the investor", "a trader", "the analyst", "the fund"},
                      {"hedged", "shorted", "diversified", "liquidated", "leveraged"},
                      {"the volatile portfolio", "a bond position", "the equity stake", "risky derivatives"}},
        {"gardening", {"the gardener", "a botanist", "my neighbor", "the farmer"},
                      {"pruned", "watered", "transplanted", "fertilized", "harvested"},
                      {"the climbing roses", "a vegetable bed", "the fruit trees", "young seedlings"}},
        {"software",  {"the engineer", "a developer", "the team", "the compiler"},
                      {"refactored", "profiled", "deployed", "benchmarked", "debugged"},
                      {"the search index", "a caching layer", "the query planner", "concurrent code"}},
        {"music",     {"the violinist", "a composer", "the quartet", "the conductor"},
                      {"performed", "rehearsed", "arranged", "improvised", "recorded"},
                      {"a haunting melody", "the final movement", "an intricate fugue", "the slow adagio"}},
        {"medicine",  {"the surgeon", "a nurse", "the physician", "the researcher"},
                      {"diagnosed", "treated", "monitored", "prescribed", "examined"},
                      {"the chronic condition", "a rare infection", "the recovering patient", "acute symptoms"}},
        {"sailing",   {"the captain", "a sailor", "the crew", "the navigator"},
                      {"trimmed", "anchored", "charted", "steered", "moored"},
                      {"the billowing sails", "a rocky harbor", "the open ocean", "strong headwinds"}},
        {"geology",   {"the geologist", "a surveyor", "the expedition", "the drill"},
                      {"sampled", "mapped", "excavated", "analyzed", "dated"},
                      {"the sedimentary layers", "a volcanic ridge", "the fault line", "ancient bedrock"}},
        {"painting",  {"the artist", "a painter", "the muralist", "the student"},
                      {"sketched", "shaded", "blended", "outlined", "layered"},
                      {"the vivid landscape", "a bold portrait", "the still life", "abstract shapes"}},
        {"cycling",   {"the cyclist", "a rider", "the racer", "the courier"},
                      {"pedaled", "sprinted", "climbed", "coasted", "drafted"},
                      {"the steep mountain pass", "a winding trail", "the city streets", "the final lap"}},
        {"weather",   {"the meteorologist", "a forecaster", "the station", "the satellite"},
                      {"predicted", "recorded", "tracked", "reported", "measured"},
                      {"an approaching storm", "the cold front", "heavy rainfall", "the heat wave"}},
    };
  }

  size_t num_topics() const { return topics_.size(); }

  std::string sentence(size_t i) const {
    const size_t T = topics_.size();
    const Topic& t = topics_[i % T];
    size_t k = i / T;
    const std::string& subj = t.subjects[k % t.subjects.size()];
    const std::string& verb = t.verbs[(k / t.subjects.size()) % t.verbs.size()];
    const std::string& obj =
        t.objects[(k / (t.subjects.size() * t.verbs.size())) % t.objects.size()];
    static const char* tmpl[] = {
        "During the season %s %s %s.",
        "At dawn %s carefully %s %s.",
        "In the report %s %s %s again.",
        "Yesterday %s %s %s without hesitation.",
        "Near the coast %s slowly %s %s.",
        "Every morning %s %s %s with focus.",
        "Late that night %s finally %s %s.",
        "For the record %s %s %s once more.",
    };
    const size_t nt = sizeof(tmpl) / sizeof(tmpl[0]);
    const char* fmt = tmpl[(k / (t.subjects.size() * t.verbs.size() * t.objects.size())) % nt];
    char buf[256];
    std::snprintf(buf, sizeof buf, fmt, subj.c_str(), verb.c_str(), obj.c_str());
    return std::string(buf);
  }

 private:
  struct Topic {
    const char* name;
    std::vector<std::string> subjects;
    std::vector<std::string> verbs;
    std::vector<std::string> objects;
  };
  std::vector<Topic> topics_;
};

}  // namespace chaos
