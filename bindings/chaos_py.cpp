// pybind11 bindings for the native chaos Engine. tokenize -> embed -> search
// all run in C++; the GIL is released around add/search so the native work
// (and any surrounding Python threads) run unblocked.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <string>

#include "chaos/engine.hpp"

namespace py = pybind11;
using namespace chaos;

PYBIND11_MODULE(_core, m) {
  m.doc() = "chaos: CPU-only native semantic search (MiniLM + flat/HNSW)";

  py::class_<Doc>(m, "Hit", "A search result: document id, similarity score, and text.")
      .def_readonly("id", &Doc::id)
      .def_readonly("score", &Doc::score)
      .def_readonly("text", &Doc::text)
      .def("__repr__", [](const Doc& d) {
        char buf[48];
        std::snprintf(buf, sizeof buf, " score=%.3f>", d.score);
        return "<Hit id=" + d.id + std::string(buf) + " " + d.text;
      });

  py::class_<Engine>(m, "Engine",
                     "Low-level engine: embeds text with MiniLM and serves top-k over a\n"
                     "flat (exact) or HNSW (approximate) index, keyed by string ids.")
      .def(py::init<const std::string&, const std::string&, const std::string&,
                    int, size_t, size_t, size_t>(),
           py::arg("model_path"), py::arg("vocab_path"), py::arg("index") = "flat",
           py::arg("threads") = 4, py::arg("hnsw_m") = 16,
           py::arg("ef_construction") = 200, py::arg("ef_search") = 64,
           "Load MiniLM from model_path/vocab_path and create an index.")
      .def("upsert", &Engine::upsert, py::arg("id"), py::arg("text"),
           py::call_guard<py::gil_scoped_release>(),
           "Insert or replace one document by id; returns True if it updated an existing id.")
      .def("upsert_many", &Engine::upsert_many, py::arg("ids"), py::arg("texts"),
           py::call_guard<py::gil_scoped_release>(),
           "Batch upsert; returns (added, updated) counts.")
      .def("search", &Engine::search, py::arg("query"), py::arg("k") = 10,
           py::call_guard<py::gil_scoped_release>(),
           "Return up to k Hits for the query, most similar first.")
      .def("set_ef_search", &Engine::set_ef_search, py::arg("ef"),
           "Tune HNSW recall/latency (no-op for the flat index).")
      .def("save", &Engine::save, py::arg("path"),
           py::call_guard<py::gil_scoped_release>(),
           "Persist the built index (ids, text, vectors, HNSW graph) to a file.")
      .def("load", &Engine::load, py::arg("path"),
           py::call_guard<py::gil_scoped_release>(),
           "Reload a saved index without re-embedding.")
      .def_property_readonly("dim", &Engine::dim)
      .def_property_readonly("is_hnsw", &Engine::is_hnsw)
      .def("__len__", &Engine::size);
}
