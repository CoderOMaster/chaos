cmake_minimum_required(VERSION 3.16)
project(chaos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Edge-oriented optimization: tune for the host by default; override with
# -DCHAOS_ARCH=<arch> when cross-compiling for a specific edge target.
set(CHAOS_ARCH "native" CACHE STRING "Target CPU arch for -mcpu/-march")

# Flags safe to propagate to consumers (optimization, arch, warnings).
add_library(chaos_flags INTERFACE)
target_compile_options(chaos_flags INTERFACE -O3 -fvisibility=hidden -Wall -Wextra)

# -fno-exceptions/-fno-rtti apply to the core library's OWN translation units
# only -- they must NOT propagate to consumers like the pybind11 module, which
# require exceptions and RTTI. (embedder_onnx.cpp / engine.cpp re-enable both
# per-source above via COMPILE_OPTIONS.)
add_library(chaos_core_flags INTERFACE)
target_compile_options(chaos_core_flags INTERFACE -fno-exceptions -fno-rtti)

include(CheckCXXCompilerFlag)
# ARM uses -mcpu, x86 uses -march. Probe both.
check_cxx_compiler_flag("-mcpu=${CHAOS_ARCH}" HAS_MCPU)
check_cxx_compiler_flag("-march=${CHAOS_ARCH}" HAS_MARCH)
if(HAS_MCPU)
  target_compile_options(chaos_flags INTERFACE -mcpu=${CHAOS_ARCH})
elseif(HAS_MARCH)
  target_compile_options(chaos_flags INTERFACE -march=${CHAOS_ARCH})
endif()

find_package(Threads REQUIRED)

# --- core library ---
# Core library: index + distance kernels. No embedder here -- the only embedder
# is the real MiniLM model, which lives in the ONNX Runtime section below.
add_library(chaos
  src/flat_index.cpp
  src/hnsw_index.cpp)
target_include_directories(chaos PUBLIC include)
target_link_libraries(chaos PUBLIC chaos_flags Threads::Threads PRIVATE chaos_core_flags)

# --- core test (no model needed): SIMD math + top-k on numeric fixtures ---
enable_testing()
add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance PRIVATE chaos)
add_test(NAME test_distance COMMAND test_distance)

# --- Python module (pybind11). Implies the real embedder. ---
option(CHAOS_PYTHON "Build the pybind11 Python module" OFF)
if(CHAOS_PYTHON)
  set(CHAOS_ONNX ON CACHE BOOL "" FORCE)  # the Engine needs the real embedder
endif()

# --- real MiniLM path via ONNX Runtime ---
# Enable with: -DCHAOS_ONNX=ON -DORT_ROOT=/path/to/onnxruntime
# All benchmarks and the recall test use the real model, so they live here.
option(CHAOS_ONNX "Build the ONNX Runtime MiniLM embedder, benches, recall test" OFF)
if(NOT DEFINED ORT_ROOT)
  set(ORT_ROOT /opt/homebrew/opt/onnxruntime)  # Homebrew default; override with -DORT_ROOT
endif()
if(CHAOS_ONNX)
  target_sources(chaos PRIVATE src/embedder_onnx.cpp src/tokenizer.cpp src/engine.cpp)
  target_compile_definitions(chaos PUBLIC CHAOS_ONNX=1)
  # The ONNX Runtime C++ API throws, so this TU needs exceptions/RTTI even
  # though the rest of the library builds without them.
  set_source_files_properties(src/embedder_onnx.cpp src/engine.cpp PROPERTIES
    COMPILE_OPTIONS "-fexceptions;-frtti")
  # Homebrew nests the headers under include/onnxruntime; add both.
  target_include_directories(chaos PUBLIC
    ${ORT_ROOT}/include ${ORT_ROOT}/include/onnxruntime)
  # Prefer the unversioned symlink; fall back to the versioned .so shipped in
  # ONNX Runtime release tarballs (Linux). Link exactly one.
  file(GLOB ORT_LIB ${ORT_ROOT}/lib/libonnxruntime.dylib
                    ${ORT_ROOT}/lib/libonnxruntime.so)
  if(NOT ORT_LIB)
    file(GLOB ORT_LIB ${ORT_ROOT}/lib/libonnxruntime.so.*)
    list(SORT ORT_LIB)
    list(GET ORT_LIB 0 ORT_LIB)
  endif()
  target_link_libraries(chaos PUBLIC ${ORT_LIB})

  # Dev-only executables/tests. Skipped for the Python wheel build, which just
  # needs the module.
  if(NOT CHAOS_PYTHON)
    set(MODELS ${CMAKE_SOURCE_DIR}/models)

    add_executable(bench_e2e bench/bench_e2e.cpp)
    target_link_libraries(bench_e2e PRIVATE chaos)

    add_executable(test_tokenizer tests/test_tokenizer.cpp)
    target_link_libraries(test_tokenizer PRIVATE chaos)
    add_test(NAME test_tokenizer COMMAND test_tokenizer ${MODELS}/vocab.txt)

    add_executable(test_hnsw tests/test_hnsw.cpp)
    target_link_libraries(test_hnsw PRIVATE chaos)
    add_test(NAME test_hnsw COMMAND test_hnsw ${MODELS}/model.onnx ${MODELS}/vocab.txt)
  endif()
endif()

# --- pybind11 module: chaos._core ---
if(CHAOS_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core bindings/chaos_py.cpp)
  target_link_libraries(_core PRIVATE chaos)
  # chaos propagates -fno-exceptions/-rtti via its flags; pybind needs both, and
  # a later flag wins.
  target_compile_options(_core PRIVATE -fexceptions -frtti)
  # Install into the package dir for scikit-build-core to pick up.
  install(TARGETS _core LIBRARY DESTINATION chaos)
endif()
