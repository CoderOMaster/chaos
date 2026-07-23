#!/usr/bin/env bash
# Download a prebuilt ONNX Runtime (C/C++ release, with headers + libs) for the
# current platform into $ORT_ROOT (default /tmp/ort). Run by cibuildwheel's
# before-all so the extension can link it; the repair step then bundles the
# shared library into the wheel. Override ORT_VERSION / ORT_ROOT via env.
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.20.1}"
ORT_ROOT="${ORT_ROOT:-/tmp/ort}"

sys="$(uname -s)"
mach="$(uname -m)"
case "${sys}-${mach}" in
  Darwin-arm64)   pkg="onnxruntime-osx-arm64-${ORT_VERSION}" ;;
  Darwin-x86_64)  pkg="onnxruntime-osx-x86_64-${ORT_VERSION}" ;;
  Linux-x86_64)   pkg="onnxruntime-linux-x64-${ORT_VERSION}" ;;
  Linux-aarch64)  pkg="onnxruntime-linux-aarch64-${ORT_VERSION}" ;;
  *) echo "ci_fetch_onnxruntime: unsupported platform ${sys}-${mach}" >&2; exit 1 ;;
esac

url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${pkg}.tgz"
echo "Fetching ${url}"
work="$(mktemp -d)"
curl -fL "${url}" | tar xz -C "${work}"

rm -rf "${ORT_ROOT}"
mv "${work}/${pkg}" "${ORT_ROOT}"
echo "ONNX Runtime ${ORT_VERSION} -> ${ORT_ROOT}"
ls "${ORT_ROOT}/lib"
