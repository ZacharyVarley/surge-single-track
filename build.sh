#!/usr/bin/env bash
# Fetch SURGE (SPPARKS, branch SURGE) at a pinned commit and build its CUDA executable.
#
# usage:  conda activate surge && bash build.sh
#
# Settings, all optional (environment variables):
#   SURGE_COMMIT  commit to build (default: the one this demo was tested with)
#   SURGE_DIR     where the source checkout goes (default: ./spparks)
#   CUDA_HOME     CUDA toolkit root (default: the conda env's, else /usr/local/cuda, else the
#                 nvcc on PATH)
#   KOKKOS_ARCH   Kokkos GPU token, e.g. TURING75, AMPERE86, ADA89, HOPPER90, BLACKWELL120
#                 (default: read from nvidia-smi)
#   PREC          mixed (default, about 2x faster on consumer GPUs) or double
#   JOBS          parallel compile jobs (default: all cores)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SURGE_REPO=${SURGE_REPO:-https://github.com/trodger-snl/spparks.git}
SURGE_COMMIT=${SURGE_COMMIT:-96fb7587d304eb02185480d2143a913d5198feae}
SURGE_DIR=${SURGE_DIR:-$HERE/spparks}
PREC=${PREC:-mixed}
JOBS=${JOBS:-$(nproc)}
BUILD=$SURGE_DIR/build-cuda-$PREC

die() { echo "build.sh: $*" >&2; exit 1; }

# --- toolchain -------------------------------------------------------------
[ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/g++" ] && [ -x "$CONDA_PREFIX/bin/mpicxx" ] ||
  die "activate the conda env first (conda env create -f environment.yml; conda activate surge)"
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then CUDA_HOME=$CONDA_PREFIX
  elif [ -x /usr/local/cuda/bin/nvcc ]; then CUDA_HOME=/usr/local/cuda
  elif command -v nvcc >/dev/null; then CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  else die "no CUDA toolkit found; install one or set CUDA_HOME"; fi
fi
export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH
# conda's nvcc activation adds -ccbin through these; nvcc_wrapper passes its own -ccbin
unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
cuda_ver=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
gcc_major=$("$CONDA_PREFIX/bin/g++" -dumpversion | cut -d. -f1)
echo "nvcc:  CUDA $cuda_ver ($CUDA_HOME/bin/nvcc)"
echo "g++:   $("$CONDA_PREFIX/bin/g++" --version | head -1)"
case $cuda_ver in   # newest host gcc each nvcc accepts
  12.[0-3]) max_gcc=12 ;; 12.[4-7]) max_gcc=13 ;; 12.*) max_gcc=14 ;; *) max_gcc=15 ;;
esac
[ "$gcc_major" -le "$max_gcc" ] ||
  die "CUDA $cuda_ver needs gcc <= $max_gcc, the env has gcc $gcc_major: conda install gcc=$max_gcc gxx=$max_gcc"

if [ -z "${KOKKOS_ARCH:-}" ]; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ') ||
    die "nvidia-smi failed; set KOKKOS_ARCH by hand"
  case $cc in
    6.0) KOKKOS_ARCH=PASCAL60 ;;  6.1) KOKKOS_ARCH=PASCAL61 ;;
    7.0) KOKKOS_ARCH=VOLTA70 ;;   7.2) KOKKOS_ARCH=VOLTA72 ;;   7.5) KOKKOS_ARCH=TURING75 ;;
    8.0) KOKKOS_ARCH=AMPERE80 ;;  8.6) KOKKOS_ARCH=AMPERE86 ;;  8.7) KOKKOS_ARCH=AMPERE87 ;;
    8.9) KOKKOS_ARCH=ADA89 ;;     9.0) KOKKOS_ARCH=HOPPER90 ;;
    10.0) KOKKOS_ARCH=BLACKWELL100 ;; 10.3) KOKKOS_ARCH=BLACKWELL103 ;;
    12.0) KOKKOS_ARCH=BLACKWELL120 ;; 12.1) KOKKOS_ARCH=BLACKWELL121 ;;
    *) die "unknown compute capability '$cc'; set KOKKOS_ARCH by hand" ;;
  esac
fi
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) -> Kokkos $KOKKOS_ARCH, precision $PREC"

# --- sources ---------------------------------------------------------------
if [ ! -d "$SURGE_DIR/.git" ]; then
  git init -q "$SURGE_DIR"
  git -C "$SURGE_DIR" remote add origin "$SURGE_REPO"
fi
if ! git -C "$SURGE_DIR" cat-file -e "$SURGE_COMMIT^{commit}" 2>/dev/null; then
  echo "fetching SURGE $SURGE_COMMIT from $SURGE_REPO"
  git -C "$SURGE_DIR" fetch -q --depth 1 origin "$SURGE_COMMIT"
fi
git -C "$SURGE_DIR" -c advice.detachedHead=false checkout -q "$SURGE_COMMIT"
git -C "$SURGE_DIR" submodule update -q --init --depth 1 lib/kokkos    # Kokkos, pinned by SURGE

# --- configure and build ---------------------------------------------------
# nvcc_wrapper (from Kokkos) compiles everything; it calls nvcc with the conda g++ as host compiler.
export NVCC_WRAPPER_DEFAULT_COMPILER=$CONDA_PREFIX/bin/g++
export OMPI_CXX=$SURGE_DIR/lib/kokkos/bin/nvcc_wrapper
echo "configuring $BUILD (log: $BUILD.cmake.log)"
cmake -S "$SURGE_DIR/src" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$SURGE_DIR/lib/kokkos/bin/nvcc_wrapper" \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/gcc" \
  -DSPPARKS_PACKAGE_KOKKOS=ON -DSPPARKS_KOKKOS_BACKEND=CUDA -DSPPARKS_KOKKOS_ARCH="$KOKKOS_ARCH" \
  -DKOKKOS_PREC="$PREC" \
  -DSPPARKS_AGGRESSIVE_OPTIMIZATION=OFF -DSPPARKS_NATIVE_OPTIMIZATION=OFF \
  -DSPPARKS_ENABLE_JPEG=OFF -DSPPARKS_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON > "$BUILD.cmake.log" 2>&1 || { tail -30 "$BUILD.cmake.log"; die "configure failed"; }
echo "compiling with $JOBS jobs (log: $BUILD.build.log; a few minutes)"
cmake --build "$BUILD" -j "$JOBS" > "$BUILD.build.log" 2>&1 ||
  { grep -m 20 -B2 -A6 "error" "$BUILD.build.log"; die "build failed"; }
echo "built $BUILD/spparks"
