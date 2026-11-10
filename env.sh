# source me before running SURGE:  source env.sh
# Sets SURGE_BIN and, under WSL, applies the GPU-driver library fix.
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SURGE_BIN=${SURGE_BIN:-$_here/spparks/build-cuda-${PREC:-mixed}/spparks}
export OMP_NUM_THREADS=1

# WSL: the GPU driver lives in /usr/lib/wsl. If Linux-driver copies of libcuda and
# libnvidia-ptxjitcompiler sit in /usr/lib/x86_64-linux-gnu (Ubuntu's nvidia-cuda-toolkit
# package pulls them in through libnvidia-compute-*), CUDA finds that ptxjitcompiler and
# segfaults at startup. Load the WSL driver's own libraries first.
if [ -d /usr/lib/wsl/lib ]; then
  _drv=$(dirname "$(ls /usr/lib/wsl/drivers/*/libnvidia-ptxjitcompiler.so.1 2>/dev/null | head -1)")
  [ "$_drv" = "." ] && _drv=""
  export LD_LIBRARY_PATH=${_drv:+$_drv:}/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  unset _drv
fi
[ -x "$SURGE_BIN" ] || echo "env.sh: warning: $SURGE_BIN not found; run build.sh first" >&2
unset _here
