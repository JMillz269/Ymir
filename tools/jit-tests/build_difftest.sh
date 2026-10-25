#!/bin/sh
# Builds the shared library used by sh2_difftest.py. Usage: build_difftest.sh <PIC core build dir> <output .so>
B=${1:-/tmp/bcorepic}; OUT=${2:-/tmp/libsh2diff.so}; R=$(cd "$(dirname "$0")/../.." && pwd)
exec g++ -std=gnu++20 -O1 -fPIC -shared -DYmir_ENABLE_DEVLOG=1 -DYmir_EXTRA_INLINING=1 -DYmir_DEV_ASSERTIONS=0 -DYmir_LOCAL_BUILD=1 \
  -I$R/libs/ymir-core/include -I$R/libs/ymir-core/src -I$B/libs/ymir-core/include -I$B/_cmrc/include -I$R/vendor/fmt/include \
  -I$R/vendor/mio/include -I$R/vendor/concurrentqueue/concurrentqueue -I$R/vendor/xxHash/xxHash -I$R/vendor/libchdr/libchdr/include \
  -I$R/vendor/stb/stb -I$R/vendor/stb/ymir -I$R/vendor/dr_libs/include \
  $R/tools/jit-tests/sh2_difftest_lib.cpp -o "$OUT" -Wl,--whole-archive $B/libs/ymir-core/libymir-core.a -Wl,--no-whole-archive \
  $B/vendor/fmt/libfmt.a $B/vendor/xxHash/libxxHash.a $B/vendor/libchdr/libchdr-static.a \
  $B/vendor/libchdr/libchdr/deps/miniz-3.1.2/libminiz.a $B/vendor/libchdr/libchdr/deps/zstd-1.5.7/libzstd.a \
  $B/vendor/libchdr/libchdr/deps/lzma-26.02/libchdr-lzma.a -lpthread
