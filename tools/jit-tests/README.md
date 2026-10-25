# JIT tests

Offline checks for the dynamic recompiler code. They need Python with `keystone-engine`, `capstone` and `unicorn`
(`pip install keystone-engine capstone unicorn`) and a C++20 compiler; nothing here needs an ARM64 machine, since the
generated code is executed in the Unicorn CPU emulator.

- `check_encodings.py` verifies every instruction `ymir::jit::arm64::Assembler` can emit against Keystone, and runs the
  constant-loading sequences in Unicorn:

  ```sh
  g++ -std=c++20 -I libs/ymir-core/include tools/jit-tests/dump_encodings.cpp -o /tmp/dump_encodings
  python3 tools/jit-tests/check_encodings.py /tmp/dump_encodings
  ```

- `sh2_difftest.py` runs random SH-2 programs on two cores, one using the interpreter and one using the recompiler, and
  compares the CPU state, cycle counts and RAM after every slice of cycles. The recompiler's AArch64 blocks run in
  Unicorn; the calls they make back into the core run natively. It also has a few hand-written cases (self-modifying
  code, delay slots, loops). `TRACE=1` prints every generated block and every call into the core.

  The core must be built with position independent code first:

  ```sh
  cmake -S . -B /tmp/bcorepic -G Ninja -DYmir_LIBRARY_ONLY=ON -DYmir_ENABLE_TESTS=OFF -DYmir_ENABLE_SANDBOX=OFF \
        -DYmir_ENABLE_YMDASM=OFF -DCMAKE_BUILD_TYPE=Release -DYmir_ENABLE_IPO=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  ninja -C /tmp/bcorepic ymir-core
  tools/jit-tests/build_difftest.sh /tmp/bcorepic /tmp/libsh2diff.so
  python3 tools/jit-tests/sh2_difftest.py /tmp/libsh2diff.so 1000 1   # iterations, first seed
  ```
