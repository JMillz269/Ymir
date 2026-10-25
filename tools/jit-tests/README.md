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
