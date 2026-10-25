#!/usr/bin/env python3
"""Rough cost model of the recompiler: runs random programs on the recompiler only and counts the AArch64 instructions
the generated code executes (in Unicorn), the calls back into the interpreter and the entries from the interpreter's loop.

Usage: sh2_bench.py <libsh2diff.so> [cases] [seed]   (YMIR_JIT_* environment variables change the recompiler's tuning)
"""
import os, sys
lib_path = sys.argv[1]
cases = int(sys.argv[2]) if len(sys.argv) > 2 else 300
seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
sys.argv = [sys.argv[0], lib_path, "0", "1"]
os.environ["SKIP_DIRECTED"] = "1"
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sh2_difftest.py")).read().replace("\nmain()\n", "\n")
exec(compile(src, "sh2_difftest", "exec"))
from unicorn import UC_HOOK_BLOCK
ILLEGAL = lib.t_opcode_type(0x0000, 0)

arm_instrs = 0
thunks = 0
entries = 0
cycles_total = 0
orig_on_thunk = None

for i in range(cases):
    rng = random.Random(seed0 + i)
    code = gen_program(rng, rng.choice([16, 40, 90]))
    blob = b"".join(w.to_bytes(2, "big") for w in code)
    data = bytes(rng.getrandbits(8) for _ in range(0x8000))
    st = make_state(rng)
    rec = Recomp()
    counts = [0, 0]

    def on_block(uc, address, size, user):
        counts[0] += size // 4

    def on_thunk_count(uc, address, size, user):
        counts[1] += 1

    rec.uc.hook_add(UC_HOOK_BLOCK, on_block)
    rec.uc.hook_add(UC_HOOK_CODE, on_thunk_count, begin=rec.thunk, end=rec.thunk)
    lib.t_set_mem(rec.h, RAM_BASE_H + 0x1000, blob, len(blob))
    lib.t_set_mem(rec.h, RAM_BASE_H + 0x8000, data, len(data))
    lib.t_set_mem(rec.h, RAM_BASE_H + 0x400, bytes(0x400), 0x400)
    write_state(rec.h, st)
    for step in range(8):
        cycles_total += lib.t_advance(rec.h, 200, 0)
    s = (u64 * 8)()
    lib.t_stats(rec.h, s)
    entries += s[1]
    arm_instrs += counts[0]
    thunks += counts[1]
    lib.t_destroy(rec.h)

cost = arm_instrs + 80 * thunks + 120 * entries
g = (u64 * 6)()
lib.t_global_stats(g)
print(f"{cycles_total} SH-2 cycles: {arm_instrs / cycles_total:.1f} ARM instrs/cycle, {thunks / cycles_total:.3f} interpreter calls/cycle, "
      f"{entries / cycles_total:.4f} entries/cycle => estimated cost {cost / cycles_total:.1f}/cycle; "
      f"code {g[2] / max(g[3], 1):.0f} B/insn, {g[0]} blocks compiled, {g[4] / max(g[0], 1) / 1000:.0f} us per block (slowest {g[5] / 1000:.0f} us)")
