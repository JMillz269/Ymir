#!/usr/bin/env python3
"""Differential test of the SH-2 recompiler against the interpreter.

Two SH-2 cores are built from random programs: one runs the interpreter, the other runs the recompiler, whose AArch64
blocks are executed in Unicorn. After every slice of cycles, the complete CPU state, the number of executed cycles and
the contents of RAM must match.

Usage: sh2_difftest.py <libsh2diff.so> [iterations] [seed]
"""
import ctypes as C
import random
import sys

from unicorn import (Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_PROT_ALL, UC_HOOK_CODE, UcError)
from unicorn.arm64_const import UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2, UC_ARM64_REG_X3, UC_ARM64_REG_X30, UC_ARM64_REG_SP, UC_ARM64_REG_PC

RAM_BASE_H = 0x06000000
RAM_BASE_L = 0x00200000
RAM_SIZE = 1 << 20
CODE_ADDR = RAM_BASE_H + 0x1000
RETURN_SENTINEL = 0xDEAD0000
STACK_BASE = 0x7F000000
STACK_SIZE = 0x10000

u32 = C.c_uint32
u64 = C.c_uint64
vp = C.c_void_p

lib = C.CDLL(sys.argv[1])
lib.t_create.restype = vp
lib.t_create.argtypes = [C.c_int]
lib.t_destroy.argtypes = [vp]
lib.t_object_range.argtypes = [vp, C.POINTER(u64), C.POINTER(u64)]
lib.t_set_mem.argtypes = [vp, u32, C.c_char_p, u32]
lib.t_get_ram.argtypes = [vp, C.c_char_p, C.c_char_p]
lib.t_set_state.argtypes = [vp, C.POINTER(u32)]
lib.t_get_state.argtypes = [vp, C.POINTER(u32)]
lib.t_set_intc_level.argtypes = [vp, u32]
lib.t_advance.restype = u64
lib.t_advance.argtypes = [vp, u64, u64]
lib.t_step_thunk_address.restype = u64
lib.t_step_thunk.restype = u64
lib.t_step_thunk.argtypes = [vp, u32]
lib.t_check_thunk_address.restype = u64
lib.t_check_thunk.restype = u64
lib.t_check_thunk.argtypes = [vp, u32, u32, u32]
lib.t_pool_region.argtypes = [C.POINTER(u64), C.POINTER(u64)]
lib.t_stats.argtypes = [vp, C.POINTER(u64)]
lib.t_block_table.argtypes = [vp, C.POINTER(u64), C.POINTER(u64)]
lib.t_opcode_type.restype = u32
lib.t_opcode_type.argtypes = [u32, u32]
lib.t_kind.restype = u32
lib.t_kind.argtypes = [u32]
lib.t_flush.argtypes = [vp]
EXEC_CB = C.CFUNCTYPE(u64, vp, vp, u64)
lib.t_set_executor.argtypes = [vp, EXEC_CB]

lib.t_testing_mode()

ILLEGAL = None
SNAPS = {}
import os
TRACE = bool(os.environ.get('TRACE'))


def read_state(h):
    v = (u32 * 29)()
    lib.t_get_state(h, v)
    return list(v)


def write_state(h, st):
    v = (u32 * 29)(*st)
    lib.t_set_state(h, v)


def get_ram(h):
    hi = C.create_string_buffer(RAM_SIZE)
    lo = C.create_string_buffer(RAM_SIZE)
    lib.t_get_ram(h, hi, lo)
    return hi.raw, lo.raw


NAMES = [f"R{i}" for i in range(16)] + ["PC", "SR", "GBR", "VBR", "PR", "MACH", "MACL", "wbReg", "intrPend",
                                        "intrAllow", "delaySlot", "delayTarget", "sleep"]


class Recomp:
    def __init__(self):
        self.h = lib.t_create(1)
        base, size = u64(), u64()
        lib.t_object_range(self.h, C.byref(base), C.byref(size))
        self.uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        self.uc.mem_map_ptr(base.value, size.value, UC_PROT_ALL, base.value)
        self.uc.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_ALL)
        self.thunk = lib.t_step_thunk_address()
        self.pool_mapped = False
        self.last_compiled = -1
        self.errors = []

        # Each thunk lives on its own page that only holds a RET; a code hook performs the call on the host
        self.check_thunk = lib.t_check_thunk_address()
        for addr in {self.thunk, self.check_thunk}:
            page = addr & ~0xFFF
            try:
                self.uc.mem_map(page, 0x1000, UC_PROT_ALL)
            except UcError:
                pass  # shared page
        for addr in (self.thunk, self.check_thunk):
            self.uc.mem_write(addr, (0xD65F03C0).to_bytes(4, "little"))

        def on_thunk(uc, address, size, user):
            sh2 = uc.reg_read(UC_ARM64_REG_X0)
            instr = uc.reg_read(UC_ARM64_REG_X1) & 0xFFFFFFFF
            r = lib.t_step_thunk(sh2, instr)
            if TRACE:
                print(f"    thunk instr={instr:04x} -> {r} cycles")
            uc.reg_write(UC_ARM64_REG_X0, r)

        def on_check(uc, address, size, user):
            sh2 = uc.reg_read(UC_ARM64_REG_X0)
            pc = uc.reg_read(UC_ARM64_REG_X1) & 0xFFFFFFFF
            off = uc.reg_read(UC_ARM64_REG_X2) & 0xFFFFFFFF
            store_pc = uc.reg_read(UC_ARM64_REG_X3) & 0xFFFFFFFF
            r = lib.t_check_thunk(sh2, pc, off, store_pc)
            if TRACE:
                print(f"    check pc={pc:#x} off={off} -> {r}")
            uc.reg_write(UC_ARM64_REG_X0, r)

        self.uc.hook_add(UC_HOOK_CODE, on_thunk, begin=self.thunk, end=self.thunk)
        self.uc.hook_add(UC_HOOK_CODE, on_check, begin=self.check_thunk, end=self.check_thunk)

        def executor(fn, sh2, budget):
            if not self.pool_mapped:
                pb, ps = u64(), u64()
                lib.t_pool_region(C.byref(pb), C.byref(ps))
                self.uc.mem_map_ptr(pb.value, ps.value, UC_PROT_ALL, pb.value)
                self.pool_mapped = True
            st = (u64 * 8)()
            lib.t_stats(self.h, st)
            if st[0] != self.last_compiled or st[3]:
                self.uc.ctl_flush_tb()
                self.last_compiled = st[0]
            if TRACE:
                pcnow = read_state(self.h)[16]
                print(f"  BLOCK {fn:#x} budget={budget} pc={pcnow:#x}")
                if os.environ.get("TRACE_ASM"):
                    from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
                    md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
                    code = bytes(self.uc.mem_read(fn, 0x1000))
                    for a, _, m, o in md.disasm_lite(code, fn):
                        print(f"    {a - fn:#05x}: {m} {o}")
            self.uc.reg_write(UC_ARM64_REG_X0, sh2)
            self.uc.reg_write(UC_ARM64_REG_X1, budget)
            self.uc.reg_write(UC_ARM64_REG_X30, RETURN_SENTINEL)
            self.uc.reg_write(UC_ARM64_REG_SP, STACK_BASE + STACK_SIZE - 0x100)
            try:
                self.uc.emu_start(fn, RETURN_SENTINEL)
            except UcError as e:
                pc = self.uc.reg_read(UC_ARM64_REG_PC)
                from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
                md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
                code = bytes(self.uc.mem_read(fn, 0x80))
                lines = [f"{a - fn:#x}: {m} {o}" for a, _, m, o in md.disasm_lite(code, fn)]
                regs = " ".join(f"x{i}={self.uc.reg_read(UC_ARM64_REG_X0 + i):#x}" for i in range(0, 22))
                self.errors.append(f"unicorn error {e} at pc={pc:#x} (block {fn:#x})\n" + "\n".join(lines) + "\n" + regs)
                return 0
            return self.uc.reg_read(UC_ARM64_REG_X0)

        self.cb = EXEC_CB(executor)
        lib.t_set_executor(self.h, self.cb)
        tb, ts = u64(), u64()
        lib.t_block_table(self.h, C.byref(tb), C.byref(ts))
        self.uc.mem_map_ptr(tb.value, ts.value, UC_PROT_ALL, tb.value)


def gen_program(rng, n):
    """Random words biased toward instructions the recompiler translates."""
    words = []
    inline_share = rng.choice([0.3, 0.7, 0.95])
    mac_share = 0.3 if rng.random() < 0.2 else 0.0
    while len(words) < n:
        w = rng.getrandbits(16)
        if rng.random() < mac_share:
            w = (rng.choice([0x400F, 0x000F]) | (rng.getrandbits(8) << 4))
        if lib.t_opcode_type(w, 0) == ILLEGAL:
            continue
        if w == 0x001B or (w & 0xFF00) == 0xC300:  # no SLEEP / TRAPA
            continue
        kind = lib.t_kind(w)  # 0 inline, 1 memory, 2 mac, 3 cond branch, 4 helper, 5 delayed branch, 6 end after, 7 stop
        want_inline = rng.random() < inline_share
        if want_inline and kind not in (0, 1, 2):
            continue
        if not want_inline and kind in (3, 5) and rng.random() < 0.5:
            continue
        # Keep branches local so that code loops and blocks are executed repeatedly
        top = w >> 12
        if top in (0xA, 0xB):
            w = (w & 0xF000) | (rng.randrange(-24, 24) & 0xFFF)
        elif top == 0x8 and (w >> 8) & 0xF in (0x9, 0xB, 0xD, 0xF):
            w = (w & 0xFF00) | (rng.randrange(-24, 24) & 0xFF)
        words.append(w)
    return words


def make_state(rng):
    st = [0] * 29
    for i in range(16):
        r = rng.random()
        if r < 0.5:
            st[i] = rng.choice([0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, rng.getrandbits(32), rng.getrandbits(8),
                                rng.getrandbits(16)])
        elif r < 0.62:
            st[i] = CODE_ADDR + rng.randrange(0, 160)  # stores may modify the code
        else:
            base = rng.choice([RAM_BASE_H, RAM_BASE_H, 0x26000000, 0xA6000000, RAM_BASE_L, 0x20200000, 0x06100000])
            st[i] = base + 0x8000 + rng.randrange(0, 0x4000) * (1 if rng.random() < 0.5 else 2)
    st[16] = CODE_ADDR
    st[17] = rng.getrandbits(32) & 0x000003F3  # M, Q, I3-0, S, T
    st[15] = RAM_BASE_H + 0x20000
    st[18] = RAM_BASE_H + 0x9000
    st[19] = RAM_BASE_H + 0x400
    st[20] = CODE_ADDR + 2 * rng.randrange(0, 64)
    st[21] = rng.choice([0, 1, 0xFFFFFFFF, 0x7FFF, 0xFFFF8000, rng.getrandbits(32)])
    st[22] = rng.choice([0, 0x7FFFFFFF, 0x80000000, 0x7FFFFF00, 0xFFFFFFFF, rng.getrandbits(32)])
    st[23] = 0
    return st


def bisect(seed, maxb=260):
    """Finds the smallest cycle budget (from the initial state) at which the two cores diverge."""
    for budget in ([int(__import__("os").environ["B"])] if "B" in __import__("os").environ else range(1, maxb)):
        rng = random.Random(seed)
        code = gen_program(rng, rng.choice([16, 40, 90]))
        blob = b"".join(w.to_bytes(2, "big") for w in code)
        data = bytes(rng.getrandbits(8) for _ in range(0x8000))
        st = make_state(rng)
        rng.choice([6, 12])
        rng.choice([1, 2, 3, 7, 20, 64, 200])
        interp = lib.t_create(0)
        rec = Recomp()
        for h in (interp, rec.h):
            lib.t_set_mem(h, RAM_BASE_H + 0x1000, blob, len(blob))
            lib.t_set_mem(h, RAM_BASE_H + 0x8000, data, len(data))
            lib.t_set_mem(h, RAM_BASE_H + 0x400, bytes(0x400), 0x400)
            write_state(h, st)
        if rng.random() < 0.15:
            ia = read_state(interp)
            ia[24] = 1
            ia[25] = rng.getrandbits(1)
            write_state(interp, ia)
            write_state(rec.h, ia)
        ci = lib.t_advance(interp, budget, 0)
        cr = lib.t_advance(rec.h, budget, 0)
        si, sr = read_state(interp), read_state(rec.h)
        lib.t_destroy(interp)
        lib.t_destroy(rec.h)
        if ci != cr or si != sr:
            diffs = [f"{NAMES[i]}: interp={si[i]:#x} rec={sr[i]:#x}" for i in range(29) if si[i] != sr[i]]
            # Trace the interpreter up to the last instruction executed
            print(f"  bisect: first divergence at budget {budget}; cycles interp={ci} rec={cr}; " + "; ".join(diffs))
            return
    print("  bisect: no divergence found")


def stepwise(seed, steps=600):
    """Runs the case one cycle at a time on both cores and reports the first instruction that makes them differ."""
    rng = random.Random(seed)
    code = gen_program(rng, rng.choice([16, 40, 90]))
    blob = b"".join(w.to_bytes(2, "big") for w in code)
    data = bytes(rng.getrandbits(8) for _ in range(0x8000))
    st = make_state(rng)
    interp = lib.t_create(0)
    rec = Recomp()
    for h in (interp, rec.h):
        lib.t_set_mem(h, RAM_BASE_H + 0x1000, blob, len(blob))
        lib.t_set_mem(h, RAM_BASE_H + 0x8000, data, len(data))
        lib.t_set_mem(h, RAM_BASE_H + 0x400, bytes(0x400), 0x400)
        write_state(h, st)
    for n in range(steps):
        before = read_state(interp)
        hi, _ = get_ram(interp)
        pc = before[16]
        off = (pc & 0xFFFFF) if (pc >> 24) == 6 else None
        op = int.from_bytes(hi[off:off + 2], "big") if off is not None and off + 2 <= len(hi) else None
        ci = lib.t_advance(interp, 1, 0)
        cr = lib.t_advance(rec.h, 1, 0)
        si, sr = read_state(interp), read_state(rec.h)
        if ci != cr or si != sr or get_ram(interp) != get_ram(rec.h):
            diffs = [f"{NAMES[i]}: interp={si[i]:#x} rec={sr[i]:#x}" for i in range(29) if si[i] != sr[i]]
            print(f"  stepwise: step {n} at pc={pc:#x} op={'?' if op is None else f'{op:04x}'} slot={before[26]} "
                  f"wb={before[23]}; cycles {ci}/{cr}; " + "; ".join(diffs))
            print(f"  before: R={[hex(x) for x in before[:16]]} SR={before[17]:#x} PR={before[20]:#x} GBR={before[18]:#x}")
            return
    print("  stepwise: no divergence")


def bisect_from(snap, maxb):
    st, hi, lo, level = snap
    for budget in range(1, maxb + 1):
        interp = lib.t_create(0)
        rec = Recomp()
        for h in (interp, rec.h):
            lib.t_set_intc_level(h, level)
            lib.t_set_mem(h, RAM_BASE_H, hi, len(hi))
            lib.t_set_mem(h, RAM_BASE_L, lo, len(lo))
            write_state(h, st)
        ci = lib.t_advance(interp, budget, 0)
        cr = lib.t_advance(rec.h, budget, 0)
        si, sr = read_state(interp), read_state(rec.h)
        ram_diff = get_ram(interp) != get_ram(rec.h)
        lib.t_destroy(interp)
        lib.t_destroy(rec.h)
        if ci != cr or si != sr or ram_diff:
            diffs = [f"{NAMES[i]}: interp={si[i]:#x} rec={sr[i]:#x}" for i in range(29) if si[i] != sr[i]]
            print(f"  snapshot bisect: budget {budget}: cycles {ci}/{cr} ram_diff={ram_diff}; " + "; ".join(diffs))
            print(f"  start state: PC={st[16]:#x} R={[hex(x) for x in st[:16]]} SR={st[17]:#x} wb={st[23]} "
                  f"slot={st[26]} pend={st[24]} allow={st[25]}")
            return budget
    print("  snapshot bisect: no divergence")


def run_case(seed, verbose=False):
    rng = random.Random(seed)
    code = gen_program(rng, rng.choice([16, 40, 90]))
    blob = b"".join(w.to_bytes(2, "big") for w in code)
    # Random data area
    data = bytes(rng.getrandbits(8) for _ in range(0x8000))
    st = make_state(rng)

    interp = lib.t_create(0)
    rec = Recomp()
    for h in (interp, rec.h):
        lib.t_set_mem(h, RAM_BASE_H + 0x1000, blob, len(blob))
        lib.t_set_mem(h, RAM_BASE_H + 0x8000, data, len(data))
        # Vector table used by exceptions
        lib.t_set_mem(h, RAM_BASE_H + 0x400, bytes(0x400), 0x400)
        write_state(h, st)
    level = 0
    if rng.random() < 0.3:
        level = rng.randrange(1, 16)
        for h in (interp, rec.h):
            lib.t_set_intc_level(h, level)

    snap = None
    for step in range(rng.choice([6, 12])):
        budget = rng.choice([1, 2, 3, 7, 20, 64, 200])
        if "FORCE_BUDGET%d" % step in os.environ:
            budget = int(os.environ["FORCE_BUDGET%d" % step])
        if rng.random() < 0.15:
            ia = read_state(interp)
            ia[24] = 1
            ia[25] = rng.getrandbits(1)
            write_state(interp, ia)
            write_state(rec.h, ia)
        hi_, lo_ = get_ram(interp)
        snap = (read_state(interp), hi_, lo_, level)
        if os.environ.get("TRACE_STEP") == str(step):
            tot = 0
            lines = []
            while tot < budget and len(lines) < 200:
                before = read_state(interp)
                hi2, _ = get_ram(interp)
                pc = before[16]
                off = pc & 0xFFFFF
                op = int.from_bytes(hi2[off:off + 2], "big") if (pc >> 24) == 6 else None
                c = lib.t_advance(interp, 1, 0)
                tot += c
                after = read_state(interp)
                lines.append(f"   I pc={pc:#x} op={'?' if op is None else f'{op:04x}'} slot={before[26]} wb={before[23]} "
                             f"pend={before[24]}/{after[24]} allow={before[25]} c={c} tot={tot}")
            print("\n".join(lines))
            print("   -- recompiler:")
            cr = lib.t_advance(rec.h, budget, 0)
            print("   rec cycles", cr, "pc", hex(read_state(rec.h)[16]))
            return "traced", None
        ci = lib.t_advance(interp, budget, 0)
        cr = lib.t_advance(rec.h, budget, 0)
        si, sr = read_state(interp), read_state(rec.h)
        if rec.errors:
            return f"seed {seed} step {step}: {rec.errors[0]}", None
        if ci != cr or si != sr:
            diffs = [f"{NAMES[i]}: interp={si[i]:#x} rec={sr[i]:#x}" for i in range(29) if si[i] != sr[i]]
            SNAPS[seed] = (snap, budget)
            if os.environ.get("DUMP_CODE"):
                hi_i, _ = get_ram(interp)
                hi_r, _ = get_ram(rec.h)
                print("   code words interp:", " ".join(f"{int.from_bytes(hi_i[0x1000+2*i:0x1002+2*i], 'big'):04x}" for i in range(30)))
                print("   code words rec:   ", " ".join(f"{int.from_bytes(hi_r[0x1000+2*i:0x1002+2*i], 'big'):04x}" for i in range(30)))
            return (f"seed {seed} step {step} budget {budget}: cycles interp={ci} rec={cr}; " + "; ".join(diffs),
                    code)
        if get_ram(interp) != get_ram(rec.h):
            return f"seed {seed} step {step}: RAM differs", code
    stats = (u64 * 8)()
    lib.t_stats(rec.h, stats)
    lib.t_destroy(interp)
    lib.t_destroy(rec.h)
    if TRACE:
        print("case", seed, list(stats))
    return None, list(stats)


def directed():
    """Hand-written cases."""
    ok = True

    def check(name, code, regs, budget=40, expect=None):
        nonlocal ok
        blob = b"".join(w.to_bytes(2, "big") for w in code)
        st = [0] * 29
        for i, v in regs.items():
            st[i] = v
        st[15] = RAM_BASE_H + 0x20000
        st[16] = CODE_ADDR
        interp = lib.t_create(0)
        rec = Recomp()
        for h in (interp, rec.h):
            lib.t_set_mem(h, RAM_BASE_H + 0x1000, blob, len(blob))
            write_state(h, st)
        ci = lib.t_advance(interp, budget, 0)
        cr = lib.t_advance(rec.h, budget, 0)
        si, sr = read_state(interp), read_state(rec.h)
        good = ci == cr and si == sr and not rec.errors
        if expect:
            for reg, val in expect.items():
                good = good and si[reg] == val
        print(("ok   " if good else "FAIL ") + name, "" if good else f"interp={[hex(x) for x in si[:8]]} rec={[hex(x) for x in sr[:8]]} "
              f"cycles {ci}/{cr} {rec.errors[:1]}")
        stats = (u64 * 8)()
        lib.t_stats(rec.h, stats)
        print(f"     ({stats[0]} blocks compiled, {stats[1]} entered from the interpreter, {stats[7]} jump sites)")
        if stats[1] == 0:
            print("FAIL " + name + ": the recompiler never ran")
            good = False
        lib.t_destroy(interp)
        lib.t_destroy(rec.h)
        ok = ok and good

    # The store rewrites an instruction further down the same block: MOV #0x11,R3 becomes MOV #7,R3
    check("self-modifying store inside a block",
          [0x2201, 0x0009, 0x0009, 0x0009, 0xE311, 0x0009, 0x0009],
          {0: 0xE307, 2: CODE_ADDR + 8}, expect={3: 7})
    # Same with a delay slot instruction being rewritten
    check("rewrite of a delay slot",
          [0x2201, 0x0009, 0xA002, 0xE311, 0x0009, 0x0009, 0xE411],
          {0: 0xE307, 2: CODE_ADDR + 6}, expect={3: 7})
    # A loop that rewrites its own first instruction: MOV #1,R3 becomes MOV #9,R3 on the second pass
    check("loop rewriting its head",
          [0xE301, 0x2201, 0x4410, 0x8BFB, 0x0009, 0x0009],
          {0: 0xE309, 2: CODE_ADDR, 4: 2}, expect={3: 9})
    # Two blocks that jump into each other (the second is too far away to be part of the first one)
    pingpong = [0x7301, 0xA043, 0x0009] + [0x0009] * 67          # 0x1000: ADD #1,R3 / BRA to 0x108C / NOP
    pingpong += [0x7402, 0x4510, 0x8BB6]      # 0x108C: ADD #2,R4 / DT R5 / BF back to 0x1000
    pingpong += [0x0009, 0x0009]
    check("blocks jumping into each other", pingpong, {5: 40}, budget=900)
    # The same, but the first block rewrites the first instruction of the second one on its way
    rewrite = [0x7301, 0x2201, 0xA042, 0x0009] + [0x0009] * 66
    rewrite += [0x7402, 0x4510, 0x8BB6, 0x0009, 0x0009]
    check("a block rewrites the block it jumps to", rewrite, {0: 0x7405, 2: CODE_ADDR + 0x8C, 5: 40}, budget=900)
    # Backward branch loop that runs many times
    check("counted loop", [0xE30A, 0x4310, 0x8FFE, 0x0009, 0x0009], {}, budget=200)
    return ok


def main():
    global ILLEGAL
    # The "illegal" opcode type value: find via a known illegal word
    ILLEGAL = lib.t_opcode_type(0x0000, 0)  # 0x0000 is illegal on SH-2
    if not os.environ.get("SKIP_DIRECTED") and not directed():
        sys.exit(1)
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    failures = 0
    tot = [0] * 8
    for i in range(iters):
        err, extra = run_case(seed0 + i)
        if err:
            failures += 1
            print("FAIL", err)
            if os.environ.get("STEPWISE"):
                stepwise(seed0 + i)
            elif (seed0 + i) in SNAPS:
                bisect_from(SNAPS[seed0 + i][0], SNAPS[seed0 + i][1])
            if extra and failures <= 3:
                print("  code:", " ".join(f"{w:04x}" for w in extra[:40]))
        elif extra:
            tot = [a + b for a, b in zip(tot, extra)]
    if tot[1] == 0:
        print("FAIL: the recompiler never ran")
        failures += 1
    print(f"{iters - failures}/{iters} cases matched")
    print("recompiler stats: compiled=%d executed=%d rejected=%d invalidations=%d flushes=%d inline=%d helper=%d chain=%d" %
          tuple(tot))
    sys.exit(1 if failures else 0)


main()
