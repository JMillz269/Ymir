#!/usr/bin/env python3
"""Checks the ARM64 assembler's encodings against Keystone, and its constant-loading sequences by executing them in
Unicorn. Usage: check_encodings.py <path to dump_encodings binary>"""
import re
import subprocess
import sys

from keystone import KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN, Ks
from unicorn import UC_ARCH_ARM64, UC_MODE_ARM, Uc
from unicorn.arm64_const import UC_ARM64_REG_X1

ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
out = subprocess.run([sys.argv[1]], capture_output=True, text=True, check=True).stdout.splitlines()

failures = 0
checked = 0
movimm = {}
for line in out:
    text, rest = line.split("|", 1)
    if text.startswith("@movimm"):
        movimm.setdefault(text, []).append(int(rest, 16))
        continue
    m = re.match(r"([0-9a-f]{8})(?: #(\d+))?$", rest)
    word, index = int(m.group(1), 16), int(m.group(2) or 0)
    addr = 0
    if text.endswith("@4 word1"):
        text, addr, expect_index = text.replace(" @4 word1", ""), 4, 1
    else:
        expect_index = 0
    if index != expect_index:
        continue
    enc, _ = ks.asm(text, addr)
    expected = int.from_bytes(bytes(enc[:4]), "little")
    checked += 1
    if expected != word:
        failures += 1
        print(f"MISMATCH: {text!r}: ours {word:08x}, keystone {expected:08x}")

def run(words):
    uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
    code = b"".join(w.to_bytes(4, "little") for w in words)
    uc.mem_map(0x1000, 0x1000)
    uc.mem_write(0x1000, code)
    uc.emu_start(0x1000, 0x1000 + len(code))
    return uc.reg_read(UC_ARM64_REG_X1)

for key, words in movimm.items():
    kind, value = key.split()
    value = int(value, 16)
    result = run(words)
    if kind == "@movimm32":
        result &= 0xFFFFFFFF
    checked += 1
    if result != value:
        failures += 1
        print(f"MOVIMM MISMATCH: {key}: got {result:x}")

print(f"{checked} encodings checked, {failures} failures")
sys.exit(1 if failures else 0)
