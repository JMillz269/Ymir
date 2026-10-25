#pragma once

/**
@file
@brief A small ARM64 (AArch64) assembler used by the dynamic recompilers.

Only the instructions the recompilers need are provided. Every encoding is checked against an independent assembler by
the tests in tools/jit-tests.
*/

#include <ymir/core/types.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace ymir::jit::arm64 {

/// @brief Register index (0-30 are X0-X30, 31 is XZR or SP depending on the instruction).
using Reg = uint32;

inline constexpr Reg X0 = 0, X1 = 1, X2 = 2, X3 = 3, X4 = 4, X5 = 5, X6 = 6, X7 = 7, X8 = 8, X9 = 9, X10 = 10, X11 = 11,
                     X12 = 12, X13 = 13, X14 = 14, X15 = 15, X16 = 16, X17 = 17, X18 = 18, X19 = 19, X20 = 20, X21 = 21,
                     X22 = 22, X23 = 23, X24 = 24, X25 = 25, X26 = 26, X27 = 27, X28 = 28, X29 = 29, X30 = 30;
inline constexpr Reg FP = 29;
inline constexpr Reg LR = 30;
inline constexpr Reg ZR = 31; ///< Zero register (reads as zero, writes are discarded)
inline constexpr Reg SP = 31; ///< Stack pointer (where an instruction takes SP instead of ZR)

/// @brief Operand size.
enum class Sz : uint32 { W = 0, X = 1 };

enum class Cond : uint32 {
    EQ = 0,
    NE = 1,
    CS = 2, // also HS
    CC = 3, // also LO
    MI = 4,
    PL = 5,
    VS = 6,
    VC = 7,
    HI = 8,
    LS = 9,
    GE = 10,
    LT = 11,
    GT = 12,
    LE = 13,
    AL = 14,
};

enum class Shift : uint32 { LSL = 0, LSR = 1, ASR = 2, ROR = 3 };

/// @brief Extension applied to the offset register of register-offset loads and stores.
enum class Extend : uint32 { UXTW = 2, LSL = 3, SXTW = 6, SXTX = 7 };

/// @brief Returns the inverse of a condition code.
constexpr Cond Invert(Cond cond) {
    return static_cast<Cond>(static_cast<uint32>(cond) ^ 1u);
}

/// @brief A position in the code that branches can target.
struct Label {
    sint32 position = -1; ///< Word index of the bound label, or -1 if it is not bound yet
};

/// @brief Encodes a value as an ARM64 logical immediate (bitmask immediate).
/// @param[in] value the value to encode
/// @param[in] is64 whether the operation is 64-bit; for 32-bit operations the value must be a 32-bit pattern
/// @param[out] n the N field
/// @param[out] immr the immr field
/// @param[out] imms the imms field
/// @return true if the value can be encoded
inline bool EncodeBitmaskImmediate(uint64 value, bool is64, uint32 &n, uint32 &immr, uint32 &imms) {
    if (!is64) {
        value &= 0xFFFFFFFFull;
        value |= value << 32;
    }
    if (value == 0 || value == ~0ull) {
        return false;
    }

    // Find the smallest element size that repeats across the whole value
    uint32 size = 64;
    while (size > 2) {
        const uint32 half = size / 2;
        const uint64 mask = (1ull << half) - 1;
        if ((value & mask) != ((value >> half) & mask)) {
            break;
        }
        size = half;
    }
    const uint64 elemMask = size == 64 ? ~0ull : ((1ull << size) - 1);
    uint64 elem = value & elemMask;

    // The element must be a (rotated) run of ones
    uint32 rotation = 0;
    // Rotate right until the run of ones starts at bit 0 (bit 0 set and the bit above the run is clear)
    auto rotr = [&](uint64 v, uint32 amount) -> uint64 {
        amount %= size;
        if (amount == 0) {
            return v;
        }
        return ((v >> amount) | (v << (size - amount))) & elemMask;
    };
    bool found = false;
    for (uint32 r = 0; r < size; ++r) {
        const uint64 rotated = rotr(elem, r);
        // A run of ones starting at bit 0 has the form 2^k - 1
        if ((rotated & (rotated + 1)) == 0 && rotated != 0) {
            rotation = r;
            elem = rotated;
            found = true;
            break;
        }
    }
    if (!found) {
        return false;
    }
    uint32 ones = 0;
    while ((elem >> ones) & 1) {
        ++ones;
    }

    // immr is the right rotation that turns the run starting at bit 0 into the element: rotating the run left by
    // `rotation` bits gives back the element, so immr = (size - rotation) % size
    immr = (size - rotation) % size;
    n = size == 64 ? 1 : 0;
    // imms encodes the element size in its leading bits (pattern 0b11xxxx for 2-bit, 0b110xxx..., ) followed by ones-1
    uint32 sizeBits = 0;
    switch (size) {
    case 2: sizeBits = 0b111100; break;
    case 4: sizeBits = 0b111000; break;
    case 8: sizeBits = 0b110000; break;
    case 16: sizeBits = 0b100000; break;
    case 32: sizeBits = 0b000000; break;
    case 64: sizeBits = 0b000000; break;
    }
    imms = (sizeBits | (ones - 1)) & 0x3F;
    return true;
}

/// @brief Writes ARM64 instructions into a buffer of 32-bit words.
class Assembler {
public:
    /// @brief Creates an assembler writing into the given buffer.
    /// @param[in] buffer the buffer to write instructions into
    /// @param[in] capacityWords the capacity of the buffer, in 32-bit words
    Assembler(uint32 *buffer, size_t capacityWords)
        : m_begin(buffer)
        , m_cur(buffer)
        , m_end(buffer + capacityWords) {}

    /// @brief Number of words written so far.
    size_t Size() const {
        return static_cast<size_t>(m_cur - m_begin);
    }

    /// @brief Whether the buffer ran out of space. When true the code is incomplete and must be discarded.
    bool Overflowed() const {
        return m_overflow;
    }

    uint32 *Begin() const {
        return m_begin;
    }

    /// @brief Resolves all pending label references. Returns false if a branch is out of range or a label is unbound.
    bool Finalize() {
        for (const Fixup &fx : m_fixups) {
            if (fx.label->position < 0) {
                return false;
            }
            const sint64 delta = static_cast<sint64>(fx.label->position) - static_cast<sint64>(fx.index);
            uint32 &word = m_begin[fx.index];
            switch (fx.kind) {
            case FixupKind::Imm26:
                if (delta < -(1 << 25) || delta >= (1 << 25)) {
                    return false;
                }
                word = (word & ~0x03FFFFFFu) | (static_cast<uint32>(delta) & 0x03FFFFFFu);
                break;
            case FixupKind::Imm19:
                if (delta < -(1 << 18) || delta >= (1 << 18)) {
                    return false;
                }
                word = (word & ~(0x7FFFFu << 5)) | ((static_cast<uint32>(delta) & 0x7FFFFu) << 5);
                break;
            case FixupKind::Imm14:
                if (delta < -(1 << 13) || delta >= (1 << 13)) {
                    return false;
                }
                word = (word & ~(0x3FFFu << 5)) | ((static_cast<uint32>(delta) & 0x3FFFu) << 5);
                break;
            }
        }
        m_fixups.clear();
        return !m_overflow;
    }

    /// @brief Binds a label to the current position.
    void Bind(Label &label) {
        label.position = static_cast<sint32>(Size());
    }

    /// @brief Emits a raw instruction word.
    void Emit(uint32 word) {
        if (m_cur == m_end) {
            m_overflow = true;
            return;
        }
        *m_cur++ = word;
    }

    // -- Moves --

    void Movz(Sz sz, Reg rd, uint32 imm16, uint32 shift16 = 0) {
        Emit(0x52800000u | S(sz) | (shift16 << 21) | (imm16 << 5) | rd);
    }
    void Movn(Sz sz, Reg rd, uint32 imm16, uint32 shift16 = 0) {
        Emit(0x12800000u | S(sz) | (shift16 << 21) | (imm16 << 5) | rd);
    }
    void Movk(Sz sz, Reg rd, uint32 imm16, uint32 shift16 = 0) {
        Emit(0x72800000u | S(sz) | (shift16 << 21) | (imm16 << 5) | rd);
    }

    /// @brief Loads an arbitrary constant into a register using the shortest sequence of instructions.
    void MovImm(Sz sz, Reg rd, uint64 value) {
        const bool is64 = sz == Sz::X;
        if (!is64) {
            value &= 0xFFFFFFFFull;
        }
        const uint32 halves = is64 ? 4 : 2;
        const uint64 allOnes = 0xFFFFull;

        // Count the halfwords that are all zeros and all ones to choose between MOVZ+MOVK and MOVN+MOVK
        uint32 zeros = 0, ones = 0;
        for (uint32 i = 0; i < halves; ++i) {
            const uint64 half = (value >> (i * 16)) & allOnes;
            zeros += half == 0;
            ones += half == allOnes;
        }
        if (value == 0) {
            Movz(sz, rd, 0);
            return;
        }
        uint32 n, immr, imms;
        if (EncodeBitmaskImmediate(value, is64, n, immr, imms) && zeros < halves - 1 && ones < halves - 1) {
            // Single ORR with the zero register
            Emit(0x32000000u | S(sz) | (n << 22) | (immr << 16) | (imms << 10) | (ZR << 5) | rd);
            return;
        }
        const bool useMovn = ones > zeros;
        bool first = true;
        for (uint32 i = 0; i < halves; ++i) {
            uint64 half = (value >> (i * 16)) & allOnes;
            if (useMovn) {
                half ^= allOnes;
            }
            const bool skip = useMovn ? (((value >> (i * 16)) & allOnes) == allOnes) : (half == 0);
            if (skip) {
                continue;
            }
            if (first) {
                if (useMovn) {
                    Movn(sz, rd, static_cast<uint32>(half), i);
                } else {
                    Movz(sz, rd, static_cast<uint32>(half), i);
                }
                first = false;
            } else {
                Movk(sz, rd, static_cast<uint32>((value >> (i * 16)) & allOnes), i);
            }
        }
        if (first) {
            // Every halfword was skipped: the value is all ones
            Movn(sz, rd, 0);
        }
    }

    /// @brief Register-to-register move (ORR rd, ZR, rm). Use AddImm(..., 0) when moving to or from SP.
    void Mov(Sz sz, Reg rd, Reg rm) {
        Orr(sz, rd, ZR, rm);
    }

    // -- Add/subtract --

    void AddImm(Sz sz, Reg rd, Reg rn, uint32 imm12, bool shift12 = false) {
        Emit(0x11000000u | S(sz) | (shift12 ? 1u << 22 : 0) | (imm12 << 10) | (rn << 5) | rd);
    }
    void AddsImm(Sz sz, Reg rd, Reg rn, uint32 imm12, bool shift12 = false) {
        Emit(0x31000000u | S(sz) | (shift12 ? 1u << 22 : 0) | (imm12 << 10) | (rn << 5) | rd);
    }
    void SubImm(Sz sz, Reg rd, Reg rn, uint32 imm12, bool shift12 = false) {
        Emit(0x51000000u | S(sz) | (shift12 ? 1u << 22 : 0) | (imm12 << 10) | (rn << 5) | rd);
    }
    void SubsImm(Sz sz, Reg rd, Reg rn, uint32 imm12, bool shift12 = false) {
        Emit(0x71000000u | S(sz) | (shift12 ? 1u << 22 : 0) | (imm12 << 10) | (rn << 5) | rd);
    }

    void Add(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x0B000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Adds(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x2B000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Sub(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x4B000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Subs(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x6B000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }

    void Cmp(Sz sz, Reg rn, Reg rm) {
        Subs(sz, ZR, rn, rm);
    }
    void CmpImm(Sz sz, Reg rn, uint32 imm12) {
        SubsImm(sz, ZR, rn, imm12);
    }
    void Cmn(Sz sz, Reg rn, Reg rm) {
        Adds(sz, ZR, rn, rm);
    }
    void Neg(Sz sz, Reg rd, Reg rm) {
        Sub(sz, rd, ZR, rm);
    }

    void Adc(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x1A000000u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }
    void Adcs(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x3A000000u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }
    void Sbc(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x5A000000u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }
    void Sbcs(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x7A000000u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }

    // -- Logical --

    void And(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x0A000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Orr(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x2A000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Eor(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x4A000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Ands(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x6A000000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Bic(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x0A200000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Orn(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x2A200000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Eon(Sz sz, Reg rd, Reg rn, Reg rm, Shift sh = Shift::LSL, uint32 amount = 0) {
        Emit(0x4A200000u | S(sz) | (U(sh) << 22) | (rm << 16) | (amount << 10) | (rn << 5) | rd);
    }
    void Mvn(Sz sz, Reg rd, Reg rm) {
        Orn(sz, rd, ZR, rm);
    }
    void Tst(Sz sz, Reg rn, Reg rm) {
        Ands(sz, ZR, rn, rm);
    }

    /// @brief Logical operations with a bitmask immediate. Return false (and emit nothing) if it can't be encoded.
    bool AndImm(Sz sz, Reg rd, Reg rn, uint64 imm) {
        return LogicalImm(0x12000000u, sz, rd, rn, imm);
    }
    bool OrrImm(Sz sz, Reg rd, Reg rn, uint64 imm) {
        return LogicalImm(0x32000000u, sz, rd, rn, imm);
    }
    bool EorImm(Sz sz, Reg rd, Reg rn, uint64 imm) {
        return LogicalImm(0x52000000u, sz, rd, rn, imm);
    }
    bool AndsImm(Sz sz, Reg rd, Reg rn, uint64 imm) {
        return LogicalImm(0x72000000u, sz, rd, rn, imm);
    }
    bool TstImm(Sz sz, Reg rn, uint64 imm) {
        return AndsImm(sz, ZR, rn, imm);
    }

    // -- Shifts and bitfields --

    void Lslv(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x1AC02000u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }
    void Lsrv(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x1AC02400u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }
    void Asrv(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x1AC02800u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }
    void Rorv(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x1AC02C00u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }

    void Sbfm(Sz sz, Reg rd, Reg rn, uint32 immr, uint32 imms) {
        Emit(0x13000000u | S(sz) | (N(sz) << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd);
    }
    void Bfm(Sz sz, Reg rd, Reg rn, uint32 immr, uint32 imms) {
        Emit(0x33000000u | S(sz) | (N(sz) << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd);
    }
    void Ubfm(Sz sz, Reg rd, Reg rn, uint32 immr, uint32 imms) {
        Emit(0x53000000u | S(sz) | (N(sz) << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd);
    }

    void LslImm(Sz sz, Reg rd, Reg rn, uint32 amount) {
        const uint32 bits = sz == Sz::X ? 64 : 32;
        Ubfm(sz, rd, rn, (bits - amount) % bits, bits - 1 - amount);
    }
    void LsrImm(Sz sz, Reg rd, Reg rn, uint32 amount) {
        Ubfm(sz, rd, rn, amount, sz == Sz::X ? 63 : 31);
    }
    void AsrImm(Sz sz, Reg rd, Reg rn, uint32 amount) {
        Sbfm(sz, rd, rn, amount, sz == Sz::X ? 63 : 31);
    }
    void RorImm(Sz sz, Reg rd, Reg rn, uint32 amount) {
        Extr(sz, rd, rn, rn, amount);
    }
    void Extr(Sz sz, Reg rd, Reg rn, Reg rm, uint32 lsb) {
        Emit(0x13800000u | S(sz) | (N(sz) << 22) | (rm << 16) | (lsb << 10) | (rn << 5) | rd);
    }

    void Ubfx(Sz sz, Reg rd, Reg rn, uint32 lsb, uint32 width) {
        Ubfm(sz, rd, rn, lsb, lsb + width - 1);
    }
    void Sbfx(Sz sz, Reg rd, Reg rn, uint32 lsb, uint32 width) {
        Sbfm(sz, rd, rn, lsb, lsb + width - 1);
    }
    /// @brief Inserts the low `width` bits of rn into rd at bit `lsb`.
    void Bfi(Sz sz, Reg rd, Reg rn, uint32 lsb, uint32 width) {
        const uint32 bits = sz == Sz::X ? 64 : 32;
        Bfm(sz, rd, rn, (bits - lsb) % bits, width - 1);
    }
    /// @brief Copies `width` bits of rn starting at `lsb` into the low bits of rd.
    void Bfxil(Sz sz, Reg rd, Reg rn, uint32 lsb, uint32 width) {
        Bfm(sz, rd, rn, lsb, lsb + width - 1);
    }

    void Uxtb(Reg rd, Reg rn) {
        Ubfm(Sz::W, rd, rn, 0, 7);
    }
    void Uxth(Reg rd, Reg rn) {
        Ubfm(Sz::W, rd, rn, 0, 15);
    }
    void Sxtb(Sz sz, Reg rd, Reg rn) {
        Sbfm(sz, rd, rn, 0, 7);
    }
    void Sxth(Sz sz, Reg rd, Reg rn) {
        Sbfm(sz, rd, rn, 0, 15);
    }
    void Sxtw(Reg rd, Reg rn) {
        Sbfm(Sz::X, rd, rn, 0, 31);
    }

    // -- Bit manipulation --

    void Rbit(Sz sz, Reg rd, Reg rn) {
        Emit(0x5AC00000u | S(sz) | (rn << 5) | rd);
    }
    void Rev16(Sz sz, Reg rd, Reg rn) {
        Emit(0x5AC00400u | S(sz) | (rn << 5) | rd);
    }
    void Rev(Sz sz, Reg rd, Reg rn) {
        Emit((sz == Sz::X ? 0x5AC00C00u : 0x5AC00800u) | S(sz) | (rn << 5) | rd);
    }
    void Clz(Sz sz, Reg rd, Reg rn) {
        Emit(0x5AC01000u | S(sz) | (rn << 5) | rd);
    }

    // -- Multiply and divide --

    void Madd(Sz sz, Reg rd, Reg rn, Reg rm, Reg ra) {
        Emit(0x1B000000u | S(sz) | (rm << 16) | (ra << 10) | (rn << 5) | rd);
    }
    void Msub(Sz sz, Reg rd, Reg rn, Reg rm, Reg ra) {
        Emit(0x1B008000u | S(sz) | (rm << 16) | (ra << 10) | (rn << 5) | rd);
    }
    void Mul(Sz sz, Reg rd, Reg rn, Reg rm) {
        Madd(sz, rd, rn, rm, ZR);
    }
    void Smaddl(Reg rd, Reg rn, Reg rm, Reg ra) {
        Emit(0x9B200000u | (rm << 16) | (ra << 10) | (rn << 5) | rd);
    }
    void Umaddl(Reg rd, Reg rn, Reg rm, Reg ra) {
        Emit(0x9BA00000u | (rm << 16) | (ra << 10) | (rn << 5) | rd);
    }
    void Smull(Reg rd, Reg rn, Reg rm) {
        Smaddl(rd, rn, rm, ZR);
    }
    void Umull(Reg rd, Reg rn, Reg rm) {
        Umaddl(rd, rn, rm, ZR);
    }
    void Smulh(Reg rd, Reg rn, Reg rm) {
        Emit(0x9B407C00u | (rm << 16) | (rn << 5) | rd);
    }
    void Umulh(Reg rd, Reg rn, Reg rm) {
        Emit(0x9BC07C00u | (rm << 16) | (rn << 5) | rd);
    }
    void Udiv(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x1AC00800u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }
    void Sdiv(Sz sz, Reg rd, Reg rn, Reg rm) {
        Emit(0x1AC00C00u | S(sz) | (rm << 16) | (rn << 5) | rd);
    }

    // -- Conditional select --

    void Csel(Sz sz, Reg rd, Reg rn, Reg rm, Cond cond) {
        Emit(0x1A800000u | S(sz) | (rm << 16) | (U(cond) << 12) | (rn << 5) | rd);
    }
    void Csinc(Sz sz, Reg rd, Reg rn, Reg rm, Cond cond) {
        Emit(0x1A800400u | S(sz) | (rm << 16) | (U(cond) << 12) | (rn << 5) | rd);
    }
    void Csinv(Sz sz, Reg rd, Reg rn, Reg rm, Cond cond) {
        Emit(0x5A800000u | S(sz) | (rm << 16) | (U(cond) << 12) | (rn << 5) | rd);
    }
    void Csneg(Sz sz, Reg rd, Reg rn, Reg rm, Cond cond) {
        Emit(0x5A800400u | S(sz) | (rm << 16) | (U(cond) << 12) | (rn << 5) | rd);
    }
    /// @brief rd = condition ? rn + 1 : rn
    void Cinc(Sz sz, Reg rd, Reg rn, Cond cond) {
        Csinc(sz, rd, rn, rn, Invert(cond));
    }
    /// @brief Conditional compare with an immediate: if the condition holds, compares rn with imm5, otherwise sets the
    /// flags to nzcv.
    void CcmpImm(Sz sz, Reg rn, uint32 imm5, uint32 nzcv, Cond cond) {
        Emit(0x7A400800u | S(sz) | (imm5 << 16) | (U(cond) << 12) | (rn << 5) | nzcv);
    }
    /// @brief Conditional compare of two registers.
    void Ccmp(Sz sz, Reg rn, Reg rm, uint32 nzcv, Cond cond) {
        Emit(0x7A400000u | S(sz) | (rm << 16) | (U(cond) << 12) | (rn << 5) | nzcv);
    }
    /// @brief Sets rd to 1 if the condition holds, 0 otherwise.
    void Cset(Sz sz, Reg rd, Cond cond) {
        Csinc(sz, rd, ZR, ZR, Invert(cond));
    }

    // -- Loads and stores --

    /// @brief Loads or stores with an unsigned, size-scaled immediate offset.
    /// @param[in] sizeLog2 0=byte, 1=halfword, 2=word, 3=doubleword
    void Ldr(uint32 sizeLog2, Reg rt, Reg rn, uint32 offset) {
        Emit(0x39400000u | (sizeLog2 << 30) | ((offset >> sizeLog2) << 10) | (rn << 5) | rt);
    }
    void Str(uint32 sizeLog2, Reg rt, Reg rn, uint32 offset) {
        Emit(0x39000000u | (sizeLog2 << 30) | ((offset >> sizeLog2) << 10) | (rn << 5) | rt);
    }
    /// @brief Sign-extending loads. `to64` selects a 64-bit destination.
    void Ldrs(uint32 sizeLog2, bool to64, Reg rt, Reg rn, uint32 offset) {
        Emit((to64 ? 0x39800000u : 0x39C00000u) | (sizeLog2 << 30) | ((offset >> sizeLog2) << 10) | (rn << 5) | rt);
    }
    /// @brief Loads or stores with an unscaled signed 9-bit offset.
    void Ldur(uint32 sizeLog2, Reg rt, Reg rn, sint32 offset) {
        Emit(0x38400000u | (sizeLog2 << 30) | ((static_cast<uint32>(offset) & 0x1FFu) << 12) | (rn << 5) | rt);
    }
    void Stur(uint32 sizeLog2, Reg rt, Reg rn, sint32 offset) {
        Emit(0x38000000u | (sizeLog2 << 30) | ((static_cast<uint32>(offset) & 0x1FFu) << 12) | (rn << 5) | rt);
    }
    /// @brief Loads or stores with a register offset, optionally scaled by the access size.
    void LdrReg(uint32 sizeLog2, Reg rt, Reg rn, Reg rm, Extend ext = Extend::LSL, bool scaled = false) {
        Emit(0x38600800u | (sizeLog2 << 30) | (rm << 16) | (U(ext) << 13) | (scaled ? 1u << 12 : 0) | (rn << 5) | rt);
    }
    void StrReg(uint32 sizeLog2, Reg rt, Reg rn, Reg rm, Extend ext = Extend::LSL, bool scaled = false) {
        Emit(0x38200800u | (sizeLog2 << 30) | (rm << 16) | (U(ext) << 13) | (scaled ? 1u << 12 : 0) | (rn << 5) | rt);
    }

    /// @brief Load/store pair with a signed offset (a multiple of 4 or 8 depending on the size).
    void Ldp(Sz sz, Reg rt, Reg rt2, Reg rn, sint32 offset) {
        PairInsn(0x29400000u, sz, rt, rt2, rn, offset);
    }
    void Stp(Sz sz, Reg rt, Reg rt2, Reg rn, sint32 offset) {
        PairInsn(0x29000000u, sz, rt, rt2, rn, offset);
    }
    /// @brief Pair store with pre-index writeback (e.g. function prologue: STP x29, x30, [sp, #-16]!).
    void StpPre(Sz sz, Reg rt, Reg rt2, Reg rn, sint32 offset) {
        PairInsn(0x29800000u, sz, rt, rt2, rn, offset);
    }
    /// @brief Pair load with post-index writeback (e.g. function epilogue: LDP x29, x30, [sp], #16).
    void LdpPost(Sz sz, Reg rt, Reg rt2, Reg rn, sint32 offset) {
        PairInsn(0x28C00000u, sz, rt, rt2, rn, offset);
    }

    // -- Branches --

    void B(Label &label) {
        AddFixup(label, FixupKind::Imm26);
        Emit(0x14000000u);
    }
    void Bl(Label &label) {
        AddFixup(label, FixupKind::Imm26);
        Emit(0x94000000u);
    }
    void BCond(Cond cond, Label &label) {
        AddFixup(label, FixupKind::Imm19);
        Emit(0x54000000u | U(cond));
    }
    void Cbz(Sz sz, Reg rt, Label &label) {
        AddFixup(label, FixupKind::Imm19);
        Emit(0x34000000u | S(sz) | rt);
    }
    void Cbnz(Sz sz, Reg rt, Label &label) {
        AddFixup(label, FixupKind::Imm19);
        Emit(0x35000000u | S(sz) | rt);
    }
    void Tbz(Reg rt, uint32 bit, Label &label) {
        AddFixup(label, FixupKind::Imm14);
        Emit(0x36000000u | ((bit >> 5) << 31) | ((bit & 31u) << 19) | rt);
    }
    void Tbnz(Reg rt, uint32 bit, Label &label) {
        AddFixup(label, FixupKind::Imm14);
        Emit(0x37000000u | ((bit >> 5) << 31) | ((bit & 31u) << 19) | rt);
    }
    void Br(Reg rn) {
        Emit(0xD61F0000u | (rn << 5));
    }
    void Blr(Reg rn) {
        Emit(0xD63F0000u | (rn << 5));
    }
    void Ret(Reg rn = LR) {
        Emit(0xD65F0000u | (rn << 5));
    }

    void Nop() {
        Emit(0xD503201Fu);
    }
    void Brk(uint32 imm16) {
        Emit(0xD4200000u | (imm16 << 5));
    }

private:
    enum class FixupKind { Imm26, Imm19, Imm14 };
    struct Fixup {
        size_t index;
        Label *label;
        FixupKind kind;
    };

    template <typename T>
    static constexpr uint32 U(T value) {
        return static_cast<uint32>(value);
    }
    static constexpr uint32 S(Sz sz) {
        return sz == Sz::X ? 0x80000000u : 0u;
    }
    static constexpr uint32 N(Sz sz) {
        return sz == Sz::X ? 1u : 0u;
    }

    bool LogicalImm(uint32 base, Sz sz, Reg rd, Reg rn, uint64 imm) {
        uint32 n, immr, imms;
        if (!EncodeBitmaskImmediate(imm, sz == Sz::X, n, immr, imms)) {
            return false;
        }
        Emit(base | S(sz) | (n << 22) | (immr << 16) | (imms << 10) | (rn << 5) | rd);
        return true;
    }

    void PairInsn(uint32 base, Sz sz, Reg rt, Reg rt2, Reg rn, sint32 offset) {
        const uint32 scale = sz == Sz::X ? 3 : 2;
        const uint32 imm7 = (static_cast<uint32>(offset >> scale)) & 0x7Fu;
        Emit(base | (sz == Sz::X ? 0x80000000u : 0u) | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt);
    }

    void AddFixup(Label &label, FixupKind kind) {
        m_fixups.push_back({Size(), &label, kind});
    }

    uint32 *m_begin;
    uint32 *m_cur;
    uint32 *m_end;
    bool m_overflow = false;
    std::vector<Fixup> m_fixups;
};

} // namespace ymir::jit::arm64
