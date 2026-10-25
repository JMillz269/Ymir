// Emits instructions with ymir::jit::arm64::Assembler and prints "<assembly text>|<hex word>" lines, which
// check_encodings.py verifies against an independent assembler (Keystone).

#include <ymir/jit/arm64_assembler.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace ymir::jit::arm64;

namespace {

uint32 g_buf[16];

template <typename Fn>
void Case(const std::string &text, Fn &&fn) {
    Assembler a{g_buf, 16};
    fn(a);
    a.Finalize();
    for (size_t i = 0; i < a.Size(); ++i) {
        std::printf("%s|%08x%s\n", text.c_str(), g_buf[i], a.Size() > 1 ? (" #" + std::to_string(i)).c_str() : "");
    }
}

} // namespace

int main() {
    // -- Moves --
    Case("movz w1, #0x1234", [](Assembler &a) { a.Movz(Sz::W, 1, 0x1234); });
    Case("movz x2, #0xabcd, lsl #16", [](Assembler &a) { a.Movz(Sz::X, 2, 0xabcd, 1); });
    Case("movz x2, #0xabcd, lsl #32", [](Assembler &a) { a.Movz(Sz::X, 2, 0xabcd, 2); });
    Case("movk x3, #0x4321, lsl #48", [](Assembler &a) { a.Movk(Sz::X, 3, 0x4321, 3); });
    Case("movn w4, #0x10", [](Assembler &a) { a.Movn(Sz::W, 4, 0x10); });
    Case("mov x5, x6", [](Assembler &a) { a.Mov(Sz::X, 5, 6); });
    Case("mov w5, w6", [](Assembler &a) { a.Mov(Sz::W, 5, 6); });

    // -- Add/sub --
    Case("add w1, w2, #123", [](Assembler &a) { a.AddImm(Sz::W, 1, 2, 123); });
    Case("add x1, x2, #4095, lsl #12", [](Assembler &a) { a.AddImm(Sz::X, 1, 2, 4095, true); });
    Case("adds w1, w2, #7", [](Assembler &a) { a.AddsImm(Sz::W, 1, 2, 7); });
    Case("sub x1, sp, #16", [](Assembler &a) { a.SubImm(Sz::X, 1, SP, 16); });
    Case("subs x1, x2, #1", [](Assembler &a) { a.SubsImm(Sz::X, 1, 2, 1); });
    Case("add w1, w2, w3", [](Assembler &a) { a.Add(Sz::W, 1, 2, 3); });
    Case("add x1, x2, x3, lsl #4", [](Assembler &a) { a.Add(Sz::X, 1, 2, 3, Shift::LSL, 4); });
    Case("add w1, w2, w3, lsr #5", [](Assembler &a) { a.Add(Sz::W, 1, 2, 3, Shift::LSR, 5); });
    Case("adds w1, w2, w3", [](Assembler &a) { a.Adds(Sz::W, 1, 2, 3); });
    Case("sub w1, w2, w3, asr #3", [](Assembler &a) { a.Sub(Sz::W, 1, 2, 3, Shift::ASR, 3); });
    Case("subs w1, w2, w3", [](Assembler &a) { a.Subs(Sz::W, 1, 2, 3); });
    Case("cmp w1, w2", [](Assembler &a) { a.Cmp(Sz::W, 1, 2); });
    Case("cmp x1, #10", [](Assembler &a) { a.CmpImm(Sz::X, 1, 10); });
    Case("cmn w1, w2", [](Assembler &a) { a.Cmn(Sz::W, 1, 2); });
    Case("neg w1, w2", [](Assembler &a) { a.Neg(Sz::W, 1, 2); });
    Case("adc w1, w2, w3", [](Assembler &a) { a.Adc(Sz::W, 1, 2, 3); });
    Case("adcs w1, w2, w3", [](Assembler &a) { a.Adcs(Sz::W, 1, 2, 3); });
    Case("sbc w1, w2, w3", [](Assembler &a) { a.Sbc(Sz::W, 1, 2, 3); });
    Case("sbcs w1, w2, w3", [](Assembler &a) { a.Sbcs(Sz::W, 1, 2, 3); });

    // -- Logical --
    Case("and w1, w2, w3", [](Assembler &a) { a.And(Sz::W, 1, 2, 3); });
    Case("orr x1, x2, x3, lsl #8", [](Assembler &a) { a.Orr(Sz::X, 1, 2, 3, Shift::LSL, 8); });
    Case("eor w1, w2, w3, ror #7", [](Assembler &a) { a.Eor(Sz::W, 1, 2, 3, Shift::ROR, 7); });
    Case("ands w1, w2, w3", [](Assembler &a) { a.Ands(Sz::W, 1, 2, 3); });
    Case("bic w1, w2, w3", [](Assembler &a) { a.Bic(Sz::W, 1, 2, 3); });
    Case("orn w1, w2, w3", [](Assembler &a) { a.Orn(Sz::W, 1, 2, 3); });
    Case("eon w1, w2, w3", [](Assembler &a) { a.Eon(Sz::W, 1, 2, 3); });
    Case("mvn w1, w2", [](Assembler &a) { a.Mvn(Sz::W, 1, 2); });
    Case("tst w1, w2", [](Assembler &a) { a.Tst(Sz::W, 1, 2); });

    struct ImmCase {
        uint64 value;
        bool is64;
    };
    const ImmCase immCases[] = {{0xFF, false},       {0xFF00, false},        {0x0000FFFF, false}, {0xFFFF0000, false},
                                {0x1, false},        {0x80000000, false},    {0x7FFFFFFF, false}, {0x55555555, false},
                                {0xAAAAAAAA, false}, {0x0F0F0F0F, false},   {0x00FF00FF, false}, {0xFFFFFFFE, false},
                                {0x3, true},         {0xFFFFFFFF, true},     {0x1FFFFFFFFull, true},
                                {0xFFFF0000FFFF0000ull, true}, {0x8000000000000000ull, true}, {0xFFFFFFFFFFFFFFFEull, true}};
    for (const auto &c : immCases) {
        char text[96];
        std::snprintf(text, sizeof(text), "and %c1, %c2, #0x%llx", c.is64 ? 'x' : 'w', c.is64 ? 'x' : 'w',
                      static_cast<unsigned long long>(c.value));
        Case(text, [&](Assembler &a) { a.AndImm(c.is64 ? Sz::X : Sz::W, 1, 2, c.value); });
        std::snprintf(text, sizeof(text), "orr %c1, %c2, #0x%llx", c.is64 ? 'x' : 'w', c.is64 ? 'x' : 'w',
                      static_cast<unsigned long long>(c.value));
        Case(text, [&](Assembler &a) { a.OrrImm(c.is64 ? Sz::X : Sz::W, 1, 2, c.value); });
        std::snprintf(text, sizeof(text), "eor %c1, %c2, #0x%llx", c.is64 ? 'x' : 'w', c.is64 ? 'x' : 'w',
                      static_cast<unsigned long long>(c.value));
        Case(text, [&](Assembler &a) { a.EorImm(c.is64 ? Sz::X : Sz::W, 1, 2, c.value); });
    }
    Case("tst w1, #0x1", [](Assembler &a) { a.TstImm(Sz::W, 1, 1); });

    // -- Shifts and bitfields --
    Case("lsl w1, w2, w3", [](Assembler &a) { a.Lslv(Sz::W, 1, 2, 3); });
    Case("lsr x1, x2, x3", [](Assembler &a) { a.Lsrv(Sz::X, 1, 2, 3); });
    Case("asr w1, w2, w3", [](Assembler &a) { a.Asrv(Sz::W, 1, 2, 3); });
    Case("ror w1, w2, w3", [](Assembler &a) { a.Rorv(Sz::W, 1, 2, 3); });
    for (uint32 n : {1u, 2u, 8u, 16u, 31u}) {
        Case("lsl w1, w2, #" + std::to_string(n), [&](Assembler &a) { a.LslImm(Sz::W, 1, 2, n); });
        Case("lsr w1, w2, #" + std::to_string(n), [&](Assembler &a) { a.LsrImm(Sz::W, 1, 2, n); });
        Case("asr w1, w2, #" + std::to_string(n), [&](Assembler &a) { a.AsrImm(Sz::W, 1, 2, n); });
        Case("ror w1, w2, #" + std::to_string(n), [&](Assembler &a) { a.RorImm(Sz::W, 1, 2, n); });
    }
    Case("lsl x1, x2, #40", [](Assembler &a) { a.LslImm(Sz::X, 1, 2, 40); });
    Case("lsr x1, x2, #40", [](Assembler &a) { a.LsrImm(Sz::X, 1, 2, 40); });
    Case("extr w1, w2, w3, #16", [](Assembler &a) { a.Extr(Sz::W, 1, 2, 3, 16); });
    Case("ubfx w1, w2, #4, #8", [](Assembler &a) { a.Ubfx(Sz::W, 1, 2, 4, 8); });
    Case("sbfx w1, w2, #4, #8", [](Assembler &a) { a.Sbfx(Sz::W, 1, 2, 4, 8); });
    Case("bfi w1, w2, #8, #16", [](Assembler &a) { a.Bfi(Sz::W, 1, 2, 8, 16); });
    Case("bfxil w1, w2, #8, #16", [](Assembler &a) { a.Bfxil(Sz::W, 1, 2, 8, 16); });
    Case("uxtb w1, w2", [](Assembler &a) { a.Uxtb(1, 2); });
    Case("uxth w1, w2", [](Assembler &a) { a.Uxth(1, 2); });
    Case("sxtb w1, w2", [](Assembler &a) { a.Sxtb(Sz::W, 1, 2); });
    Case("sxth w1, w2", [](Assembler &a) { a.Sxth(Sz::W, 1, 2); });
    Case("sxtb x1, w2", [](Assembler &a) { a.Sxtb(Sz::X, 1, 2); });
    Case("sxtw x1, w2", [](Assembler &a) { a.Sxtw(1, 2); });

    // -- Bit manipulation --
    Case("rbit w1, w2", [](Assembler &a) { a.Rbit(Sz::W, 1, 2); });
    Case("rev16 w1, w2", [](Assembler &a) { a.Rev16(Sz::W, 1, 2); });
    Case("rev w1, w2", [](Assembler &a) { a.Rev(Sz::W, 1, 2); });
    Case("rev x1, x2", [](Assembler &a) { a.Rev(Sz::X, 1, 2); });
    Case("clz w1, w2", [](Assembler &a) { a.Clz(Sz::W, 1, 2); });

    // -- Multiply/divide --
    Case("madd w1, w2, w3, w4", [](Assembler &a) { a.Madd(Sz::W, 1, 2, 3, 4); });
    Case("msub x1, x2, x3, x4", [](Assembler &a) { a.Msub(Sz::X, 1, 2, 3, 4); });
    Case("mul w1, w2, w3", [](Assembler &a) { a.Mul(Sz::W, 1, 2, 3); });
    Case("smaddl x1, w2, w3, x4", [](Assembler &a) { a.Smaddl(1, 2, 3, 4); });
    Case("umaddl x1, w2, w3, x4", [](Assembler &a) { a.Umaddl(1, 2, 3, 4); });
    Case("smull x1, w2, w3", [](Assembler &a) { a.Smull(1, 2, 3); });
    Case("umull x1, w2, w3", [](Assembler &a) { a.Umull(1, 2, 3); });
    Case("smulh x1, x2, x3", [](Assembler &a) { a.Smulh(1, 2, 3); });
    Case("umulh x1, x2, x3", [](Assembler &a) { a.Umulh(1, 2, 3); });
    Case("udiv w1, w2, w3", [](Assembler &a) { a.Udiv(Sz::W, 1, 2, 3); });
    Case("sdiv x1, x2, x3", [](Assembler &a) { a.Sdiv(Sz::X, 1, 2, 3); });

    // -- Conditional select --
    const struct {
        const char *name;
        Cond cond;
    } conds[] = {{"eq", Cond::EQ}, {"ne", Cond::NE}, {"hs", Cond::CS}, {"lo", Cond::CC}, {"mi", Cond::MI},
                 {"pl", Cond::PL}, {"vs", Cond::VS}, {"vc", Cond::VC}, {"hi", Cond::HI}, {"ls", Cond::LS},
                 {"ge", Cond::GE}, {"lt", Cond::LT}, {"gt", Cond::GT}, {"le", Cond::LE}};
    for (const auto &c : conds) {
        Case(std::string("csel w1, w2, w3, ") + c.name, [&](Assembler &a) { a.Csel(Sz::W, 1, 2, 3, c.cond); });
        Case(std::string("csinc x1, x2, x3, ") + c.name, [&](Assembler &a) { a.Csinc(Sz::X, 1, 2, 3, c.cond); });
        Case(std::string("csinv w1, w2, w3, ") + c.name, [&](Assembler &a) { a.Csinv(Sz::W, 1, 2, 3, c.cond); });
        Case(std::string("csneg w1, w2, w3, ") + c.name, [&](Assembler &a) { a.Csneg(Sz::W, 1, 2, 3, c.cond); });
        Case(std::string("cset w1, ") + c.name, [&](Assembler &a) { a.Cset(Sz::W, 1, c.cond); });
    }

    Case("cinc w1, w2, eq", [](Assembler &a) { a.Cinc(Sz::W, 1, 2, Cond::EQ); });
    Case("cinc x1, x2, ne", [](Assembler &a) { a.Cinc(Sz::X, 1, 2, Cond::NE); });
    Case("ccmp w1, #5, #4, ne", [](Assembler &a) { a.CcmpImm(Sz::W, 1, 5, 4, Cond::NE); });
    Case("ccmp x1, #31, #0, eq", [](Assembler &a) { a.CcmpImm(Sz::X, 1, 31, 0, Cond::EQ); });

    // -- Loads and stores --
    Case("ldrb w1, [x2, #5]", [](Assembler &a) { a.Ldr(0, 1, 2, 5); });
    Case("ldrh w1, [x2, #6]", [](Assembler &a) { a.Ldr(1, 1, 2, 6); });
    Case("ldr w1, [x2, #8]", [](Assembler &a) { a.Ldr(2, 1, 2, 8); });
    Case("ldr x1, [x2, #16]", [](Assembler &a) { a.Ldr(3, 1, 2, 16); });
    Case("ldr w1, [x2, #16380]", [](Assembler &a) { a.Ldr(2, 1, 2, 16380); });
    Case("strb w1, [x2, #5]", [](Assembler &a) { a.Str(0, 1, 2, 5); });
    Case("strh w1, [x2, #6]", [](Assembler &a) { a.Str(1, 1, 2, 6); });
    Case("str w1, [x2, #8]", [](Assembler &a) { a.Str(2, 1, 2, 8); });
    Case("str x1, [x2, #16]", [](Assembler &a) { a.Str(3, 1, 2, 16); });
    Case("ldrsb w1, [x2, #3]", [](Assembler &a) { a.Ldrs(0, false, 1, 2, 3); });
    Case("ldrsb x1, [x2, #3]", [](Assembler &a) { a.Ldrs(0, true, 1, 2, 3); });
    Case("ldrsh w1, [x2, #4]", [](Assembler &a) { a.Ldrs(1, false, 1, 2, 4); });
    Case("ldrsh x1, [x2, #4]", [](Assembler &a) { a.Ldrs(1, true, 1, 2, 4); });
    Case("ldrsw x1, [x2, #8]", [](Assembler &a) { a.Ldrs(2, true, 1, 2, 8); });
    Case("ldur w1, [x2, #-4]", [](Assembler &a) { a.Ldur(2, 1, 2, -4); });
    Case("stur x1, [x2, #-8]", [](Assembler &a) { a.Stur(3, 1, 2, -8); });
    Case("ldrb w1, [x2, x3]", [](Assembler &a) { a.LdrReg(0, 1, 2, 3, Extend::LSL, false); });
    Case("ldr w1, [x2, x3, lsl #2]", [](Assembler &a) { a.LdrReg(2, 1, 2, 3, Extend::LSL, true); });
    Case("ldr w1, [x2, w3, uxtw]", [](Assembler &a) { a.LdrReg(2, 1, 2, 3, Extend::UXTW, false); });
    Case("ldr w1, [x2, w3, sxtw #2]", [](Assembler &a) { a.LdrReg(2, 1, 2, 3, Extend::SXTW, true); });
    Case("str x1, [x2, x3, lsl #3]", [](Assembler &a) { a.StrReg(3, 1, 2, 3, Extend::LSL, true); });
    Case("strh w1, [x2, w3, uxtw]", [](Assembler &a) { a.StrReg(1, 1, 2, 3, Extend::UXTW, false); });
    Case("ldp w1, w2, [x3, #8]", [](Assembler &a) { a.Ldp(Sz::W, 1, 2, 3, 8); });
    Case("ldp x1, x2, [x3, #-16]", [](Assembler &a) { a.Ldp(Sz::X, 1, 2, 3, -16); });
    Case("stp x29, x30, [sp, #-16]!", [](Assembler &a) { a.StpPre(Sz::X, 29, 30, SP, -16); });
    Case("ldp x29, x30, [sp], #16", [](Assembler &a) { a.LdpPost(Sz::X, 29, 30, SP, 16); });
    Case("stp w1, w2, [x3, #20]", [](Assembler &a) { a.Stp(Sz::W, 1, 2, 3, 20); });

    // -- Branches (label offsets are checked as relative encodings: each case branches 3 words forward) --
    Case("b #12", [](Assembler &a) {
        Label l;
        a.B(l);
        a.Nop();
        a.Nop();
        a.Bind(l);
    });
    Case("bl #12", [](Assembler &a) {
        Label l;
        a.Bl(l);
        a.Nop();
        a.Nop();
        a.Bind(l);
    });
    for (const auto &c : conds) {
        Case(std::string("b.") + c.name + " #12", [&](Assembler &a) {
            Label l;
            a.BCond(c.cond, l);
            a.Nop();
            a.Nop();
            a.Bind(l);
        });
    }
    Case("cbz w1, #12", [](Assembler &a) {
        Label l;
        a.Cbz(Sz::W, 1, l);
        a.Nop();
        a.Nop();
        a.Bind(l);
    });
    Case("cbnz x1, #12", [](Assembler &a) {
        Label l;
        a.Cbnz(Sz::X, 1, l);
        a.Nop();
        a.Nop();
        a.Bind(l);
    });
    Case("tbz w1, #5, #12", [](Assembler &a) {
        Label l;
        a.Tbz(1, 5, l);
        a.Nop();
        a.Nop();
        a.Bind(l);
    });
    Case("tbnz x1, #40, #12", [](Assembler &a) {
        Label l;
        a.Tbnz(1, 40, l);
        a.Nop();
        a.Nop();
        a.Bind(l);
    });
    Case("b #0 @4 word1", [](Assembler &a) {
        Label l;
        a.Bind(l);
        a.Nop();
        a.B(l);
    });
    Case("br x16", [](Assembler &a) { a.Br(16); });
    Case("blr x17", [](Assembler &a) { a.Blr(17); });
    Case("ret", [](Assembler &a) { a.Ret(); });
    Case("nop", [](Assembler &a) { a.Nop(); });
    Case("brk #0xf00d", [](Assembler &a) { a.Brk(0xF00D); });

    // -- Constant loads (MovImm picks different sequences; check each with a Python-side evaluation) --
    for (uint64 v : {0ull, 1ull, 0xFFFFull, 0x10000ull, 0xDEADBEEFull, 0xFFFFFFFFull, 0x123456789ABCDEF0ull,
                     0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFF0000ull, 0x00000000FFFF0000ull, 0x0000FFFF0000FFFFull,
                     0x5555555555555555ull, 0x8000000000000000ull, 0xFFFF0000FFFF1234ull}) {
        Assembler a{g_buf, 16};
        a.MovImm(Sz::X, 1, v);
        for (size_t i = 0; i < a.Size(); ++i) {
            std::printf("@movimm64 %llx|%08x\n", static_cast<unsigned long long>(v), g_buf[i]);
        }
    }
    for (uint64 v : {0ull, 1ull, 0xFFFFull, 0x10000ull, 0xDEADBEEFull, 0xFFFFFFFFull, 0x80000000ull, 0xFFFF0000ull,
                     0x55555555ull, 0xFFFF1234ull}) {
        Assembler a{g_buf, 16};
        a.MovImm(Sz::W, 1, v);
        for (size_t i = 0; i < a.Size(); ++i) {
            std::printf("@movimm32 %llx|%08x\n", static_cast<unsigned long long>(v), g_buf[i]);
        }
    }
    return 0;
}
