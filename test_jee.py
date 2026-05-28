#!/usr/bin/env python3
"""JEE problem test harness for Vaaani.

Tests Socratic mode and Direct+CoT mode against 20 JEE problems
(10 Physics, 10 Mathematics) using the real llm.py prompt builder
and DeepSeek API.
"""
import sys, json, time, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from llm import build_prompt, call_deepseek, LLMResponse
from intent import classify

DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"

# ── Mock chunks with JEE context ──────────────────────────────────────────

PHYSICS_CHUNKS = [
    {"text": "Projectile range: R = v0² sin(2θ) / g. Maximum height: H = v0² sin²θ / (2g). Time of flight: T = 2v0 sin θ / g. For a launch angle of 30°, sin 60° = √3/2 ≈ 0.866. With g = 9.8 m/s².", "source": "jee-physics-kinematics.md"},
    {"text": "Equations of motion: v = u + at, s = ut + (1/2)at², v² = u² + 2as. For free fall from rest: v = gt, h = (1/2)gt², v = √(2gh).", "source": "jee-physics-kinematics.md"},
    {"text": "Newton's Second Law: F = ma. For an incline of angle θ: acceleration down frictionless incline = g sin θ. Normal force = mg cos θ. Friction force f = μN.", "source": "jee-physics-kinematics.md"},
    {"text": "Work-Energy theorem: W_net = ΔKE = (1/2)mv² - (1/2)mu². PE = mgh. For conservative forces: KE₁ + PE₁ = KE₂ + PE₂. Power P = F·v.", "source": "jee-physics-kinematics.md"},
    {"text": "Coulomb's law: F = k q₁q₂/r², k = 9×10⁹ N·m²/C². Electric field E = kQ/r². Potential V = kQ/r. Capacitance C = Q/V. Energy in capacitor U = (1/2)CV².", "source": "jee-physics-advanced.md"},
    {"text": "Ohm's law: V = IR. Power P = I²R = V²/R. Kirchhoff's laws: KCL (sum of currents at junction = 0) and KVL (sum of voltages around loop = 0). Series resistance: Req = R₁ + R₂. Parallel: 1/Req = 1/R₁ + 1/R₂.", "source": "jee-physics-advanced.md"},
    {"text": "Thermodynamics first law: ΔU = Q - W. Ideal gas: PV = nRT. Adiabatic process: PV^γ = constant where γ = Cp/Cv = 5/3 for monatomic, 7/5 for diatomic. Carnot efficiency: η = 1 - Tc/Th.", "source": "jee-physics-advanced.md"},
    {"text": "Centripetal acceleration a_c = v²/r = ω²r. Centripetal force F_c = mv²/r. For a banked curve: v_max = √(rg tan θ). Torque τ = rF sin θ. Moment of inertia I = Σmr². Angular momentum L = Iω.", "source": "jee-physics-kinematics.md"},
]

MATH_CHUNKS = [
    {"text": "Derivative rules: d/dx(x^n) = nx^(n-1). Product rule: d/dx(uv) = u'v + uv'. Chain rule: d/dx f(g(x)) = f'(g(x))g'(x). d/dx(sin x) = cos x. d/dx(cos x) = -sin x. d/dx(e^x) = e^x. d/dx(ln x) = 1/x.", "source": "jee-math-calculus.md"},
    {"text": "Integration by parts: ∫u dv = uv - ∫v du. Choose u using LIATE: Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential. Example: ∫x sin x dx: u=x (A), dv=sin x dx. Then du=dx, v=-cos x. Result: -x cos x + sin x + C.", "source": "jee-math-calculus.md"},
    {"text": "∫x^n dx = x^(n+1)/(n+1) + C (n≠-1). ∫sin x dx = -cos x + C. ∫cos x dx = sin x + C. ∫e^x dx = e^x + C. ∫sec²x dx = tan x + C. ∫(1/x) dx = ln|x| + C.", "source": "jee-math-calculus.md"},
    {"text": "Definite integrals: ∫_a^b f(x)dx = F(b) - F(a). Properties: ∫_0^a f(x)dx = ∫_0^a f(a-x)dx. For even functions: ∫_{-a}^a f(x)dx = 2∫_0^a f(x)dx. For odd: integral = 0.", "source": "jee-math-calculus.md"},
    {"text": "Determinant of 2×2 matrix: |a b; c d| = ad - bc. Inverse: A⁻¹ = (1/det)[d -b; -c a]. For 3×3: det = a(ei-fh) - b(di-fg) + c(dh-eg). Eigenvalues: solutions of |A - λI| = 0. Trace = sum of diagonal elements.", "source": "jee-math-calculus.md"},
    {"text": "Probability formulas: P(A∪B) = P(A)+P(B)-P(A∩B). P(A|B) = P(A∩B)/P(B). Bayes: P(A|B) = P(B|A)P(A)/P(B). Binomial: P(X=k) = C(n,k)p^k(1-p)^(n-k). Mean μ=np, variance σ²=np(1-p).", "source": "jee-math-calculus.md"},
]

# ── JEE Problem Set ───────────────────────────────────────────────────────

JEE_PROBLEMS = [
    # ── PHYSICS (1-10) ──
    {
        "id": "P1",
        "subject": "Physics",
        "topic": "Projectile Motion",
        "query": "A ball is thrown at an angle of 30° with an initial speed of 20 m/s. Find the horizontal range. Take g = 10 m/s².",
        "chunks": PHYSICS_CHUNKS,
        "expected": "approx 34.6 m",
    },
    {
        "id": "P2",
        "subject": "Physics",
        "topic": "Kinematics",
        "query": "A ball is thrown horizontally from the top of a tower 45 m high with a speed of 15 m/s. Find the time taken to hit the ground and the horizontal distance covered. Take g = 10 m/s².",
        "chunks": PHYSICS_CHUNKS,
        "expected": "3 seconds, 45 m",
    },
    {
        "id": "P3",
        "subject": "Physics",
        "topic": "Newton's Laws",
        "query": "A block of mass 5 kg is placed on a frictionless incline of angle 30°. Find the acceleration of the block down the incline.",
        "chunks": PHYSICS_CHUNKS,
        "expected": "5 m/s²",
    },
    {
        "id": "P4",
        "subject": "Physics",
        "topic": "Work-Energy",
        "query": "A 2 kg ball is dropped from a height of 20 m. Using energy conservation, find its speed just before hitting the ground. Take g = 10 m/s².",
        "chunks": PHYSICS_CHUNKS,
        "expected": "20 m/s",
    },
    {
        "id": "P5",
        "subject": "Physics",
        "topic": "Circular Motion",
        "query": "A car of mass 1000 kg moves around a circular track of radius 50 m at a speed of 10 m/s. Find the centripetal force acting on the car.",
        "chunks": PHYSICS_CHUNKS,
        "expected": "2000 N",
    },
    {
        "id": "P6",
        "subject": "Physics",
        "topic": "Electrostatics",
        "query": "Two point charges of +2 μC and +3 μC are placed 10 cm apart in vacuum. Find the force between them. Take k = 9×10⁹ N·m²/C².",
        "chunks": PHYSICS_CHUNKS,
        "expected": "5.4 N",
    },
    {
        "id": "P7",
        "subject": "Physics",
        "topic": "Current Electricity",
        "query": "A 100 W, 220 V bulb is connected to a 220 V supply. Find the current flowing through the bulb and its resistance.",
        "chunks": PHYSICS_CHUNKS,
        "expected": "0.454 A, 484 Ω",
    },
    {
        "id": "P8",
        "subject": "Physics",
        "topic": "Thermodynamics",
        "query": "A Carnot engine operates between temperatures 500 K and 300 K. Calculate its efficiency. If it absorbs 1000 J of heat from the hot reservoir, how much work does it do?",
        "chunks": PHYSICS_CHUNKS,
        "expected": "40%, 400 J",
    },
    {
        "id": "P9",
        "subject": "Physics",
        "topic": "Capacitors",
        "query": "A 10 μF capacitor is charged to a potential difference of 100 V. Find the energy stored in the capacitor.",
        "chunks": PHYSICS_CHUNKS,
        "expected": "0.05 J or 50 mJ",
    },
    {
        "id": "P10",
        "subject": "Physics",
        "topic": "Rotational Motion",
        "query": "A solid sphere of mass 2 kg and radius 0.1 m rolls without slipping down an incline. Its moment of inertia about its center is (2/5)MR². If its center of mass moves at 5 m/s at the bottom, what is its total kinetic energy?",
        "chunks": PHYSICS_CHUNKS,
        "expected": "35 J",
    },
    # ── MATH (11-20) ──
    {
        "id": "M1",
        "subject": "Math",
        "topic": "Integration by Parts",
        "query": "Evaluate the integral: ∫ x·sin x dx",
        "chunks": MATH_CHUNKS,
        "expected": "-x cos x + sin x + C",
    },
    {
        "id": "M2",
        "subject": "Math",
        "topic": "Integration by Parts",
        "query": "Evaluate the integral: ∫ x·e^x dx",
        "chunks": MATH_CHUNKS,
        "expected": "e^x(x-1) + C",
    },
    {
        "id": "M3",
        "subject": "Math",
        "topic": "Derivatives",
        "query": "Find the derivative of f(x) = x³ sin x with respect to x.",
        "chunks": MATH_CHUNKS,
        "expected": "3x² sin x + x³ cos x",
    },
    {
        "id": "M4",
        "subject": "Math",
        "topic": "Integration",
        "query": "Evaluate: ∫ cos x dx from x = 0 to x = π/2",
        "chunks": MATH_CHUNKS,
        "expected": "1",
    },
    {
        "id": "M5",
        "subject": "Math",
        "topic": "Chain Rule",
        "query": "If y = ln(sin x), find dy/dx.",
        "chunks": MATH_CHUNKS,
        "expected": "cot x",
    },
    {
        "id": "M6",
        "subject": "Math",
        "topic": "Definite Integrals",
        "query": "Evaluate the definite integral: ∫₀^π sin x dx",
        "chunks": MATH_CHUNKS,
        "expected": "2",
    },
    {
        "id": "M7",
        "subject": "Math",
        "topic": "Derivatives",
        "query": "Find the derivative of f(x) = e^x cos x with respect to x.",
        "chunks": MATH_CHUNKS,
        "expected": "e^x(cos x - sin x)",
    },
    {
        "id": "M8",
        "subject": "Math",
        "topic": "Integration",
        "query": "Evaluate: ∫ (3x² + 2x + 1) dx",
        "chunks": MATH_CHUNKS,
        "expected": "x³ + x² + x + C",
    },
    {
        "id": "M9",
        "subject": "Math",
        "topic": "Integration by Parts",
        "query": "Evaluate the integral: ∫ ln x dx",
        "chunks": MATH_CHUNKS,
        "expected": "x(ln x - 1) + C",
    },
    {
        "id": "M10",
        "subject": "Math",
        "topic": "Derivatives",
        "query": "Find dy/dx if y = tan⁻¹(x).",
        "chunks": MATH_CHUNKS,
        "expected": "1/(1+x²)",
    },
]


def run_test(problem, mode="direct"):
    """Run a single JEE problem through the prompt+LLM pipeline."""
    socratic = mode == "socratic"
    query = problem["query"]
    chunks = problem["chunks"]
    subject = problem["subject"]

    intent = classify(query)
    messages = build_prompt(
        query, chunks,
        memory_block="",
        intent=intent,
        structured=False,
        socratic=socratic,
    )

    try:
        resp = call_deepseek(messages, stream=False, json_mode=False)
    except Exception as e:
        return {"error": str(e), "answer": "", "tokens": 0}

    choice = resp.get("choices", [{}])[0]
    answer = choice.get("message", {}).get("content", "")
    tokens = resp.get("usage", {}).get("total_tokens", 0)

    return {"answer": answer, "tokens": tokens, "error": None}


def grade_answer(expected, actual, problem_id):
    """Simple keyword-based grading. Checks if the expected key result appears."""
    expected_lower = expected.lower().replace(" ", "")
    actual_lower = actual.lower().replace(" ", "")

    # Extract key numbers/expressions from expected
    import re
    key_terms = re.findall(r'[\d.]+|[-+*/^()]|sin|cos|tan|ln|sqrt|cot|csc|sec', expected_lower)
    key_terms = [t for t in key_terms if len(t) > 1 or t.isdigit()]

    if not key_terms:
        return True  # Can't grade, assume OK

    matches = sum(1 for t in key_terms if t in actual_lower)
    return matches >= max(1, len(key_terms) * 0.5)


def print_separator(char="─", width=70):
    print(DIM + char * width + RESET)


def main():
    print(BOLD + CYAN + "\n  VAAANI JEE PROBLEM TEST SUITE" + RESET)
    print(f"  {len([p for p in JEE_PROBLEMS if p['subject']=='Physics'])} Physics + "
          f"{len([p for p in JEE_PROBLEMS if p['subject']=='Math'])} Math problems\n")

    results = []

    for problem in JEE_PROBLEMS:
        pid = problem["id"]
        subject = problem["subject"]
        topic = problem["topic"]

        print_separator()
        print(f"  {BOLD}{pid} | {subject} | {topic}{RESET}")
        print(f"  {YELLOW}Q: {problem['query'][:120]}{'...' if len(problem['query'])>120 else ''}{RESET}")
        print(f"  {DIM}Expected: {problem['expected']}{RESET}\n")

        # ── Direct + CoT mode ──
        print(f"  {CYAN}[Direct + CoT]{RESET}")
        start = time.time()
        direct = run_test(problem, mode="direct")
        elapsed_d = time.time() - start

        if direct["error"]:
            print(f"    {RED}ERROR: {direct['error'][:200]}{RESET}")
        else:
            d_ok = grade_answer(problem["expected"], direct["answer"], pid)
            status = f"{GREEN}PASS{RESET}" if d_ok else f"{YELLOW}REVIEW{RESET}"
            print(f"    {status}  ({direct['tokens']} tokens, {elapsed_d:.1f}s)")
            # Show first 500 chars of answer
            ans_preview = direct["answer"][:500].replace("\n", "\n    ")
            print(f"    {DIM}{ans_preview}{RESET}")
            if len(direct["answer"]) > 500:
                print(f"    {DIM}... (truncated, total {len(direct['answer'])} chars){RESET}")

            results.append({
                "id": pid,
                "subject": subject,
                "topic": topic,
                "mode": "direct",
                "pass": d_ok,
                "tokens": direct["tokens"],
                "time_s": round(elapsed_d, 1),
                "error": None,
            })

        print()

        # ── Socratic mode ──
        print(f"  {CYAN}[Socratic]{RESET}")
        start = time.time()
        socratic = run_test(problem, mode="socratic")
        elapsed_s = time.time() - start

        if socratic["error"]:
            print(f"    {RED}ERROR: {socratic['error'][:200]}{RESET}")
        else:
            s_ok = grade_answer(problem["expected"], socratic["answer"], pid)
            # Socratic mode should NOT give the direct answer — it should ask questions
            # So "passing" here means it's asking questions, not giving the answer
            is_asking = "?" in socratic["answer"] or any(
                w in socratic["answer"].lower() for w in ["what do you", "can you", "how would", "which principle"]
            )
            status = f"{GREEN}SOCRATIC{RESET}" if is_asking else f"{YELLOW}DIRECT{RESET}"
            print(f"    {status}  ({socratic['tokens']} tokens, {elapsed_s:.1f}s)")
            ans_preview = socratic["answer"][:500].replace("\n", "\n    ")
            print(f"    {DIM}{ans_preview}{RESET}")
            if len(socratic["answer"]) > 500:
                print(f"    {DIM}... (truncated){RESET}")

            results.append({
                "id": pid,
                "subject": subject,
                "topic": topic,
                "mode": "socratic",
                "pass": is_asking,
                "tokens": socratic["tokens"],
                "time_s": round(elapsed_s, 1),
                "error": None,
            })

    # ── Summary ──
    print_separator("═")
    total = len(results)
    direct_results = [r for r in results if r["mode"] == "direct"]
    socratic_results = [r for r in results if r["mode"] == "socratic"]
    direct_pass = sum(1 for r in direct_results if r["pass"])
    socratic_pass = sum(1 for r in socratic_results if r["pass"])
    tokens_total = sum(r["tokens"] for r in results)

    print(f"\n  {BOLD}SUMMARY{RESET}\n")
    print(f"  Direct + CoT:  {GREEN}{direct_pass}{RESET}/{len(direct_results)} passed")
    print(f"  Socratic:      {GREEN}{socratic_pass}{RESET}/{len(socratic_results)} in questioning mode")
    print(f"  Total tokens:  {tokens_total}")
    print(f"  Accuracy:      direct={direct_pass/len(direct_results)*100:.0f}%  "
          f"socratic_ratio={socratic_pass/len(socratic_results)*100:.0f}%")
    print()

    # ── Per-subject breakdown ──
    for subj in ["Physics", "Math"]:
        subj_direct = [r for r in direct_results if r["subject"] == subj]
        subj_pass = sum(1 for r in subj_direct if r["pass"])
        print(f"  {subj}: {GREEN}{subj_pass}{RESET}/{len(subj_direct)} direct | "
              f"avg {sum(r['tokens'] for r in subj_direct)//max(1,len(subj_direct))} tokens/problem")

    print()
    return direct_pass == len(direct_results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
