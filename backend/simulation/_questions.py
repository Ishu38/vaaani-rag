"""JEE question pools with difficulty ratings (1-5).

Difficulty scale:
  1 = basic recall / formula plug-in
  2 = single concept, straightforward
  3 = multi-step, moderate
  4 = multi-concept synthesis
  5 = tricky, requires deep insight
"""

POOL_PHYSICS = [
    {"query": "A ball is thrown vertically upward with speed 20 m/s. How high does it go? (g=10 m/s²)", "answer": "20 m", "topic": "kinematics", "difficulty": 1.0, "hint": "Use v² = u² - 2gh with v=0 at max height."},
    {"query": "A car accelerates uniformly from rest to 20 m/s in 10 s. Find the distance covered.", "answer": "100 m", "topic": "kinematics", "difficulty": 1.0, "hint": "Use s = (u+v)t/2."},
    {"query": "A 2 kg block is on a frictionless 30° incline. Find its acceleration. (g=10 m/s²)", "answer": "5 m/s²", "topic": "newtons_laws", "difficulty": 1.5, "hint": "a = g sin θ"},
    {"query": "Find the work done by a force F = 10 N moving a body 5 m in the direction of the force.", "answer": "50 J", "topic": "work_energy", "difficulty": 1.0, "hint": "W = Fd cos θ, with θ=0."},
    {"query": "A 1000 kg car at 10 m/s has what kinetic energy?", "answer": "50000 J", "topic": "work_energy", "difficulty": 1.0, "hint": "KE = ½mv²"},
    {"query": "A stone is dropped from a cliff 80 m high. Find time to hit ground. (g=10 m/s²)", "answer": "4 s", "topic": "kinematics", "difficulty": 1.5, "hint": "h = ½gt²"},
    {"query": "A projectile is launched at 45° with v₀ = 28.28 m/s. Find its range. (g=10 m/s²)", "answer": "80 m", "topic": "projectile_motion", "difficulty": 2.0, "hint": "R = v₀² sin(2θ) / g"},
    {"query": "An object of mass 5 kg experiences a net force of 20 N. Find its acceleration.", "answer": "4 m/s²", "topic": "newtons_laws", "difficulty": 1.0, "hint": "F = ma"},
    {"query": "A 50 kg person stands on a scale in an elevator accelerating upward at 2 m/s². What does the scale read? (g=10 m/s²)", "answer": "600 N", "topic": "newtons_laws", "difficulty": 2.5, "hint": "N = m(g+a)"},
    {"query": "A block of mass m is pushed against a wall with horizontal force F. Coefficient of static friction is μ. What minimum F prevents it from sliding? (g=10)", "answer": "mg/μ", "topic": "friction", "difficulty": 3.0, "hint": "Friction f=μN where N=F. Equilibrium: f=mg."},
    {"query": "A simple pendulum of length 1 m has what period? (Use g=π² m/s²)", "answer": "2 s", "topic": "oscillations", "difficulty": 1.5, "hint": "T = 2π√(L/g)"},
    {"query": "A body of mass 2 kg moving at 4 m/s collides elastically with a stationary 2 kg body. Find velocities after collision.", "answer": "0 m/s and 4 m/s", "topic": "momentum", "difficulty": 2.5, "hint": "For equal masses in elastic collision, velocities swap."},
    {"query": "A bullet of mass 10 g moving at 400 m/s embeds in a 2 kg block at rest on frictionless surface. Find their common velocity.", "answer": "1.99 m/s", "topic": "momentum", "difficulty": 2.0, "hint": "Conservation of momentum: m₁v₁ = (m₁+m₂)V"},
    {"query": "An ideal gas at constant temperature expands to double its volume. The pressure becomes:", "answer": "half", "topic": "thermodynamics", "difficulty": 1.5, "hint": "PV = constant at constant T."},
    {"query": "Two resistors 4Ω and 6Ω are in parallel. The equivalent resistance is:", "answer": "2.4 Ω", "topic": "current_electricity", "difficulty": 1.5, "hint": "1/Req = 1/4 + 1/6"},
    {"query": "A 12V battery with internal resistance 2Ω is connected to a 10Ω external resistor. Find terminal voltage.", "answer": "10 V", "topic": "current_electricity", "difficulty": 2.0, "hint": "V_terminal = E - Ir where I = E/(R+r)"},
    {"query": "If the charge on a capacitor doubles, the stored energy becomes:", "answer": "4 times", "topic": "electrostatics", "difficulty": 1.5, "hint": "U = Q²/(2C)"},
    {"query": "Two point charges +4 μC and -2 μC are 30 cm apart. Where on the line joining them is the electric field zero?", "answer": "51 cm from the +4 μC charge (outside)", "topic": "electrostatics", "difficulty": 4.0, "hint": "Set field magnitudes equal: k|q₁|/x² = k|q₂|/(d-x)². Zero field lies outside, closer to smaller charge."},
    {"query": "A wire of resistance R is stretched to triple its length. New resistance is:", "answer": "9R", "topic": "current_electricity", "difficulty": 2.5, "hint": "R ∝ L/A and volume constant → A ∝ 1/L. So R ∝ L²."},
    {"query": "A transformer has 1000 primary turns and 100 secondary turns. If primary voltage is 220V, secondary voltage is:", "answer": "22 V", "topic": "electromagnetism", "difficulty": 1.5, "hint": "Vs/Vp = Ns/Np"},
    {"query": "Light of wavelength 600 nm produces first minimum at angle 30° in single slit diffraction. Slit width is:", "answer": "1200 nm", "topic": "optics", "difficulty": 2.0, "hint": "a sin θ = λ"},
    {"query": "A converging lens has focal length 20 cm. Object at 40 cm produces image at:", "answer": "40 cm (real, inverted, same size)", "topic": "optics", "difficulty": 2.0, "hint": "1/f = 1/v + 1/u"},
    {"query": "A satellite orbits at height h = R above Earth's surface. Its orbital speed is (√_______ times surface escape velocity):", "answer": "1/2", "topic": "gravitation", "difficulty": 3.0, "hint": "v_orb = √(GM/2R), v_esc = √(2GM/R)"},
    {"query": "A body of mass 2 kg is released from a height of 40 m. Just before impact its kinetic energy is: (g=10)", "answer": "800 J", "topic": "work_energy", "difficulty": 1.5, "hint": "KE = mgh = PE converted"},
    {"query": "An alpha particle (charge +2e) enters uniform magnetic field B perpendicularly. Compared to a proton entering at same speed, radius ratio r_α/r_p is:", "answer": "2", "topic": "magnetic_effects", "difficulty": 3.0, "hint": "r = mv/qB. Alpha has 4× mass, 2× charge → r_α/r_p = (4m·v/2eB)/(m·v/eB) = 2"},
]

POOL_MATH = [
    {"query": "Evaluate: ∫ x² dx", "answer": "x³/3 + C", "topic": "integration", "difficulty": 1.0, "hint": "Power rule: ∫ x^n dx = x^(n+1)/(n+1) + C"},
    {"query": "Find d/dx (sin 2x)", "answer": "2 cos 2x", "topic": "derivatives", "difficulty": 1.5, "hint": "Chain rule."},
    {"query": "Find the limit: lim(x→0) (sin x)/x", "answer": "1", "topic": "limits", "difficulty": 1.0, "hint": "Standard limit."},
    {"query": "Solve for x: 2^x = 32", "answer": "5", "topic": "algebra", "difficulty": 1.0, "hint": "Write 32 as 2^5."},
    {"query": "If A = [[1,2],[3,4]], find det(A).", "answer": "-2", "topic": "matrices", "difficulty": 1.0, "hint": "det = ad - bc = 1×4 - 2×3"},
    {"query": "What is the equation of a circle with center (2,-3) and radius 5?", "answer": "(x-2)²+(y+3)²=25", "topic": "coordinate_geometry", "difficulty": 1.5, "hint": "(x-h)²+(y-k)²=r²"},
    {"query": "Find the value: C(10,3) = ?", "answer": "120", "topic": "combinatorics", "difficulty": 1.0, "hint": "C(n,r) = n!/(r!(n-r)!)"},
    {"query": "Solve: |2x - 3| = 7", "answer": "x = 5 or x = -2", "topic": "algebra", "difficulty": 1.5, "hint": "2x-3 = 7 or 2x-3 = -7"},
    {"query": "Evaluate: ∫₀^1 x dx", "answer": "1/2", "topic": "integration", "difficulty": 1.0, "hint": "∫x dx = x²/2, evaluate from 0 to 1"},
    {"query": "Find dy/dx if y = x² e^x", "answer": "e^x(x²+2x)", "topic": "derivatives", "difficulty": 2.0, "hint": "Product rule: u'v + uv'"},
    {"query": "Evaluate: ∫ x e^x dx", "answer": "e^x(x-1) + C", "topic": "integration", "difficulty": 2.0, "hint": "Integration by parts. u=x, dv=e^x dx."},
    {"query": "Find the equation of the tangent to y = x² at x = 2.", "answer": "y = 4x - 4", "topic": "derivatives", "difficulty": 2.0, "hint": "Slope = dy/dx = 2x at x=2 is 4. Point is (2,4). Line: y-4 = 4(x-2)."},
    {"query": "Solve the differential equation: dy/dx = y", "answer": "y = Ce^x", "topic": "differential_equations", "difficulty": 1.5, "hint": "Separation of variables: dy/y = dx"},
    {"query": "The area bounded by y = x², x = 0, x = 2, and the x-axis is:", "answer": "8/3", "topic": "integration", "difficulty": 2.0, "hint": "∫₀² x² dx = [x³/3]₀² = 8/3"},
    {"query": "A bag has 3 red and 5 blue balls. Two are drawn without replacement. P(both red) = ?", "answer": "3/28", "topic": "probability", "difficulty": 2.0, "hint": "(3/8)×(2/7) = 6/56 = 3/28"},
    {"query": "Find the inverse of the matrix [[2,1],[5,3]]", "answer": "[[3,-1],[-5,2]]", "topic": "matrices", "difficulty": 2.5, "hint": "A⁻¹ = (1/det)[d -b; -c a]. det=6-5=1."},
    {"query": "Solve: log₂(x) + log₂(x-2) = 3", "answer": "4", "topic": "algebra", "difficulty": 2.5, "hint": "log₂(x(x-2)) = 3 → x(x-2) = 8 → x²-2x-8=0 → x=4."},
    {"query": "If z = (1+i)/(1-i), find |z|.", "answer": "1", "topic": "complex_numbers", "difficulty": 2.0, "hint": "Multiply numerator and denominator by (1+i). z = i."},
    {"query": "Evaluate: lim(x→∞) (1 + 1/x)^x", "answer": "e", "topic": "limits", "difficulty": 1.5, "hint": "Euler's definition of e."},
    {"query": "The sum of the series 1 + 1/2 + 1/4 + 1/8 + ... to infinity is:", "answer": "2", "topic": "sequences", "difficulty": 1.0, "hint": "Infinite GP with a=1, r=1/2: S = a/(1-r) = 2"},
    {"query": "Solve: sin 2x = sin x for 0 ≤ x < 2π", "answer": "0, π/3, π, 5π/3", "topic": "trigonometry", "difficulty": 3.0, "hint": "2sin x cos x = sin x → sin x(2cos x - 1) = 0"},
    {"query": "Find the maximum value of f(x) = sin x + cos x.", "answer": "√2", "topic": "trigonometry", "difficulty": 3.0, "hint": "sin x + cos x = √2 sin(x + π/4)"},
    {"query": "The function f(x) = x³ - 3x² + 3x - 1 has how many real roots?", "answer": "1", "topic": "calculus", "difficulty": 2.5, "hint": "f(x) = (x-1)³. Triple root at x=1."},
    {"query": "Two dice are rolled. Find P(sum = 7).", "answer": "1/6", "topic": "probability", "difficulty": 1.5, "hint": "Favorable: (1,6),(2,5),(3,4),(4,3),(5,2),(6,1). Total 36 outcomes."},
    {"query": "Find the area of the triangle with vertices (1,2), (4,6), (7,2).", "answer": "12", "topic": "coordinate_geometry", "difficulty": 2.0, "hint": "Area = ½|x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|"},
]

POOL_PHYSICS.sort(key=lambda q: q["difficulty"])
POOL_MATH.sort(key=lambda q: q["difficulty"])
