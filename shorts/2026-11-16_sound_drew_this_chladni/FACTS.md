# FACTS — SOURCE CODE #11 Chladni plate (16 Nov 2026)
Every on-screen number comes from our own solver (plate.py) or the sources below.

- Model: Kirchhoff thin-plate free vibration, D∇⁴w = ρh ω² w, D = E h³ / 12(1−ν²). Free edges on all four sides, centre-driven (centre bolt = driver).
  Kirchhoff, G. (1850) "Über das Gleichgewicht und die Bewegung einer elastischen Scheibe", J. reine angew. Math. 40, 51–88.
- Solver: Ritz method, orthonormal Legendre basis (27×27 = 729 functions), Gauss quadrature, generalised eigenproblem (scipy eigh). Free edges are natural boundary conditions in Ritz, so no fudge.
- Check vs the reference table: first non-rigid modes ω a² √(ρh/D) = 13.47, 19.60, 24.27, 34.80, 34.80, 61.09 (ν = 0.3). Leissa, A. W. (1969) Vibration of Plates, NASA SP-160, chapter 4 (completely free square plate). Our output: 13.47 19.60 24.27 34.80 34.80 61.09. MATCH.
- Plate: steel E = 200 GPa, ρ = 7850 kg/m³, ν = 0.3, 24 cm × 24 cm, 1 mm thick. Centre-driven resonances (Hz): 102, 269, 517, 711, 1186, 1266, 1455, 1973, 2346, 2529, 2774, 3058, 3758, 3888, 3950 ... (shown: 102, 269, 517, 711, 1186, 1455, 1973, 2529, 3058, 3950).
- Twist: ω ∝ √(D/ρh) ∝ h, so doubling thickness doubles every resonance: 1973 Hz shape needs 3946 Hz at 2 mm. Exact within the Kirchhoff model.
- Sand: grains hop with a size set by the local vibration amplitude of the driven response (sum of real modes, Q = 28) and come to rest where the plate does not move (nodal lines). Qualitative grain model; the shapes come from the solver.
- Chladni first showed these figures: E. F. F. Chladni (1787) Entdeckungen über die Theorie des Klanges, Leipzig.
- Violin makers: free violin plates are tuned by watching Chladni patterns (tap-tone/plate tuning). Hutchins, C. M. (1981) "The Acoustics of Violin Plates", Scientific American 245(4), 170–186.
- Tease (next): acoustic levitation of water droplets is real (standing-wave levitators hold water drops in air).

