# FACTS — FIX CHAIN #2 "How do 2 primes guard your bank account?" (24 Nov 2026)
Every on-screen claim + source. Unverified items were dropped.
- Caesar cipher: Julius Caesar shifted letters by 3 (Suetonius, Life of Julius Caesar 56; used in the Gallic Wars era, 58-50 BC -> '~50 BC').
- 15 = 3 × 5: arithmetic (smallest RSA-style semiprime of two odd primes; Shor's first lab demo factored 15, Vandersypen et al., Nature 2001).
- Hook wording: VO says primes 'help guard' (RSA still signs many web certificates; key exchange now often uses elliptic curves). Toy text ATTACK AT DAWN -> DWWDFN DW GDZQ computed in render.py.
- Frequency analysis: first written description by al-Kindi, 9th century (Manuscript on Deciphering Cryptographic Messages). E is the most common English letter (~12.7%, Lewand 2000 table used for bars; ciphertext bars = English shifted by 3, so H peaks).
- Key distribution problem: symmetric ciphers need both sides to share the key first (Diffie & Hellman, "New Directions in Cryptography", 1976).
- RSA: Rivest, Shamir, Adleman, 1977 (MIT; CACM 1978). Public key = n = p×q; toy example 61 × 53 = 3233 (standard textbook example).
- RSA-250 (250 digits, 829 bits) factored Feb 2020 by Boudot, Gaudry, Guillevic, Heninger, Thomé, Zimmermann: ~2,700 core-years (Intel Xeon Gold 6130 2.1 GHz reference).
- RSA-2048 = 617 decimal digits; RSA Factoring Challenge number never factored.
- Shor's algorithm, Peter Shor, 1994: factors integers in polynomial time on a quantum computer.
- Gidney (Google Quantum AI), arXiv:2505.15917 (May 2025), "How to factor 2048 bit RSA integers with less than a million noisy qubits": < 1 million noisy qubits, < 1 week. Estimate only; no such machine exists.
- Superconducting quantum computers run in dilution refrigerators at ~10 mK (the "chandelier").
- NIST FIPS 203 ML-KEM (Module-Lattice-Based Key-Encapsulation Mechanism), finalised 13 Aug 2024 (nist.gov news 2024/08, csrc.nist.gov/pubs/fips/203/final). Security rests on lattice problems (Module-LWE), related to finding close lattice points; no known efficient quantum attack.
- Wording guard: VO says a quantum computer "could" crack RSA; on-screen "estimate … no such machine yet".
- v2 (posted) on-screen extras: "Rivest · Shamir · Adleman" (RSA 1977); "61 x 53 = 3233" (arithmetic); guesses counter is an illustration of trial division growing; the 617 digits shown are random digits labelled "illustration · RSA-2048"; "UNDER 1 WEEK* — *estimate, Google 2025" (Gidney arXiv:2505.15917); "NEW LOCKS (2024) · no known shortcut" (NIST FIPS 203 ML-KEM).
