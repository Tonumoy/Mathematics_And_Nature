# FACTS — SOURCE CODE #16 "Why are Saturn's rings so flat?" (checked 26 Nov 2026)
| On screen / VO | Value | Source |
|---|---|---|
| Main rings ~10 m thick | "about 30 feet (10 meters) thick or so" | NASA Cassini rings page https://science.nasa.gov/mission/cassini/science/rings/ ; JPL "Those Amazing Rings" says "less than a kilometer" (max, older) |
| ~270,000 km across | A ring outer edge 136,775 km radius -> 273,550 km across; JPL: "about 170,000 miles in diameter" (= ~274,000 km) | JPL minibook; NASA Saturn fact sheet ring radii |
| wider than 20 Earths | 273,550 / 12,742 km = 21.5 | arithmetic |
| 100 m wide -> 0.004 mm thick | 100 m x 10 m / 273,550 km = 3.7 µm; paper ~0.1 mm | arithmetic |
| ice chunks, sand grain -> house/mountain size | "chunks of water ice, ranging in size from smaller than a grain of sand to the size of a mountain" (NASA); "as big as a house" (JPL) | NASA, JPL |
| ring radii in the render | C 74,658-92,000; B 92,000-117,580; Cassini div. to 122,170; A 122,170-136,775; Encke gap 133,589 km; Saturn eq. radius 60,268 km, polar/eq 0.902 | NASA Saturn fact sheet (nssdc.gsfc.nasa.gov) |
| why flat | inelastic collisions dissipate random (up-down) motion; total angular momentum conserved -> thin disk | Brahic 1977 A&A 54, 895; Goldreich & Tremaine 1978 Icarus 34, 227; Bridges, Hatzes & Lin 1984 Nature 309, 333 (ice is not perfectly bouncy) |
| galaxies / young solar systems flatten the same way | dissipation + conserved spin -> disks (gas) | standard (e.g. Mo, van den Bosch & White 2010, ch. 11) |
| model | sim.py: 6,000 particles, point-mass Kepler gravity, DSMC-style pair bumps in 3D cells, v_n' = -e v_n; e = 0.1 -> thickness falls ~83%; e = 1 -> thickness grows | own code; e = 0.1 is a model value, not a measured ice number |
Not claimed: ring age, real e of ring ice, exact current thickness everywhere.
