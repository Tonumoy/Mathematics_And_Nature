# SC#6 SEO (K4, final)
TITLE: Why Is the Sky Blue, Not Violet? 🌌 Real Physics in Code #shorts

DESCRIPTION:
Why is the sky blue and not violet? Violet light scatters even MORE than blue in air, so the sky "should" be violet. Here's the real reason, explained simply and solved in code.

What you see
• A real blue sky, then a simulation of sunlight hitting air molecules (short waves get flung sideways much more)
• Blue scatters about 6x more than red, so blue reaches you from every part of the sky
• Why not violet: the Sun gives off less violet, and your eyes are bad at seeing it, so the mix looks sky blue
• Change one number (the power 4 → 0) and the computed sky turns white
• A real sunset on Half Dome, and a real blue sunset on Mars (NASA Curiosity)

The code
• Rayleigh scattering: I ∝ 1/λ⁴. Blue (450 nm) scatters 5.9x more than red (700 nm); violet (400 nm) 9.4x
• Sky colour = single-scattering radiative transfer: β(λ) = (τ₅₅₀/H)·(550/λ)⁴ with τ₅₅₀ ≈ 0.097, H = 8 km, phase 3/16π(1+cos²θ) → CIE 1931 colour → sRGB
• At sunset light crosses ~38x more air (air mass ≈ 38), so most of the blue is scattered away first
• Clouds look white for a similar reason: their droplets are much bigger than light's wavelength, so all colours scatter about equally

❓ What colour is the sky on the Moon? Comment your guess 👇

Sources: Bodhaine et al. 1999, J. Atmos. Oceanic Technol. 16:1854 (Rayleigh optical depth) · Kasten & Young 1989 (air mass) · NASA/JPL-Caltech/MSSS PIA19400 (Curiosity sunset, Gale crater, 15 Apr 2015) · CIE 1931 colour matching
Image credits: "Atmospheric convection over New Braunfels" by Brainandforce, CC BY 4.0 · "Sunset on Halfdome timelapse Yosemite" by G. Edward Johnson, CC BY 4.0 (both Wikimedia Commons) · Mars image NASA/JPL-Caltech/MSSS
Code: https://github.com/Tonumoy/Mathematics_And_Nature/tree/main/shorts
Music generated from code (no samples). Narration: AI voice.
Series: SOURCE CODE #6 · Coded Cosmos shows the real code running underneath everyday things. Subscribe for the next one.

#whyistheskyblue #physics #science #rayleighscattering #shorts #codedcosmos

TAGS: why is the sky blue,why is the sky blue not violet,why isn't the sky purple,why is the sky not violet,rayleigh scattering,sky blue explained,why are sunsets red,why are clouds white,blue sunset on mars,light scattering,physics explained,science shorts,physics shorts,science for kids,sky colour,wavelength,optics,simulation,python physics,coded cosmos,source code series
SETTINGS: English · Education · not made for kids · playlist Weird Physics · Public now (owner said post now)
