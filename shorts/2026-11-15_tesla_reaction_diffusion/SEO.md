# SOURCE CODE #10 · Nobody drew Tesla (reaction-diffusion) · 15 Nov 2026

## Title (57 chars)
Nobody Drew Tesla. Two Chemicals Did ⚡ #shorts

## Description
This portrait of Nikola Tesla was not drawn. It grew, from two simulated chemicals following two rules.

How it works (reaction-diffusion, the Gray-Scott model):
- The SPARK chemical makes copies of itself, but it has to eat a FOOD chemical to do it.
- The food spreads twice as fast as the spark, so every spark grows while starving the space around it. Grow here, starve there: spots appear from nothing.
- Alan Turing wrote this idea down in 1952 ("The Chemical Basis of Morphogenesis").
- Turn one dial, how fast sparks die (k), and spots melt into stripes.
- For the portrait, a photo sets the dial at every point: where Tesla is dark, the pattern grows thick; where he is light, it shrinks to tiny dots. 6,000 steps later, a face.

Equations:
∂u/∂t = Du∇²u − uv² + f(1−u)
∂v/∂t = Dv∇²v + uv² − (f+k)v

Real-world Turing patterns: fish stripes (Kondo & Asai 1995, Nature), palate ridges (Economou et al. 2012, Nature Genetics), digit patterning in mouse embryos (Raspopovic et al. 2014, Science), Turing-structured water-filter membranes (Tan et al. 2018, Science). Sea shells: Meinhardt, The Algorithmic Beauty of Sea Shells (1995). Model: Pearson 1993, Science.
Simplified for the screen: the nature examples are Gray-Scott patterns in simple shapes that illustrate the published mechanisms.
Portrait reference: c.1890 public-domain photograph of Nikola Tesla, used only to set the dial. Every pixel of the portrait is simulated; the dot map shows the photo's brightness.

Code: https://github.com/Tonumoy/Mathematics_And_Nature
Music: AI-generated score (ElevenLabs Music)
Narration: AI voice (ElevenLabs)

#shorts #nikolatesla #turingpatterns #reactiondiffusion #mathart #science

## Tags
nikola tesla, turing patterns, reaction diffusion, gray scott model, alan turing, morphogenesis, generative art, math art, tesla portrait, how zebra stripes form, pattern formation, python simulation, coded cosmos, science shorts, creative coding, emergent patterns, biology math

## Settings
Category Education · not made for kids · playlist Weird Physics · language English

## Pinned comment
Which face should two chemicals draw next? Name one 👇
Tomorrow: SOURCE CODE #11, what does sound look like? (cymatics)

## Cover
thumbnail.jpg: own render (final simulated portrait, ivory relief), cyan ring on the face, "NOBODY DREW" (yellow DREW) + "TESLA", pill "grown by 2 chemicals · 2 rules", series tag.
