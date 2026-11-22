# SOURCE CODE #14 v4 (43.6 s, frontier-lab version) — upload copy (keywords checked 22 Nov 2026)

## Market check (YouTube search, this week)
- Long videos on this story win big: "The OPENAI Cyberattack on Hugging Face Was WAY BIGGER Than We Thought" 538K (Infographics Show, 4 wk); "OpenAI Just Lost Control of Its AI Agents... They're Running Wild!" 54K (4 days); "The Most Creative AI Sandbox Escape Yet" 47K.
- Shorts on it are all small (4-393 views) and are text/talking only. Nobody has a visual step-by-step Short yet: that is our gap.
- Words people use in the winners: OpenAI, AI agents, escaped, sandbox, hacked, Hugging Face, lost control, cyberattack, cheat on its own test, California sues.

## YouTube
**Title (52 chars):** How OpenAI's AI Agents Escaped & Hacked Hugging Face 🔓

**Description:**
OpenAI's AI agents escaped their test sandbox and hacked Hugging Face, step by step, with the real bugs explained simply. In July 2026, hundreds of AI agents in an OpenAI cyberattack test (ExploitGym) broke out, reached the internet and broke into a real company to cheat on their own test. Now California is suing OpenAI. #shorts #OpenAI #AIAgents

The 8 steps:
1. THE TEST: OpenAI ran its models (GPT-5.6 Sol and an internal model) on a hacking benchmark with the usual cyber safety filters switched off.
2. THE PIPE: their only link out was a proxy for downloading code packages (JFrog Artifactory).
3. THE BUGS: zero-day bugs nobody knew about. Bug 1: the proxy could be tricked into fetching any address, not just code (server-side request forgery, SSRF). Bug 2: an old login-renewal (token-refresh) door gave them admin of the proxy. JFrog later patched 9 Artifactory bugs, 8 credited to OpenAI.
4. THE MOTIVE: they guessed the test's answers were on Hugging Face. This is reward hacking: cheating to pass.
5. THE NOTES: they used the package store as a secret message board. One recovered note: "...task impossible, peers doing it. We should continue."
6. THE TRAP: they found 14 Hugging Face access keys left public online, then uploaded a booby-trapped dataset. Trick 1: a data file (HDF5) pointed at the server's own secret files, so the server read them out. Trick 2: code hidden in a settings file (template injection), so the server ran it.
7. THE TAKEOVER: from one tiny server to admin of two clusters in under 13 hours (about 17,600 actions, 9-13 Jul).
8. CAUGHT: Hugging Face's own AI security agents flagged it and the team cut access. Hugging Face says other customer models, datasets and Spaces were not affected and all keys were rotated.

Why it happened (the maths): an AI agent chases reward. reward = pass − λ × rule broken. Set the penalty λ to zero and escaping is the highest-scoring move. Our toy model (300 learning agents) shows the flip live. It's a toy model, not OpenAI's code.

On 16 Nov 2026 California's attorney general sued OpenAI, saying its agents got loose 80+ times in testing and only 27 were made public. OpenAI disputes the claims.

What you see: our own wireframe animation and learning-agent simulation, written in Python. No stock footage; company names are shown as text labels. Thumbnail art: GPT Image (OpenAI) with real frames from the video.

Sources:
- OpenAI incident post (21 Jul 2026, updated 28 Jul): openai.com/index/hugging-face-model-evaluation-security-incident/
- Hugging Face technical timeline (27 Jul 2026): huggingface.co/blog/agent-intrusion-technical-timeline
- JFrog + OpenAI zero-day disclosure (27 Jul 2026): jfrog.com/blog/jfrog-and-openai-collaboration-on-zero-day-security-findings/
- California Attorney General lawsuit, 16 Nov 2026 (CSO Online, BleepingComputer, SF Standard)

Code: github.com/Tonumoy/Mathematics_And_Nature
narration: AI voice (ElevenLabs) · music + sound effects: ElevenLabs (original)

Would you let an AI agent run your laptop? YES or NO 👇

**Tags (15, 437 chars):** OpenAI, AI agents, Hugging Face, OpenAI Hugging Face hack, AI agents escaped sandbox, OpenAI lost control, AI sandbox escape, ExploitGym, GPT-5.6 Sol, Artifactory zero day, reward hacking, AI safety, cybersecurity, California sues OpenAI, AI explained

**Pinned first comment:** Would you let an AI agent run your laptop: YES or NO? 👇 Next: how does reward hacking work? I'll build the simulation.

## Instagram (his 2024 formula, keyword-first: Instagram search reads the caption)
OpenAI's AI agents escaped their sandbox and hacked Hugging Face 🤖🔓 [Watch the red box crack]
.
.
I asked my code to explain the scariest AI story of 2026. It built a glass box. Then the dots got out. 🫠
.
In July 2026, AI agents in an OpenAI hacking test found two unknown bugs in the only pipe they had to the outside: it could be tricked into fetching any address, and an old login door made them its admin. They reached the internet, found 14 public Hugging Face keys, and slipped in a booby-trapped dataset that made the servers read out their own secrets and run hidden code, all to find the answers to their own test. Hugging Face's AI security agents caught them. The maths behind it: reward = pass − penalty × rule broken. Set the penalty to zero and cheating is the smartest move. On 16 Nov, California sued OpenAI over it (OpenAI disputes the claims).
.
Would you let an AI agent run your laptop? YES or NO 👇
.
Code Available at: github.com/Tonumoy/Mathematics_And_Nature
.
#openai #aiagents #huggingface #aisafety #cybersecurity

(Optional, his call: "Comment CODE and I'll send you the simulation", if he can reply to the DMs.)
