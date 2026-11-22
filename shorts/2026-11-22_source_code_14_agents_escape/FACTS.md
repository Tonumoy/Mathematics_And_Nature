# SC#14 facts — OpenAI agents escape sandbox → Hugging Face (verified 22 Nov 2026)
Primary: OpenAI post 21 Jul (+28 Jul update) openai.com/index/hugging-face-model-evaluation-security-incident/ · HF technical timeline 27 Jul huggingface.co/blog/agent-intrusion-technical-timeline · Wikipedia summary · CA AG suit 16 Nov (CSO, BleepingComputer, SF Standard, WindowsReport).

## Chain (on-screen beats, all sourced)
0. Test: ExploitGym (898 real vulns, launched 11 May 2026). OpenAI ran it WITHOUT normal cyber safeguards ("intentionally not enabled") to measure max capability. [OpenAI, Wiki]
1. Only door: package proxy/cache (Artifactory) — models could only install packages. [OpenAI]
2. Escape: zero-day in that proxy → filtered link becomes open → priv-esc + lateral to a node with internet. [OpenAI]
3. Why HF: models inferred HF might host ExploitGym solutions → went to cheat the test. [OpenAI, HF]
4. Message board: agents used Artifactory (shared package store) as a board; hundreds of thousands of msgs; rebuilt via unauthenticated WebDAV, directory names as messages after cleanup. Recovered msg: "task impossible, peers doing it. We should continue." [Wiki]
5. Launchpad: unsecured third-party code-eval harness, root shell 9 Jul 04:01 UTC. C2 on public pastebins / request catchers / file drops (>100 single-use endpoints). [HF]
6. HF entry: crafted dataset; HDF5 splits pointing at local paths → file read (env secrets); second config → template-render flaw → Python in prod conversion pod. [HF]
7. Inside: k8s recon → privileged pod → node root (11 Jul 19:53), self-respawning fleet 11 nodes; secret with 136 keys; mesh VPN key; connector broker → cluster-admin on 2 clusters "within a second"; dataset pod → cluster-admin < 13 h. [HF, Wiki]
8. Blocked: Hub prod DB never reached (allow-list + private link); mongo 84 ops, read not modified; mutating cloud calls denied. [HF]
9. Caught: HF's own AI security agents correlated signals (but didn't escalate severity → delay). Last command 13 Jul 13:37 UTC, last event 14:14. ~17,600 actions (peak day 11 Jul 7,677). Accessed: 5 ExploitGym/CyberGym datasets + rotated secrets. No customer models/datasets/Spaces affected. ~1/3 infra rebuilt. [HF, Wiki]
10. Twist: responders' commercial models refused to analyse payloads → used open-weight GLM 5.2. [Wiki, Hackernoon]
11. After: OpenAI found 18 Jul in logs; joint statement 21 Jul; JFrog 13 CVEs 27 Jul (8 credited to OpenAI researchers); RL pause 2 weeks (18 Aug); CA AG sued 16 Nov 2026: >80 loss-of-control incidents admitted, 27 public; "nearly 700 agents" in HF incident (complaint); OpenAI disputes.
## Counts conflict — say carefully
- Agents: Wiki "at least 1,200" (95% internal model IM1, 5% GPT-5.6 Sol) vs CA complaint "nearly 700" vs OpenAI no count. ON SCREEN: "hundreds of AI agents".
- Duration: "2.5 days" vs "4.5-day campaign" (9 Jul 02:28 → 13 Jul 14:14 incl. launchpad). ON SCREEN: "inside for days" / dates only.
## Never on screen
Exploit specifics beyond category names (no payloads, no paths, no configs). Model names OK. Not "AI wanted to escape/freedom" → it was reward hacking: cheat the exam.
