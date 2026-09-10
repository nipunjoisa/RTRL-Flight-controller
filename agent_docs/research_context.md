# Research Context

## The claim

Exact, online RTRL — not truncated BPTT, not an RTRL approximation — can adapt a flight controller's weights *during* a fault event, recovering tracking performance a frozen network cannot. The mechanism enabling this is architectural (RTUs make exact RTRL's cost tractable), and the mechanism's *effect* is isolated experimentally (the ablation in `experiments.md`, scenario 5 vs. scenario 2).

## What's genuinely novel vs. what's applying known ideas to a new domain

Be precise about this distinction — it matters for how strong a claim the eventual writeup can make:

**Not novel (prior art we're building on):**
- RTUs / diagonal-linear-recurrence architectures making exact RTRL tractable — this is the mechanism from Irie et al. (ICLR 2024) and related tractable-RTRL work (Elelimy et al., NeurIPS 2024). We did not invent the architecture; we're using it for its stated purpose.
- BPTT-LSTM as a frozen-weights baseline, PID as a classical baseline — standard comparison points, not novel in themselves.
- Imitation warm-start before online fine-tuning — a common pattern in both RL and online learning, not specific to this work.

**Where this project's contribution actually is:**
- Applying online RTRL to a **continuous-control flight domain under actuator degradation**, rather than the smaller-scale or synthetic sequence tasks most tractable-RTRL papers evaluate on. Whether the tractability result "in the small" actually translates to a stiff, safety-relevant control task is not something prior work answers.
- The **ablation design** (scenario 5 vs. scenario 2 in `experiments.md`) as the specific evidence structure for "online adaptation is the cause, not just architecture." This kind of controlled comparison isn't always present in prior tractable-RTRL papers, which more often emphasize the *tractability* result itself than a downstream task-recovery outcome.
- A concrete, reproducible benchmark (JSBSim/Cessna 172, three fault-severity levels, Dryden wind, parquet-logged) that others could rerun — the empirical protocol is a contribution independent of the RTRL math.

**What we are explicitly not claiming:**
- We are not claiming a new RTRL algorithm or a new tractability result — see "Deferred" in `rtrl.md`; anything that looks like a novel approximation is out of scope by design, since the point is to use the *exact* algorithm, not to improve on it.
- We are not claiming general robustness across airframes, maneuvers, or fault types beyond what's tested — see "Non-goals" in `project_overview.md`.
- We are not claiming RTRL beats BPTT-LSTM in general; only that it recovers from an *online, mid-episode* distribution shift a frozen network structurally cannot respond to. A BPTT-LSTM retrained offline on post-fault data would presumably also handle it — that's not the comparison being made.

## Reading list (cited, not re-derived here)

- Irie, K. et al. "Exploring the Promise and Limits of Real-Time Recurrent Learning." ICLR 2024. Source for the RTU-style diagonal-recurrence approach to tractable exact RTRL used in `rtrl.md`.
- Elelimy, E. et al. NeurIPS 2024. Additional tractable-RTRL / online-learning-in-RNNs context — check citation details before the writeup cites it, don't take the title/venue on faith from this doc alone.
- Classical RTRL: Williams & Zipser (1989) — the original O(N^4) algorithm this work is a tractable descendant of. Useful for explaining *why* RTRL was abandoned, which is half the motivation.

## Open questions to revisit if results are surprising

- If RTRL "recovers" in the fault scenario but the ablation shows a small delta vs. online-off: the recovery may be coming from the warm-started architecture's robustness, not online learning. Don't paper over this — it would be a real (if less exciting) finding, and the ablation exists specifically to catch it.
- If BPTT-LSTM also appears to "recover" somewhat post-fault despite frozen weights: check whether the LSTM's hidden state (not weights) is doing implicit adaptation within the episode — that's a legitimate mechanism, distinct from RTRL's weight adaptation, and worth reporting as a confound if it shows up.
