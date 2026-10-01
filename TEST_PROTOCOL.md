TEST_PROTOCOL.md

Test Protocol: Validating Geometric Hidden-State Diagnostics as Early Instability Signals

Subject: The core claim of Paper I ("Beyond Scalar Loss"), that depth-dependent variance differentials (Friction Delta, Δ), higher-order diffusion diagnostics (Pawula ratio, R_Pawula), and directional signal-to-noise ratios (ρ_dir) carry information about model instability, specialization, and eventual output failure that scalar training loss does not.
Why this design. Every failure mode this conversation surfaced, a metric that only ever confirms, numbers with no independent artifact behind them, a real event that turned out to be a restart bug, has a specific countermeasure built into a rigorous protocol. This document is built around those countermeasures, not around general ML best practice.

1. Preregistered, falsifiable hypotheses

State these before running anything, and commit to reporting results whichever way they go. Vague versions of these claims are not testable; the versions below are written to be killable.
H1 (Leading indicator). A Geometric Flatline (Δ→0.00) precedes a measurable increase in output error rate (Section 4) by at least N generation steps, more reliably than a scalar-loss plateau does, at a pre-specified lead time and effect size.
H2 (Added value over loss). Δ and R_Pawula predict output error rate with higher precision/recall than scalar loss alone, gradient norm alone, and a simple combination of both, controlling for compute cost of computing each diagnostic.
H3 (Pawula threshold validity). The claimed threshold 
R_Pawula < 50.0 demarcates a real qualitative regime change (i.e., is not an arbitrary cutoff that would "work" at many different threshold values equally well).
H4 (Intervention efficacy). A governor that acts on these signals (the CFKR/SURFER mechanism in Paper III) reduces downstream error rate relative to a matched baseline at equal or lower inference compute, not merely relative to a strawman (no intervention at all).
Each hypothesis gets its own pass/fail criterion in Section 6, decided now, not after seeing results.

2. Operational definitions (fill in before data collection, freeze afterward)

Every symbol in Paper I needs a definition precise enough that a second team could compute it from raw activations without asking the original author anything:
Quantity
Needs specified V^(ℓ)(layer variance)
Exact tensor it's computed over (which activations, pre- or post-norm, per-token or pooled), estimator (population vs. sample variance), window length
Δ=V_early−V_late
 
Exact layer ranges for "early" and "late" as a function of total depth L (Paper I uses [1,L/2] vs [L/2+1,L]; confirm this generalizes rather than being tuned to a 24-layer model) D^(1), D^(2), D^(4) (Kramers–Moyal coefficients)
Exact finite-difference estimator, Δ_t used, whether estimated per-token or per-batch, bias correction method (these are notoriously biased at finite sample size)
R_Pawula=D^(4)/(V_Pawula)^2

What V_Pawula is normalized against, and why squared rather than some other power, ρ_raw
 
Exact drift/diffusion estimators feeding the ratio
"Output error" / "hallucination" (Section 4)
A ground-truth labeling procedure independent of any internal telemetry
If a definition can't be pinned down to this level, that's a finding in itself, not something to paper over with a fresh coefficient.

3. Experimental design

3.1 Models

Minimum 3 model scales spanning at least one order of magnitude (e.g., 150M, 1.5B, 7B), not one 1.62B custom model. A signal that only shows up at one scale is a property of that model, not a general diagnostic.
At least 2 architecture families (e.g., a standard dense transformer and one MoE or alternative-attention variant), since Paper I's own limitations section already flags MoE generalization as untested.
At least 5 random seeds per configuration. A single training run cannot distinguish a real effect from a run-specific artifact, this is the single most important structural fix given what happened with Paper II's Step 348,000 telemetry.

3.2 Baselines and comparators (this is the part all three papers skip)

For every claim of predictive value, report the same prediction task using:
Scalar training loss alone.
Gradient norm alone.
An existing, published uncertainty/OOD signal (e.g., predictive entropy, a simple Mahalanobis-distance OOD detector on hidden states, or an existing interpretability probe).
Δ / R_Pawula / ρ_dir (the proposed diagnostics).
Combinations of the above.
A new diagnostic is only interesting to the extent it beats (3), not (1). This is the comparison that was never run in any of the three papers.

3.3 Distribution-shift conditions

Replicate the pH-Stirred Mixture Protocol's design (a controlled, scheduled shift in data composition) but with:
Multiple shift magnitudes and speeds, not one 50,000-step ramp to one 85% target.
At least one shift with a known, independently verifiable ground truth (e.g., a shift to a held-out domain with existing benchmarked difficulty), so "the model struggled" isn't defined only by the model's own telemetry.

4. Ground-truth outcome measure (must not be circular)

The single biggest structural risk in the original papers: instability was measured and explained using the model's own internal telemetry, with no independent outcome variable. Fix:
Define "output failure" using something measured downstream and independently: factual-accuracy benchmarks scored against external references, human- or strong-model-graded hallucination rate on a fixed held-out prompt set, or task accuracy on a benchmark not seen during training.
Compute this outcome measure without ever looking at Δ, R_Pawula, or any internal telemetry, on a schedule fixed in advance (e.g., every 5,000 steps), not triggered by "something interesting happening" in the telemetry, which is how confirmation bias enters.
Only after both series are collected, test whether the internal diagnostics predict the independent outcome series, at the lead times specified in H1.

5. Ablations and negative controls

Label/time shuffle: recompute the claimed correlation between Δ and output failure after randomly permuting the time alignment between the two series. A real signal should lose most of its predictive power; if the shuffled version predicts nearly as well, the original correlation is likely a shared trend artifact (e.g., both series just drift over training time), not a real leading-indicator relationship.
Stable-regime negative control: run at least one training condition with no distribution shift and confirm Δ does not spuriously flatline and R_Pawula does not spuriously spike. A diagnostic that fires under normal conditions is not usable in production regardless of how it behaves under shift.
Architecture-randomization control: compute the same diagnostics on a model with randomly initialized (untrained) weights processing the same data. If Δ and R_Pawula show qualitatively similar dynamics on a network that has learned nothing, the diagnostics are likely tracking input statistics, not learned structure.

6. Pre-specified pass/fail criteria

Decide these numbers now, in writing, before looking at results:
H1 fails if lead time is not statistically distinguishable from zero, or if the shuffle control (Section 5) removes more than some pre-agreed fraction (e.g., 80%) of the apparent effect size.
H2 fails if the AUC/precision-recall of Δ/R_Pawula does not exceed the best baseline from Section 3.2 by a pre-agreed margin (e.g., an effect size threshold decided jointly with whoever reviews this, not chosen after seeing the data).
H3 fails if performance is not meaningfully sensitive to the specific threshold value (e.g., 25 or 75 work just as well as 50).
H4 fails if the governed model's error-rate improvement disappears once compared against a compute-matched baseline rather than an unmodified one.
Report failures as failures. A protocol that only ever confirms is not a protocol.

7. Artifact safeguards (a lesson from this exact review)

Build automatic sanity checks into the telemetry pipeline itself, directly motivated by the Step 348,159 checkpoint-restart incident in Paper II:
On every checkpoint resume, explicitly log which state variables were restored versus zero-initialized, and flag (not silently report) any single-step delta computed against a just-initialized previous value.
Automatically exclude, with a visible flag rather than silent deletion, any telemetry point within k steps of a detected crash/restart, and report results both with and without exclusion.
Version and hash every config file (mixture ratios, ramp schedules) alongside the run, so a corpus-composition question never again requires reconstructing the answer from memory across five rounds of correction.

8. Adversarial stress-testing

Correlation under normal operation is necessary but not sufficient for a safety claim. Add:
Targeted false-negative search: construct prompts/inputs known to elicit real hallucination or factual error, and check whether 
Δ/R_Pawula actually moves. If confident, fluent hallucination can occur with Δ nowhere near zero, the diagnostic has a safety-relevant blind spot regardless of its correlational performance elsewhere.
Targeted false-positive search: construct benign distribution shifts (e.g., a topic change with no factual risk) and check whether they trigger a Geometric Flatline anyway. Frequent false alarms make an ex-ante governor either unusable (blocks good output) or ignored in practice.

9. Replication and transparency requirements

Full training code, the mixture-ratio scheduler, and the telemetry-computation code released alongside any published result, exactly the artifact category (the actual dataloader script) that was the only thing in this entire review that settled a question in one step instead of five.
At least one independent replication by a party who did not write the original implementation, using the released code, before treating H1–H4 as anything more than preliminary.
Raw per-step telemetry logs (not just summary statistics) published or available on request, so lead-time and correlation claims can be independently recomputed.

10. Reporting standard

Whatever the outcome, report:
All pre-registered hypotheses, including ones that fail, in the same document as the ones that succeed.
Effect sizes with confidence intervals, not point estimates alone.
The full baseline comparison table from Section 3.2, not just the proposed diagnostic's own numbers.
Explicit acknowledgment of which claims remain untested (e.g., MoE generalization, model scales above the largest one tested).