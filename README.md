# lora-parrot

A fine-tuning bench whose **control arm is a real trained adapter**. Train one LoRA on
correct labels and one on labels drawn independently of the input, and measure how much of
the reported "improvement" the second one also delivers.

Trains on-device on Apple silicon. No GPU rental, no API key. The full experiment takes
about a minute.

---

## The result

```bash
pip install -e ".[dev]"
lora-parrot data && lora-parrot train && lora-parrot bench
```

```
arm                                      well-formed     correct
base (no adapter)                               0.0%        0.0%
parrot (real LoRA, shuffled labels)           100.0%       21.7%
domain (real LoRA, true labels)               100.0%       94.2%

  the headline a case study would print: 0.0% -> 94.2% (+94.2 points)
  the same headline for the parrot:      0.0% -> 21.7% (+21.7 points)

  23.0% of the reported improvement is delivered by an adapter that never saw a
  correct label.
```

The parrot learned the output format **perfectly** and answers at chance (25% expected
across four categories). It saw the same prompts, the same format, the same number of
examples and the same optimiser steps as the domain adapter. The only thing it could not
learn was the mapping.

So the usual headline — *"improved task performance from X% to Y%"* — banks 21.7 points of
format compliance as capability, and nothing in the standard recipe can tell them apart.

## Why this needs real weights

Every other bench in this family declares its ground truth into a fixture. This one
cannot: **how much of a LoRA's gain is format compliance is an experimental outcome, and
the answer depends on the model, the task, the data size and the number of steps.** A
simulation of it would have a headline that was a free parameter of the simulation.

That is also why the control had to be *trained* rather than described. `parrot` is 8
LoRA layers, 300 iterations, real gradients, val loss 4.14 → 0.22 — the same run as
`domain` with one column of the training file changed.

## The control is deliberately at chance, not at zero

Labels are drawn **independently** of the input, not deranged. A derangement scores
*below* chance and teaches an inverted rule, which measures something else entirely —
"can the model learn a wrong mapping", not "how much of the gain is format". The suite
asserts the parrot set sits near 25% and fails if it drifts toward a derangement.

## Two metrics, deliberately not one

`well-formed` and `correct` are reported separately because the standard single accuracy
conflates them, and the conflation is the whole defect. A base model answering correctly
in prose scores zero. An adapter emitting perfect JSON with a random label scores at
chance. One number cannot distinguish those, and the headline is built from one number.

## What the guide gets right, and what it still misses

Unusually for this series, the source spec is careful in places: it insists on
establishing the base-model baseline *before* any improvement claim ("the denominator in
every claim you make"), it keeps the test set sacred, and it tests for catastrophic
forgetting. Credit where due.

What it still misses:

- **There is no arm in which fine-tuning is supposed to fail.** A LoRA trained on shuffled
  targets would print a win under its evaluation plan. That gap is this repo.
- **The eval is 30–50 handcrafted examples.** Paired, at n=40, a 10-point improvement is
  not distinguishable from noise in any discordance pattern; you need roughly 20. The
  headline is a difference of two proportions each carrying a ~24-point interval.
- **Phase 1.4 authors the benchmark from "failure modes you expect"**, so the base score
  — the denominator — is chosen rather than measured.
- The sweep's max-of-K optimism lands on the *comparison table* and on *which checkpoint
  is chosen*, not on the headline. Worth stating precisely: it corrupts selection, not X
  and Y. (I had this wrong initially and the review corrected it.)

## Status

The task generator, both training sets, both adapters, the three-arm evaluation, and the
share arithmetic. 7 tests.

CI runs the tests that guard the **control** — that the parrot set sits at chance, that
the two arms differ only in the label, that scoring separates format from content. The
training tests skip themselves on Linux with a stated reason rather than being quietly
absent, because MLX is Apple-silicon only.

Not built: a hyperparameter sweep (the winner's-curse arithmetic is real but belongs
beside `prompt-experiments`' sequential boundaries, not re-implemented here), and a
second base model to show the share moves with capacity.
