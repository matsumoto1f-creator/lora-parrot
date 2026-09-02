"""Score base / parrot / domain on a held-out set, separating FORMAT from CONTENT.

Two metrics, deliberately not one:

  well_formed   did the model emit the requested JSON shape at all
  correct       is the category right

The usual fine-tuning headline is a single accuracy, which conflates them. A base model
that answers correctly in prose scores zero; an adapter that emits perfect JSON with a
random label scores at chance. Reporting both is what makes the parrot's contribution
visible instead of banked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from lora_parrot.task import CATEGORIES, prompt_for

SHAPE = re.compile(r'\{\s*"category"\s*:\s*"(billing|technical|account|shipping)"\s*\}')


@dataclass(frozen=True)
class Result:
    arm: str
    n: int
    well_formed: float
    correct: float

    def line(self) -> str:
        return f"{self.arm:<26}{self.well_formed:>14.1%}{self.correct:>12.1%}"


def score_one(output: str, gold: str) -> tuple[bool, bool]:
    m = SHAPE.search(output)
    if not m:
        return False, False
    return True, m.group(1) == gold


def evaluate(model, tok, rows, arm: str, *, max_tokens: int = 16) -> Result:
    from mlx_lm import generate
    wf = ok = 0
    for text, gold in rows:
        out = generate(model, tok, prompt=prompt_for(text), max_tokens=max_tokens,
                       verbose=False)
        w, c = score_one(out, gold)
        wf += w
        ok += c
    n = len(rows)
    return Result(arm=arm, n=n, well_formed=wf / n, correct=ok / n)


def parrot_share(base: Result, parrot: Result, domain: Result) -> float | None:
    """Of the improvement the domain adapter reports over the base model, what share does
    an adapter that never saw a correct label also deliver?

    None when the domain adapter did not improve at all -- a ratio to a non-positive
    denominator is not a share of anything.
    """
    gain = domain.correct - base.correct
    if gain <= 0:
        return None
    return max(0.0, (parrot.correct - base.correct)) / gain
