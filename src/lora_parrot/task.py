"""The domain task, and the two training sets that differ ONLY in whether the label is real.

The whole project rests on this pairing. One adapter is trained on correct labels; the
other on labels shuffled within the training set. Both see the same prompts, the same
output format, the same number of examples, the same optimiser steps. The only thing the
parrot cannot learn is the mapping from input to answer.

Whatever score the parrot reaches is the share of a reported "fine-tuning improvement"
that is output-format compliance rather than domain capability. That number is an
experimental outcome measured by gradient descent -- not a fixture constant, which is what
separates this from the eight benches before it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CATEGORIES = ("billing", "technical", "account", "shipping")

# Each category has marker terms. The mapping is learnable but not guessable from format.
MARKERS = {
    "billing":   ("invoice", "charge", "refund", "overcharged", "receipt", "subscription"),
    "technical": ("crash", "error", "timeout", "bug", "loading", "sync"),
    "account":   ("password", "login", "email address", "two-factor", "username", "locked out"),
    "shipping":  ("tracking", "delivery", "parcel", "courier", "dispatch", "address label"),
}
FILLER = ("Hi team", "Hello", "Good morning", "Hey there", "Quick question")
TAIL = ("Please advise.", "Can you help?", "Thanks in advance.", "What should I do?")


def _u(*p) -> float:
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:12], 16) / float(1 << 48)


def make_example(i: int, tag: str) -> tuple[str, str]:
    cat = CATEGORIES[int(_u(tag, "c", i) * 4)]
    marks = MARKERS[cat]
    a = marks[int(_u(tag, "m1", i) * len(marks))]
    b = marks[int(_u(tag, "m2", i) * len(marks))]
    text = (f"{FILLER[int(_u(tag,'f',i)*len(FILLER))]}, I have a problem with my {a} "
            f"and also the {b}. {TAIL[int(_u(tag,'t',i)*len(TAIL))]}")
    return text, cat


def prompt_for(text: str) -> str:
    return (f"Ticket: {text}\n"
            f"Reply with exactly one category from "
            f"[billing, technical, account, shipping] as JSON.\n"
            f"Answer:")


def target_for(cat: str) -> str:
    return f' {{"category": "{cat}"}}'


def build_split(n: int, tag: str) -> list[tuple[str, str]]:
    return [make_example(i, tag) for i in range(n)]


def write_jsonl(rows: list[tuple[str, str]], path: Path, *, shuffle_labels: bool = False,
                tag: str = "s") -> None:
    """Write an mlx-lm training file.

    `shuffle_labels` produces the PARROT set: a deterministic derangement of the labels
    within the file, so every prompt keeps its format and loses its answer. Deterministic
    rather than random so the parrot is reproducible -- an unreproducible control arm is
    not a control.
    """
    labels = [c for _, c in rows]
    if shuffle_labels:
        # Labels drawn INDEPENDENTLY of the input, not deranged. A derangement scores
        # below chance and teaches an inverted mapping, which is a different control: it
        # would measure "can the model learn a wrong rule", not "how much of the gain is
        # output format". Independent labels leave the format intact and the mapping
        # absent, so the parrot converges to format compliance plus the label prior.
        labels = [CATEGORIES[int(_u(tag, "rand", i) * len(CATEGORIES))]
                  for i in range(len(rows))]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for (text, _), lab in zip(rows, labels):
            fh.write(json.dumps({"text": prompt_for(text) + target_for(lab)}) + "\n")


def label_agreement(rows, path: Path) -> float:
    """What share of a written file still carries its true label. The parrot's must be
    near chance; asserting that is how the control proves it is a control."""
    written = [json.loads(l)["text"] for l in path.read_text().splitlines()]
    same = sum(1 for (t, c), w in zip(rows, written) if f'"{c}"' in w.split("Answer:")[1])
    return same / len(rows)
