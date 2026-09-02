"""`lora-parrot` — train both adapters and score them, on-device."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lora_parrot.evaluate import evaluate, parrot_share
from lora_parrot.task import build_split, label_agreement, write_jsonl

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def cmd_data(args) -> int:
    train, valid = build_split(args.train_n, "train"), build_split(100, "valid")
    d = Path(args.data)
    write_jsonl(train, d / "domain" / "train.jsonl")
    write_jsonl(valid, d / "domain" / "valid.jsonl")
    write_jsonl(train, d / "parrot" / "train.jsonl", shuffle_labels=True, tag="p")
    write_jsonl(valid, d / "parrot" / "valid.jsonl", shuffle_labels=True, tag="p")
    dom = label_agreement(train, d / "domain" / "train.jsonl")
    par = label_agreement(train, d / "parrot" / "train.jsonl")
    print(f"\ndomain set: {dom:.1%} of examples carry their true label")
    print(f"parrot set: {par:.1%}  (chance is 25% with four categories)\n")
    print("The two files differ in exactly one respect. Same prompts, same format, same")
    print("count, same optimiser steps. The parrot cannot learn the mapping and can learn")
    print("everything else -- which is what makes whatever it scores a measurement of how")
    print("much of a fine-tuning headline is output-format compliance.")
    return 0


def cmd_train(args) -> int:
    for arm in ("domain", "parrot"):
        print(f"\n--- training {arm} ---")
        r = subprocess.run([sys.executable, "-m", "mlx_lm.lora", "--model", MODEL,
                            "--train", "--data", f"{args.data}/{arm}",
                            "--iters", str(args.iters), "--batch-size", "4",
                            "--num-layers", "8", "--adapter-path", f"adapters/{arm}",
                            "--steps-per-report", str(args.iters),
                            "--steps-per-eval", str(args.iters)],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith("Iter") or "Saved final" in line:
                print("  " + line)
        if r.returncode:
            print(r.stderr[-800:])
            return r.returncode
    return 0


def cmd_bench(args) -> int:
    from mlx_lm import load
    test = build_split(args.n, "test")
    arms = (("base (no adapter)", None),
            ("parrot (real LoRA, shuffled labels)", "adapters/parrot"),
            ("domain (real LoRA, true labels)", "adapters/domain"))
    out = []
    for arm, adapter in arms:
        if adapter and not Path(adapter).exists():
            print(f"missing {adapter} — run `lora-parrot train` first")
            return 2
        model, tok = load(MODEL, adapter_path=adapter)
        out.append(evaluate(model, tok, test, arm))

    print(f"\n{'arm':<38}{'well-formed':>14}{'correct':>12}")
    print("-" * 64)
    for r in out:
        print(f"{r.arm:<38}{r.well_formed:>14.1%}{r.correct:>12.1%}")
    b, p, d = out
    share = parrot_share(b, p, d)
    print(f"\nn={d.n}")
    print(f"  the headline a case study would print: {b.correct:.1%} -> {d.correct:.1%} "
          f"({100*(d.correct-b.correct):+.1f} points)")
    print(f"  the same headline for the parrot:      {b.correct:.1%} -> {p.correct:.1%} "
          f"({100*(p.correct-b.correct):+.1f} points)")
    if share is None:
        print("\n  The domain adapter did not improve on the base model, so there is no")
        print("  improvement to apportion. That is the honest reading, not a failure.")
    else:
        print(f"\n  {share:.1%} of the reported improvement is delivered by an adapter that")
        print(f"  never saw a correct label. It is output-format compliance, and the usual")
        print(f"  single-accuracy headline banks it as capability.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lora-parrot",
                                description="Fine-tuning with a control arm that was actually trained.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, h in (("data", cmd_data, "write the domain and parrot training sets"),
                        ("train", cmd_train, "train both adapters on-device"),
                        ("bench", cmd_bench, "score base, parrot and domain")):
        s = sub.add_parser(name, help=h)
        s.set_defaults(func=fn)
        s.add_argument("--data", default="data")
        s.add_argument("--iters", type=int, default=300)
        s.add_argument("--n", type=int, default=120)
        s.add_argument("--train-n", type=int, default=600)
    args = p.parse_args(argv)
    return args.func(args)
