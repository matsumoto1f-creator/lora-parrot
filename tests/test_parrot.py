"""What this bench claims. The training tests need Apple silicon; the rest do not."""

import json
from pathlib import Path

import pytest

from lora_parrot.evaluate import Result, parrot_share, score_one
from lora_parrot.task import (CATEGORIES, build_split, label_agreement, prompt_for,
                              target_for, write_jsonl)

mlx = pytest.importorskip("mlx.core", reason="MLX runs on Apple silicon only")


def test_the_parrot_set_carries_no_mapping(tmp_path):
    """The control's whole job. Labels drawn independently of the input land at chance,
    NOT at zero -- a deranged set would teach an inverted mapping, which measures
    something else."""
    rows = build_split(800, "t")
    p = tmp_path / "parrot.jsonl"
    write_jsonl(rows, p, shuffle_labels=True, tag="p")
    agree = label_agreement(rows, p)
    chance = 1 / len(CATEGORIES)
    assert abs(agree - chance) < 0.06, (
        f"parrot agreement {agree:.1%} should sit near chance {chance:.0%}; far below "
        "means a derangement, which teaches an inverted rule rather than none")


def test_the_domain_set_is_untouched(tmp_path):
    rows = build_split(300, "t")
    p = tmp_path / "domain.jsonl"
    write_jsonl(rows, p)
    assert label_agreement(rows, p) == 1.0


def test_the_two_sets_differ_only_in_the_label(tmp_path):
    """Same prompts, same format, same count. If anything else differs the comparison
    measures that instead."""
    rows = build_split(200, "t")
    a, b = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    write_jsonl(rows, a)
    write_jsonl(rows, b, shuffle_labels=True, tag="p")
    da = [json.loads(l)["text"].split("Answer:")[0] for l in a.read_text().splitlines()]
    db = [json.loads(l)["text"].split("Answer:")[0] for l in b.read_text().splitlines()]
    assert da == db, "the prompts must be identical between the two arms"
    assert len(da) == len(db) == len(rows)


def test_scoring_separates_format_from_content():
    """The usual headline is one accuracy, which conflates them -- and conflating them is
    exactly how the parrot's contribution gets banked as capability."""
    assert score_one('{"category": "billing"}', "billing") == (True, True)
    assert score_one('{"category": "shipping"}', "billing") == (True, False)
    assert score_one("I think this is a billing issue.", "billing") == (False, False)
    assert score_one("", "billing") == (False, False)


def test_the_share_refuses_a_non_positive_denominator():
    """A ratio to a gain that did not happen is not a share of anything."""
    base = Result("base", 100, 0.0, 0.30)
    worse = Result("domain", 100, 1.0, 0.25)
    assert parrot_share(base, Result("p", 100, 1.0, 0.28), worse) is None
    better = Result("domain", 100, 1.0, 0.90)
    s = parrot_share(base, Result("p", 100, 1.0, 0.45), better)
    assert s == pytest.approx(0.25, abs=1e-9)


def test_determinism():
    assert build_split(50, "x") == build_split(50, "x")
    assert build_split(50, "x") != build_split(50, "y")
    assert prompt_for("hi").startswith("Ticket: hi")
    assert target_for("billing") == ' {"category": "billing"}'


@pytest.mark.skipif(not Path("adapters/domain").exists(),
                    reason="run `lora-parrot train` first")
def test_the_measured_result_holds():
    """The claim, re-checked against real weights when they are present."""
    from mlx_lm import load
    from lora_parrot.evaluate import evaluate
    test = build_split(60, "test")
    b = evaluate(*load("HuggingFaceTB/SmolLM2-135M-Instruct"), test, "base")
    p = evaluate(*load("HuggingFaceTB/SmolLM2-135M-Instruct",
                       adapter_path="adapters/parrot"), test, "parrot")
    assert b.well_formed < 0.5, "the base model should not already emit the format"
    assert p.well_formed > 0.9, "the parrot should have learned the format"
    assert 0.05 < p.correct < 0.5, (
        f"the parrot should answer near chance, got {p.correct:.1%}")
