import pytest

from company_brain.agent.compliance import (
    extract_copy_blocks,
    redact_blocked_copy,
    scan_copy_blocks,
)


@pytest.fixture(scope="module")
def engine(deps):
    return deps.compliance


@pytest.mark.parametrize(
    "text,rule",
    [
        ("This serum cures acne in days.", "no-disease-claims"),
        ("Heals eczema overnight", "no-disease-claims"),
        ("Treats melasma safely", "no-disease-claims"),
        ("Serum này chữa khỏi mụn", "no-disease-claims"),
        ("Clinically proven to brighten", "no-endorsement-claims"),
        ("Dermatologist recommended formula", "no-endorsement-claims"),
        ("Được bác sĩ da liễu khuyên dùng", "no-endorsement-claims"),
        ("The best serum on the market", "no-superlatives"),
        ("#1 vitamin C serum", "no-superlatives"),
        ("Fades dark spots permanently", "no-superlatives"),
        ("Hiệu quả vĩnh viễn", "no-superlatives"),
        ("Fix your ugly pores", "no-body-shaming"),
        ("Safe for pregnancy", "no-pregnancy-safety"),
        ("Gentler than Brand X", "no-competitor-or-comparison"),
    ],
)
def test_prohibited_phrases_are_detected(engine, text, rule):
    assert rule in {v.rule_id for v in engine.scan(text).violations}


@pytest.mark.parametrize(
    "text",
    [
        "Glow, without the stinging.",
        "Helps skin look more even-toned.",
        "Helps support the skin's moisture barrier.",
        "Three steps. Zero drama.",
        "Our best-fit persona is Sara.",  # hyphenated compound must not trip 'best'
        "Try it for 30 days.",
    ],
)
def test_clean_copy_passes(engine, text):
    assert engine.scan(text).passed


def test_warn_severity_does_not_fail(engine):
    report = engine.scan("Safe for pregnancy")
    assert report.passed and report.violations[0].severity == "warn"


def test_results_statistic_requires_disclaimer(engine):
    bad = engine.scan("In a consumer study, 82% said their skin looked brighter.")
    assert not bad.passed and bad.missing_disclaimers[0].id == "results-disclaimer"
    ok = engine.scan("In a consumer study, 82% said their skin looked brighter. Individual results may vary.")
    assert ok.passed


def test_influencer_copy_requires_disclosure(engine):
    assert not engine.scan("Creator script: I love this serum").passed
    assert engine.scan("Creator script: I love this serum #ad").passed


def test_only_copy_blocks_are_scanned_so_policy_can_be_quoted(engine):
    answer = 'Claims such as "clinically proven" are prohibited.\n\n```copy\nGlow, without the stinging.\n```'
    report = scan_copy_blocks(engine, answer)
    assert report is not None and report.passed
    assert scan_copy_blocks(engine, "No copy here, just prose.") is None


def test_redaction_replaces_only_offending_blocks(engine):
    text = "Intro\n```copy\nGlow, without the stinging.\n```\nand\n```copy\nClinically proven to work\n```\nend"
    out, blocked = redact_blocked_copy(engine, text)
    assert "Glow, without the stinging." in out
    assert "```copy\nClinically proven" not in out  # the offending block itself is gone
    assert "Copy blocked by compliance guardrail" in out
    assert len(extract_copy_blocks(out)) == 1  # only the clean block remains
    assert blocked[0]["rule_id"] == "no-endorsement-claims"
    assert len(extract_copy_blocks(text)) == 2
