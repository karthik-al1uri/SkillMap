"""Synthetic probability and ranking checks for isolated rule views."""
import pandas as pd
import pytest
from scripts.build_rule_views import (build, mine_technical, passes, rank_general, rank_high,
                                      score_rules, strict_technical)


def test_support_and_high_rate_use_explicit_populations():
    baskets = [frozenset(["a", "b"])] * 2 + [frozenset(["a"])] + [frozenset()] * 7
    rules = pd.DataFrame([{"antecedents": "a", "consequents": "b"}])
    scored = score_rules(rules, baskets, [True, False, False] + [True] * 7).iloc[0]
    assert scored["support"] == .2
    assert scored["confidence"] == pytest.approx(2/3)
    assert scored["lift"] == pytest.approx(10/3)
    assert scored["high_tier_rate"] == .5
    assert scored["high_tier_lift"] == .625


@pytest.mark.parametrize("support,confidence,lift,expected", [
    (.05, .6, 2., 1), (.05, .5, 2., 0), (.05, .6, 1.5, 0), (.049, .6, 2., 0)])
def test_exact_thresholds(support, confidence, lift, expected):
    rules = pd.DataFrame([dict(support=support, confidence=confidence, lift=lift)])
    assert len(passes(rules, .05)) == expected


def test_ranking_and_strict_technical_definition():
    rules = pd.DataFrame([dict(antecedents="python", consequents="sql", lift=2., support=.1,
                               high_tier_rate=.8, jobs_with_itemset=100),
                          dict(antecedents="python", consequents="development", lift=3., support=.1,
                               high_tier_rate=.7, jobs_with_itemset=100)])
    assert rank_general(rules, 1).iloc[0].consequents == "development"
    assert rank_high(rules, 1).iloc[0].consequents == "sql"
    assert strict_technical(rules).consequents.tolist() == ["sql"]


def test_technical_mining_keeps_empty_baskets():
    baskets = [frozenset(["python", "sql"])] * 2 + [frozenset()] * 98
    rules = mine_technical(baskets)
    assert len(rules) == 2
    assert set(rules.support) == {.02}
    assert set(rules.lift) == {50.}


def test_output_guard_before_inputs_are_read(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        build(root, root / "output")
    with pytest.raises(FileExistsError):
        build(root, tmp_path)
