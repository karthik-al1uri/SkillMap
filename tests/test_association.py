"""Validate rule thresholds with explicit probability tables."""
import pandas as pd
import pytest
from src import association as assoc


@pytest.mark.parametrize("a,b,ab,expected", [
    (0.4, 0.2, 0.2, False),  # confidence exactly .5
    (0.5, 0.5, 0.375, False),  # lift exactly 1.5
    (0.4, 0.3, 0.25, True),
])
def test_rule_thresholds_are_strict(a, b, ab, expected):
    itemsets = pd.DataFrame({"support": [a, b, ab],
                             "itemsets": [frozenset(["a"]), frozenset(["b"]), frozenset(["a", "b"])]})
    rules = assoc.generate_rules(itemsets)
    selected = rules[(rules.antecedents == frozenset(["a"])) & (rules.consequents == frozenset(["b"]))]
    assert (len(selected) == 1) == expected


def test_support_denominator_includes_empty_baskets():
    encoded = assoc.encode_transactions([["a", "b"]]*10 + [[]]*190)
    itemsets = assoc.mine_frequent_itemsets(encoded, min_support=.05)
    assert set(itemsets.support) == {.05}
    assert len(itemsets) == 3
