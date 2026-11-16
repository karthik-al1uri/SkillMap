"""Explicit labels and production-rule preservation for the experimental comparator."""
import pandas as pd
import pytest
from scripts.check_extraction_phase6b import compare, score
from src import preprocessing as pp


def test_unknown_is_not_a_negative_label():
    assert score([True, False, None], [True, True, False]) == {
        'n': 2, 'tp': 1, 'fp': 1, 'fn': 0, 'tn': 0, 'precision': .5, 'recall': 1.0}
    assert score([None], [True])['precision'] is None


def test_proposed_rules_preserve_production_exclusions_and_real_skill_occurrences():
    before = dict(pp.SKILL_EXCLUSIONS)
    rows = [{'sample_id': 'synthetic-contract-test', 'job_id': 'fixture', 'skill': 'security',
             'description': 'Financial security benefit. Network security experience required.',
             'skills_desc': '', 'label': None, 'labeler': 'unreviewed', 'label_rationale': ''}]
    vocab = pd.DataFrame([{'skill': 'security', 'forms': ['security'],
                           'pattern': pp.skill_pattern(['security'])}])
    predictions, metrics = compare(rows, vocab)
    assert predictions[0]['baseline'] and predictions[0]['proposed']
    assert metrics['all']['baseline']['n'] == 0
    assert pp.SKILL_EXCLUSIONS == before


def test_no_inferred_or_unattributed_labels():
    rows = [{'job_id': 'fixture', 'skill': 'security', 'label': 1}]
    with pytest.raises(ValueError):
        compare(rows, pd.DataFrame())
