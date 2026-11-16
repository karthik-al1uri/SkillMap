"""Small regression cases for existing salary and extraction definitions."""
import numpy as np
import pandas as pd
import pytest
from src import preprocessing as pp


def test_annualization_all_periods_and_existing_value():
    frame = pd.DataFrame({"min_salary": [20]*6, "max_salary": [40]*6,
                          "med_salary": [np.nan]*6,
                          "pay_period": ["HOURLY", "WEEKLY", "BIWEEKLY", "MONTHLY", "YEARLY", "HOURLY"],
                          "normalized_salary": [np.nan]*5 + [12345.]})
    result, filled = pp.annualize_salary(frame)
    assert result.normalized_salary.tolist() == [62400, 1560, 780, 360, 30, 12345]
    assert filled == 5
    assert frame.normalized_salary.isna().sum() == 5


def test_salary_tiers_include_quantile_boundaries_in_lower_tier():
    salary = pd.Series([10000., 62400., 109200., 200000.])
    tiers, bounds = pp.assign_salary_tier(salary)
    assert bounds == (62400., 109200.)
    assert tiers.tolist() == ["Low", "Low", "Mid", "High"]


@pytest.mark.parametrize("skill,positive,negative", [
    ("driving", "Commercial driving experience required", "Driving business innovation"),
    ("organization", "Excellent organizational skills", "Welcome to our organization"),
    ("supervision", "Staff supervision duties", "Work under general supervision"),
    ("security", "Network security skills", "Social security number"),
    ("reporting", "Financial reporting experience", "Reporting directly to the manager"),
    ("excel", "Microsoft Excel proficiency", "You will excel in this role"),
])
def test_fixed_skill_senses(skill, positive, negative):
    vocab = pd.DataFrame([{"skill": skill, "forms": [skill],
                           "pattern": pp.SKILL_PATTERNS.get(skill, pp.skill_pattern([skill]))}])
    frame = pd.DataFrame({"description": [positive, negative], "skills_desc": ["", ""]})
    assert pp.match_description_skills(frame, vocab).tolist() == [[skill], []]


def test_nested_phrase_does_not_imply_short_skill():
    vocab = pd.DataFrame([{"skill": s, "forms": [s], "pattern": pp.skill_pattern([s])}
                          for s in ["project management", "management"]])
    frame = pd.DataFrame({"description": ["project management", "project management and staff management"],
                          "skills_desc": ["", ""]})
    assert pp.match_description_skills(frame, vocab).tolist() == [
        ["project management"], ["project management", "management"]]
