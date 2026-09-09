import pandas as pd

from analysis import CT_ITEMS, cronbach_alpha, effect_size_epsilon_squared, validate_and_score


def test_cronbach_alpha_is_one_for_perfectly_parallel_items():
    values = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [1, 2, 3, 4]})
    assert cronbach_alpha(values) == 1.0


def test_epsilon_squared_is_non_negative():
    assert effect_size_epsilon_squared([pd.Series([1, 2]), pd.Series([2, 3])], 0.1) == 0.0


def test_scoring_reconstructs_ct_mean_and_flags_missing_identifier():
    base = {"Year": [1], "Gender": ["Female"], "Department": ["CSE"], "Usage_Freq": ["Daily"], "CGPA": [3.5]}
    for item in CT_ITEMS:
        base[item] = [4]
    for item in ["RelyOnAI_Answers", "ReducesCritical", "LimitsSkills", "MemoryRetentionDrop", "HelpsLearnFast", "HelpsStudyEfficiency", "PROD1_TimeSaved", "PROD2_TaskCompletion", "PROD3_ConceptClarity"]:
        base[item] = [3]
    scored, issues, _ = validate_and_score(pd.DataFrame(base))
    assert scored.loc[0, "ct_recomputed"] == 4
    assert scored.loc[0, "usage_ordinal"] == 4
    assert any("StudentID" in issue for issue in issues)
