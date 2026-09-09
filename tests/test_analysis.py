import json

import pandas as pd
import pytest

from analysis import (
    CT_ITEMS,
    SOURCE,
    cronbach_alpha,
    effect_size_epsilon_squared,
    run_analysis,
    validate_and_score,
    workbook_sha256,
)


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


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Usage_Freq", "Weekly", "Usage_Freq"),
        ("Usage_Freq", None, "Usage_Freq"),
        ("CT1_IndependentAnalysis", 6, "CT1_IndependentAnalysis"),
        ("CGPA", "not-a-number", "CGPA"),
    ],
)
def test_invalid_analytic_input_fails_before_scoring(column, value, message):
    base = {"Year": [1], "Gender": ["Female"], "Department": ["CSE"], "Usage_Freq": ["Daily"], "CGPA": [3.5]}
    for item in CT_ITEMS:
        base[item] = [4]
    for item in ["RelyOnAI_Answers", "ReducesCritical", "LimitsSkills", "MemoryRetentionDrop", "HelpsLearnFast", "HelpsStudyEfficiency", "PROD1_TimeSaved", "PROD2_TaskCompletion", "PROD3_ConceptClarity"]:
        base[item] = [3]
    base[column] = [value]

    with pytest.raises(ValueError, match=message):
        validate_and_score(pd.DataFrame(base))


def test_end_to_end_run_writes_expected_outputs_and_preserves_source(tmp_path):
    source_hash_before = workbook_sha256(SOURCE)
    output_dir = tmp_path / "analysis_outputs"

    run_analysis(SOURCE, output_dir)

    expected_files = {
        "adjusted_model.csv",
        "analysis_metadata.json",
        "correlations_with_critical_thinking.csv",
        "critical_thinking_by_usage_frequency.csv",
        "data_quality_profile.csv",
        "frequency_anova_sensitivity.csv",
        "report.md",
        "scale_reliability.csv",
        "scored_data.csv",
        "figures/constructs_by_usage_frequency.png",
        "figures/critical_thinking_by_usage_frequency.png",
        "figures/critical_thinking_vs_dependency.png",
    }
    assert all((output_dir / path).is_file() for path in expected_files)
    assert workbook_sha256(SOURCE) == source_hash_before

    metadata = json.loads((output_dir / "analysis_metadata.json").read_text(encoding="utf-8"))
    frequency = pd.read_csv(output_dir / "critical_thinking_by_usage_frequency.csv")
    assert metadata["source_sha256"] == source_hash_before
    assert metadata["n_responses"] == 400
    assert frequency["Usage_Freq"].tolist() == ["Rarely", "Sometimes", "Often", "Daily"]
    assert frequency["n"].tolist() == [14, 88, 183, 115]
    assert frequency["ct_mean"].round(2).tolist() == [3.64, 3.55, 3.22, 2.84]
    assert "association, not impact" in (output_dir / "report.md").read_text(encoding="utf-8")
