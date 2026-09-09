"""Reproducible analysis of GenAI use and self-reported critical thinking.

The source workbook is read only. Run from the repository root:
    .\\.venv\\Scripts\\python.exe analysis.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keep Matplotlib's cache local to the project; some managed environments deny
# writes to the user-profile cache directory.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kruskal, spearmanr
from statsmodels.stats.anova import anova_lm


SOURCE = ROOT / "Survey_Data_GenAI_Critical_Thinking_Bangladesh.xlsx"
OUT = ROOT / "outputs"

CT_ITEMS = ["CT1_IndependentAnalysis", "CT2_ProblemSolving", "CT3_EvaluateSources", "CT4_ArgueReasoning"]
DEPENDENCY_ITEMS = ["RelyOnAI_Answers", "ReducesCritical", "LimitsSkills", "MemoryRetentionDrop"]
BENEFIT_ITEMS = ["HelpsLearnFast", "HelpsStudyEfficiency", "PROD1_TimeSaved", "PROD2_TaskCompletion", "PROD3_ConceptClarity"]
LIKERT_ITEMS = CT_ITEMS + DEPENDENCY_ITEMS + BENEFIT_ITEMS
USAGE_ORDER = ["Rarely", "Sometimes", "Often", "Daily"]


def cronbach_alpha(frame: pd.DataFrame) -> float:
    """Return alpha for complete item responses; NaN if it cannot be calculated."""
    k = frame.shape[1]
    if k < 2 or len(frame) < 2:
        return float("nan")
    item_variances = frame.var(axis=0, ddof=1).sum()
    total_variance = frame.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_variances / total_variance)) if total_variance else float("nan")


def effect_size_epsilon_squared(groups: list[pd.Series], h_stat: float) -> float:
    """Kruskal-Wallis epsilon-squared; bounded below by zero for readability."""
    n = sum(len(group) for group in groups)
    k = len(groups)
    return max(0.0, float((h_stat - k + 1) / (n - k))) if n > k else float("nan")


def bootstrap_mean_ci(values: pd.Series, reps: int = 3000, seed: int = 20260908) -> tuple[float, float]:
    values = values.dropna().to_numpy()
    if len(values) < 2:
        raise ValueError("At least two complete responses are required to calculate a bootstrap confidence interval.")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(reps, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def workbook_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_score(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    required = ["Year", "Gender", "Department", "Usage_Freq", "CGPA", *LIKERT_ITEMS]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Data sheet is missing required fields: {missing}")

missing_values = [column for column in required if data[column].isna().any()]
    if missing_values:
        raise ValueError(f"Analytic fields contain missing value(s): {missing_values}")

    scored = data.copy()
    numeric_fields = ["Year", "CGPA", *LIKERT_ITEMS]
for column in numeric_fields:
        converted = pd.to_numeric(scored[column], errors="coerce")
        non_numeric = scored[column].notna() & converted.isna()
        if non_numeric.any():
            raise ValueError(f"{column} contains {int(non_numeric.sum())} non-numeric value(s).")
        scored[column] = converted

    invalid_usage = sorted(set(scored["Usage_Freq"]) - set(USAGE_ORDER))
    if invalid_usage:
        raise ValueError(f"Unexpected Usage_Freq value(s): {invalid_usage}.")
    if not scored["Year"].between(1, 4).all():
        raise ValueError("Year contains value(s) outside the documented 1–4 range.")
    for column in LIKERT_ITEMS:
        if not scored[column].between(1, 5).all():
            raise ValueError(f"{column} contains value(s) outside the documented 1–5 Likert range.")
        if not scored["CGPA"].between(0, 4).all():
        raise ValueError("CGPA contains value(s) outside the 0–4 range.")

issues: list[str] = []
    if "StudentID" not in data.columns:
        issues.append("Codebook documents StudentID, but the Data sheet does not contain it; respondent-level duplicate checking is impossible.")
    issues.append("Codebook calls CT1 reverse-worded, while its documented composite and the supplied values use an unreversed arithmetic mean; confirm the original questionnaire anchors before using the score beyond this workbook.")
    if data.empty:
        raise ValueError("Data sheet contains no responses.")
    if data.duplicated().any():
        issues.append(f"{int(data.duplicated().sum())} fully duplicated response row(s) detected.")
    scored["ct_recomputed"] = scored[CT_ITEMS].mean(axis=1)
    scored["dependency_recomputed"] = scored[DEPENDENCY_ITEMS].mean(axis=1)
    scored["benefit_recomputed"] = scored[BENEFIT_ITEMS].mean(axis=1)
    scored["usage_ordinal"] = pd.Categorical(scored["Usage_Freq"], categories=USAGE_ORDER, ordered=True).codes + 1

    supplied = {
        "CT_Composite": "ct_recomputed",
        "Dependency_Composite": "dependency_recomputed",
        "Benefit_Composite": "benefit_recomputed",
    }
    for original, calculated in supplied.items():
        if original not in scored.columns:
            issues.append(f"Workbook does not include supplied {original}; independent reconstruction was used.")
        else:
            difference = (pd.to_numeric(scored[original], errors="coerce") - scored[calculated]).abs()
            if (difference > 1e-8).any():
                issues.append(f"{original} differs from its documented item mean in {int((difference > 1e-8).sum())} row(s).")

    quality = pd.DataFrame(
        {
            "field": scored.columns,
            "missing_n": [int(scored[col].isna().sum()) for col in scored.columns],
            "missing_pct": [round(100 * scored[col].isna().mean(), 2) for col in scored.columns],
            "unique_n": [int(scored[col].nunique(dropna=True)) for col in scored.columns],
        }
    )
    return scored, issues, quality


def make_figures(scored: pd.DataFrame, frequency: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = "Blues"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.stripplot(data=scored, x="Usage_Freq", y="ct_recomputed", order=USAGE_ORDER, color="#91b6d9", alpha=0.34, jitter=0.17, ax=ax)
    means = frequency.set_index("Usage_Freq").reindex(USAGE_ORDER)
    ax.errorbar(range(len(USAGE_ORDER)), means["ct_mean"],
                yerr=[means["ct_mean"] - means["ci_low"], means["ci_high"] - means["ct_mean"]],
                fmt="o-", color="#0b3c6f", capsize=4, linewidth=2, label="Mean and bootstrap 95% CI")
    ax.set(xlabel="Academic GenAI-use frequency", ylabel="Self-reported critical-thinking score (1–5)", ylim=(1, 5), title="Critical thinking by GenAI-use frequency")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "critical_thinking_by_usage_frequency.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=scored, x="dependency_recomputed", y="ct_recomputed", scatter_kws={"alpha": 0.32, "s": 30, "color": "#548ab7"}, line_kws={"color": "#b22222", "linewidth": 2}, ax=ax)
    ax.set(xlabel="Reported reliance / negative-experience scale (1–5)", ylabel="Self-reported critical-thinking score (1–5)", ylim=(1, 5), xlim=(1, 5), title="Reported reliance/negative experiences are associated with lower CT scores")
    fig.tight_layout()
    fig.savefig(figures_dir / "critical_thinking_vs_dependency.png", dpi=220)
    plt.close(fig)

    long = scored.melt(id_vars="Usage_Freq", value_vars=["ct_recomputed", "dependency_recomputed", "benefit_recomputed"], var_name="measure", value_name="score")
    labels = {"ct_recomputed": "Critical thinking", "dependency_recomputed": "Reported reliance / negative experiences", "benefit_recomputed": "Perceived benefits"}
    long["measure"] = long["measure"].map(labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.pointplot(data=long, x="Usage_Freq", y="score", hue="measure", order=USAGE_ORDER, errorbar=("ci", 95), palette=palette, dodge=0.25, ax=ax)
    ax.set(xlabel="Academic GenAI-use frequency", ylabel="Mean score (1–5)", ylim=(1, 5), title="Use frequency, perceived benefits, reliance, and critical thinking")
    ax.legend(title="Measure", loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "constructs_by_usage_frequency.png", dpi=220)
    plt.close(fig)


def build_report(results: dict, issues: list[str]) -> str:
    usage = results["usage"]
    correlations = results["correlations"]
    model = results["model"]
    quality_messages = "\n".join(f"- {issue}" for issue in issues) or "- No validation failures were detected."
    usage_lines = "\n".join(
        f"| {row.Usage_Freq} | {int(row.n)} | {row.ct_mean:.2f} | {row.ci_low:.2f}–{row.ci_high:.2f} |"
        for row in usage.itertuples()
    )
    model_lines = "\n".join(
        f"| {row.term} | {row.estimate:.3f} | {row.ci_low:.3f} to {row.ci_high:.3f} | {row.p_value:.3g} |"
        for row in model.itertuples() if row.term in ["usage_ordinal", "dependency_recomputed", "benefit_recomputed", "CGPA"]
    )
    return f"""# Generative AI use and students’ critical thinking

## Decision brief

This analysis uses {results['n']} student survey responses from the workbook's `Data` sheet. Critical thinking is a **self-reported** 1–5 mean of four items. The headline result is association, not impact: academic GenAI-use frequency has a Spearman correlation of {correlations.loc[correlations.measure == 'Usage frequency', 'rho'].iloc[0]:.2f} with critical thinking (p={correlations.loc[correlations.measure == 'Usage frequency', 'p_value'].iloc[0]:.3g}). The frequency-group comparison is {'statistically distinguishable' if results['usage_test']['p_value'] < .05 else 'not statistically distinguishable'} (Kruskal–Wallis H={results['usage_test']['H']:.2f}, p={results['usage_test']['p_value']:.3g}; epsilon-squared={results['usage_test']['epsilon_squared']:.3f}).

The reported reliance/negative-experience scale has the strongest observed association with critical thinking (Spearman rho={correlations.loc[correlations.measure == 'Reported reliance / negative experiences', 'rho'].iloc[0]:.2f}, p={correlations.loc[correlations.measure == 'Reported reliance / negative experiences', 'p_value'].iloc[0]:.3g}). This scale includes statements such as “GenAI use reduces my critical thinking,” so it captures perceptions rather than independently observed effects. Its association is useful for identifying students to understand better, but it does not show that reliance caused a decline.

## What leaders should take from this

- Do not interpret more frequent use as proof that GenAI lowers critical thinking. The sample shows a {'small' if results['usage_test']['epsilon_squared'] < .06 else 'meaningful'} negative association, but it cannot tell whether use precedes lower scores, lower-scoring students use GenAI more, or other factors explain both.
- Consider piloting guidance that promotes independent problem-solving, source evaluation, and transparent AI-assisted workflows, then evaluate it with objective or longitudinal measures. Students reporting stronger reliance/negative experiences are the group most consistently associated with lower self-reported critical-thinking scores; the survey does not establish that guidance will change those scores.
- Keep the benefits in view: perceived learning/productivity benefits and reported reliance/negative experiences are separate constructs. These responses alone do not justify a blanket restriction policy.
- The noteworthy exception is the smallest group: only {results['rare_n']} students report rare use, so its higher mean has a wide interval. Daily users report both the highest perceived benefits ({results['daily_benefit']:.2f}) and the highest reported reliance/negative-experience score ({results['daily_dependency']:.2f}), alongside the lowest CT mean; this is a co-occurrence, not a causal sequence.

## Critical-thinking score by use frequency

| Use frequency | n | Mean CT | Bootstrap 95% CI |
|---|---:|---:|---:|
{usage_lines}

## Adjusted association model

Ordinary least squares with HC3 robust standard errors: critical-thinking score regressed on ordinal use frequency (Rarely=1 … Daily=4), reported reliance/negative-experience and benefit scores, CGPA, year, gender, and department. The coefficients are adjusted associations, not estimated effects: measured adjustment cannot remove unmeasured confounding, reverse causation, or common-method bias.

| Predictor | Estimated association | 95% CI | p-value |
|---|---:|---:|---:|
{model_lines}

Model sample n={results['model_n']}; R²={results['r_squared']:.3f}. Full coefficients are in `outputs/adjusted_model.csv`.

## Data quality and analytical limits

{results['validation_summary']}

{quality_messages}

- The design is cross-sectional, convenience/selection processes are not described, and all key constructs are self-reported. Reverse causation and unmeasured factors (prior ability, assignment design, socioeconomic context, digital access, instructor policy) remain plausible.
- Likert means are treated as approximately continuous for the descriptive and regression summaries; the non-parametric frequency test is included as a distribution-free complement.
- The workbook's supplied `Cluster` categories are not used for inference because their derivation is not reproducible from the workbook. The analysis independently reconstructs the documented composite scores from items.
- Results describe this sample of students in Bangladesh and should not be generalized to other populations without replication.

## Reproduction

From the repository root on Windows:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-lock.txt
.\\.venv\\Scripts\\python.exe analysis.py
.\\.venv\\Scripts\\python.exe -m pytest -q
```

The pipeline reads the original workbook without writing to it. Input checksum (SHA-256): `{results['sha256']}`.
"""


def run_analysis(source: Path = SOURCE, output_dir: Path = OUT) -> None:
    """Run the analysis, replacing only the selected generated-output directory."""
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source workbook not found: {source}")
    if output_dir == source.parent:
        raise ValueError("Output directory must not be the source workbook directory.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)

    data = pd.read_excel(source, sheet_name="Data")
    scored, issues, quality = validate_and_score(data)
    quality.to_csv(output_dir / "data_quality_profile.csv", index=False)
    scored.to_csv(output_dir / "scored_data.csv", index=False)

    reliability = pd.DataFrame(
        {"scale": ["Critical thinking", "Reported reliance / negative experiences", "Perceived benefits"],
         "items": [len(CT_ITEMS), len(DEPENDENCY_ITEMS), len(BENEFIT_ITEMS)],
         "cronbach_alpha": [cronbach_alpha(scored[CT_ITEMS]), cronbach_alpha(scored[DEPENDENCY_ITEMS]), cronbach_alpha(scored[BENEFIT_ITEMS])]}
    )
    reliability.to_csv(output_dir / "scale_reliability.csv", index=False)

    usage_rows = []
    for category in USAGE_ORDER:
        values = scored.loc[scored["Usage_Freq"] == category, "ct_recomputed"]
        low, high = bootstrap_mean_ci(values)
        usage_rows.append({"Usage_Freq": category, "n": len(values), "ct_mean": values.mean(), "ct_sd": values.std(ddof=1), "ci_low": low, "ci_high": high})
    usage = pd.DataFrame(usage_rows)
    usage.to_csv(output_dir / "critical_thinking_by_usage_frequency.csv", index=False)
    groups = [scored.loc[scored["Usage_Freq"] == group, "ct_recomputed"] for group in USAGE_ORDER]
    h_stat, p_value = kruskal(*groups)
    usage_test = {"H": float(h_stat), "p_value": float(p_value), "epsilon_squared": effect_size_epsilon_squared(groups, h_stat)}

    correlations = []
    for label, column in [("Usage frequency", "usage_ordinal"), ("Reported reliance / negative experiences", "dependency_recomputed"), ("Perceived benefits", "benefit_recomputed"), ("CGPA", "CGPA")]:
        valid = scored[[column, "ct_recomputed"]].dropna()
        rho, p = spearmanr(valid[column], valid["ct_recomputed"])
        correlations.append({"measure": label, "n": len(valid), "rho": rho, "p_value": p})
    correlation_table = pd.DataFrame(correlations)
    correlation_table.to_csv(output_dir / "correlations_with_critical_thinking.csv", index=False)

    model_data = scored.dropna(subset=["ct_recomputed", "usage_ordinal", "dependency_recomputed", "benefit_recomputed", "CGPA", "Year", "Gender", "Department"])
    model = smf.ols("ct_recomputed ~ usage_ordinal + dependency_recomputed + benefit_recomputed + CGPA + C(Year) + C(Gender) + C(Department)", data=model_data).fit(cov_type="HC3")
    ci = model.conf_int()
    model_table = pd.DataFrame({"term": model.params.index, "estimate": model.params.values, "std_error_hc3": model.bse.values, "ci_low": ci[0].values, "ci_high": ci[1].values, "p_value": model.pvalues.values})
    model_table.to_csv(output_dir / "adjusted_model.csv", index=False)
    anova = anova_lm(smf.ols("ct_recomputed ~ C(Usage_Freq)", data=model_data).fit(), typ=2)
    anova.to_csv(output_dir / "frequency_anova_sensitivity.csv")
    make_figures(scored, usage, figures_dir)

    missing_total = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    validation_summary = f"The `Data` sheet has {missing_total} missing cell(s) and {duplicate_rows} fully duplicated row(s)."
    results = {"n": len(scored), "model_n": len(model_data), "sha256": workbook_sha256(source), "usage": usage, "usage_test": usage_test, "correlations": correlation_table, "model": model_table, "r_squared": float(model.rsquared), "validation_summary": validation_summary, "rare_n": int((scored['Usage_Freq'] == 'Rarely').sum()), "daily_benefit": float(scored.loc[scored['Usage_Freq'] == 'Daily', 'benefit_recomputed'].mean()), "daily_dependency": float(scored.loc[scored['Usage_Freq'] == 'Daily', 'dependency_recomputed'].mean())}
    (output_dir / "analysis_metadata.json").write_text(json.dumps({"source_file": source.name, "source_sha256": results["sha256"], "n_responses": len(scored), "validation_issues": issues, "usage_test": usage_test}, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(build_report(results, issues), encoding="utf-8")
    print(f"Analysis complete. Read {len(scored)} rows and wrote {output_dir}.")


def main() -> None:
    parser = ArgumentParser(description="Analyze the GenAI and critical-thinking survey workbook.")
    parser.add_argument("--source", type=Path, default=SOURCE, help="Source workbook to read (default: project workbook).")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Generated-output directory to replace (default: outputs).")
    args = parser.parse_args()
    run_analysis(args.source, args.output_dir)


if __name__ == "__main__":
    main()
