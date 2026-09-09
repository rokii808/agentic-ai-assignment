# Generative AI use and students’ critical thinking

## Decision brief

This analysis uses 400 student survey responses from the workbook's `Data` sheet. Critical thinking is a **self-reported** 1–5 mean of four items. The headline result is association, not impact: academic GenAI-use frequency has a Spearman correlation of -0.32 with critical thinking (p=1.09e-10). The frequency-group comparison is statistically distinguishable (Kruskal–Wallis H=39.73, p=1.22e-08; epsilon-squared=0.093).

The reported reliance/negative-experience scale has the strongest observed association with critical thinking (Spearman rho=-0.53, p=9.47e-30). This scale includes statements such as “GenAI use reduces my critical thinking,” so it captures perceptions rather than independently observed effects. Its association is useful for identifying students to understand better, but it does not show that reliance caused a decline.

## What leaders should take from this

- Do not interpret more frequent use as proof that GenAI lowers critical thinking. The sample shows a meaningful negative association, but it cannot tell whether use precedes lower scores, lower-scoring students use GenAI more, or other factors explain both.
- Consider piloting guidance that promotes independent problem-solving, source evaluation, and transparent AI-assisted workflows, then evaluate it with objective or longitudinal measures. Students reporting stronger reliance/negative experiences are the group most consistently associated with lower self-reported critical-thinking scores; the survey does not establish that guidance will change those scores.
- Keep the benefits in view: perceived learning/productivity benefits and reported reliance/negative experiences are separate constructs. These responses alone do not justify a blanket restriction policy.
- The noteworthy exception is the smallest group: only 14 students report rare use, so its higher mean has a wide interval. Daily users report both the highest perceived benefits (3.45) and the highest reported reliance/negative-experience score (4.20), alongside the lowest CT mean; this is a co-occurrence, not a causal sequence.

## Critical-thinking score by use frequency

| Use frequency | n | Mean CT | Bootstrap 95% CI |
|---|---:|---:|---:|
| Rarely | 14 | 3.64 | 3.21–4.02 |
| Sometimes | 88 | 3.55 | 3.40–3.71 |
| Often | 183 | 3.22 | 3.10–3.33 |
| Daily | 115 | 2.84 | 2.71–2.98 |

## Adjusted association model

Ordinary least squares with HC3 robust standard errors: critical-thinking score regressed on ordinal use frequency (Rarely=1 … Daily=4), reported reliance/negative-experience and benefit scores, CGPA, year, gender, and department. The coefficients are adjusted associations, not estimated effects: measured adjustment cannot remove unmeasured confounding, reverse causation, or common-method bias.

| Predictor | Estimated association | 95% CI | p-value |
|---|---:|---:|---:|
| usage_ordinal | -0.106 | -0.210 to -0.002 | 0.0456 |
| dependency_recomputed | -0.381 | -0.474 to -0.288 | 8.98e-16 |
| benefit_recomputed | 0.025 | -0.066 to 0.116 | 0.591 |
| CGPA | 0.135 | -0.116 to 0.386 | 0.292 |

Model sample n=400; R²=0.309. Full coefficients are in `outputs/adjusted_model.csv`.

## Data quality and analytical limits

The `Data` sheet has 0 missing cell(s) and 0 fully duplicated row(s).

- Codebook documents StudentID, but the Data sheet does not contain it; respondent-level duplicate checking is impossible.
- Codebook calls CT1 reverse-worded, while its documented composite and the supplied values use an unreversed arithmetic mean; confirm the original questionnaire anchors before using the score beyond this workbook.

- The design is cross-sectional, convenience/selection processes are not described, and all key constructs are self-reported. Reverse causation and unmeasured factors (prior ability, assignment design, socioeconomic context, digital access, instructor policy) remain plausible.
- Likert means are treated as approximately continuous for the descriptive and regression summaries; the non-parametric frequency test is included as a distribution-free complement.
- The workbook's supplied `Cluster` categories are not used for inference because their derivation is not reproducible from the workbook. The analysis independently reconstructs the documented composite scores from items.
- Results describe this sample of students in Bangladesh and should not be generalized to other populations without replication.

## Reproduction

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe analysis.py
.\.venv\Scripts\python.exe -m pytest -q
```

The pipeline reads the original workbook without writing to it. Input checksum (SHA-256): `ac83024416d69ed486ee6627410ef09e68b0850d25d505f8a93a7e40a364eb2c`.
