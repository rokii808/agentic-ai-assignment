# GenAI use and critical-thinking survey analysis

This repository contains a reproducible, read-only analysis of `Survey_Data_GenAI_Critical_Thinking_Bangladesh.xlsx`. It writes derived data, statistical tables, figures, and a decision-focused report to `outputs/`; it never modifies the source workbook.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe analysis.py
.\.venv\Scripts\python.exe -m pytest -q
```

Start with `outputs/report.md`. The script validates required columns and response ranges, reconstructs composite scales from the documented items, reports reliability, estimates descriptive and adjusted associations, and creates PNG figures.

The analysis is intentionally observational: it reports associations in self-reported cross-sectional data and does not claim that GenAI use causes a change in critical thinking.
