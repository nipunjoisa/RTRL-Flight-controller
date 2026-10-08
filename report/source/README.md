# Report source

Generator for `../RTRL_Flight_Controller_Report.docx` / `.pdf` (VTU mini-project
report, BCA701). Numbers in the report come from `../tables/main_results.md`,
which is itself generated from `data/results/sweep_results.parquet` — if the
sweep is re-run, update the tables here to match rather than hand-editing the
`.docx`.

## Rebuild

```bash
npm install docx          # only dependency
node build.js             # -> ../RTRL_Flight_Controller_Report.docx
```

`make_arch.py` regenerates `assets/arch.png` (Figure 1, the pipeline diagram);
it needs only matplotlib, so the project venv works:

```bash
.venv/Scripts/python.exe make_arch.py
```

## PDF export

`ExportAsFixedFormat` hangs in this environment even on an empty document, so
`printpdf.ps1` drives the "Microsoft Print to PDF" queue instead:

```powershell
.\printpdf.ps1
```

Opening the `.docx` in Word and using File → Save as PDF works too.

## Page numbers

The Table of Contents, List of Figures and List of Tables carry literal page
numbers (no Word TOC field). `probe.ps1` reads the real page of every heading
and caption back out of Word's own pagination — run it after any change to the
body, then update the `TOC` / `FIGURES` / `TABLES` arrays at the top of
`build.js` to match.

```powershell
.\probe.ps1
```
