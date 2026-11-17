# Battery triage: second life, recycling or retest?

Code, data extract, results and figures for the article

> **Second Life, Recycling or Retest? Uncertainty-Aware Machine-Learning Triage of End-of-Life Lithium-Ion Batteries from Voltage Relaxation**
> Dounia Tebr, Imane Satauri, Zeineb El Hammoumi, Youssef Charkaoui
> Laboratory of Computer Science, Innovation, and Artificial Intelligence (LI3A), Faculty of Sciences Dhar El Mahraz (FSDM), Sidi Mohamed Ben Abdellah University (USMBA), Fez, Morocco
> Manuscript submitted for publication, 2026.

A recycler must send each used lithium-ion battery to a second life or to recycling, often without the state of health (SoH) that the EU battery passport will hold. This repository estimates SoH from the voltage rest after a full charge, calibrates the model's uncertainty, and routes each battery to **second life**, **recycling** or a **full retest** by expected cost. Every number and figure in the article can be regenerated from the files here.

## What is in this repository

| Folder or file | Contents |
|---|---|
| `extraction_tool/` | `battery_extract_tool.html`: a page that runs in any web browser and reduces the 6 GB of raw Zhu et al. CSV files to the 4 MB extract used here (nothing is uploaded; the files are read locally). `core.js` and `page_template.html` are its sources. |
| `extract.py` | The same extraction in Python, used to check the browser tool. |
| `data/` | `zhu_relaxation_extract.csv.gz`: one row per cycle with the rest curve after the charge and the discharge capacity (derived from Zhu et al., CC BY 4.0). See `data/README.md` for the sources and for the retired-cell files. |
| `*.py` (top level) | The analysis, one script per step (run order below). The `check_step*.py` scripts recompute the reported numbers with separately written code. |
| `results/` | The result files behind every table and figure of the article, plus the run logs. |
| `figures/` | Figures 2 to 6 of the article (PNG 300 dpi, PDF, SVG). |
| `demo/` | `battery_triage_bench.html`: the Battery Triage Bench, a browser demo of the calibrated Gaussian process and the decision rule. Open it in any browser; it needs no installation. |

## Requirements

Python 3.11 or newer (tested with 3.13). Install the packages with

```
pip install -r requirements.txt
```

`jax`, `flax` and `optax` are needed only for the neural-network comparison (`cnn.py`, `compare_step4.py`, `check_step4.py`); `openpyxl` only for the retired-cell step (`salinas.py`).

## Reproducing the results

Run every script from the repository's top folder. Intermediate files (`data/*.pkl`) and new result files are written to `data/`; the files in `results/` are the ones reported in the article, for comparison.

| Step | Command | Output | Article |
|---|---|---|---|
| 0 | (optional) `extraction_tool/battery_extract_tool.html` on the raw data | `data/zhu_relaxation_extract.csv.gz` (already included) | Section 3 |
| 1 | `python load.py` | `data/clean.pkl` (58,806 cycles from 130 cells) | Section 3, Table 1 |
| 2 | `python baselines.py` then `python check_step2.py` | `results_step2.json` | Section 5.1, Table 2, Figure 2 |
| 3 | `python calibrate.py`, `python decide.py`, `python check_step3.py` | `results_step3_calibration.json`, `results_step3_grid.csv` | Sections 5.1 and 5.2, Table 3, Figure 3 |
| 4 | `python cnn.py 1000`, `python compare_step4.py`, `python check_step4.py` | `results_step4.json`, `results_step4_grid.csv` | Section 5.1, Table 2 |
| 5a | `python robust.py time` | `results_step5_time.csv` | Section 5.4, Figure 5 |
| 5b | `python robust.py loco`, then `python check_step5.py` | `results_step5_loco.csv` | Section 5.5, Figure 6 |
| 5c | `python salinas.py` (needs the retired-cell files, see `data/README.md`) | `results_step5_salinas.csv`, `results_step5_salinas_clf.csv` | Section 5.6, Table 5 |
| 5d | `python coldstart.py` | `results_coldstart.csv` | Section 5.5 (cold start: cells or measurements) |
| 6 | `python case_study.py` | `results_case_study.json` | Sections 4.7 and 5.3, Table 4, Figure 4 |
| 7 | `python figures.py` | `figures/` | Figures 2 to 6 |
| 8 | `python export_demo.py` | `demo/model.json` (then rebuild the page from `demo/template.html`) | Section 4.9 |

`cnn.py pilot` repeats the choice of the number of training steps. All random seeds are fixed. Steps 2 to 5 take from a few minutes to about an hour each on a laptop CPU.

## Data sources

- Zhu, J. et al. Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation [dataset]. Zenodo, version 2, 2022. https://doi.org/10.5281/zenodo.6405084 (CC BY 4.0). Article: *Nature Communications* 13, 2261 (2022), https://doi.org/10.1038/s41467-022-29837-w
- Salinas, F. Data for: A second life for Li-ion cells rescued from notebook batteries [dataset]. Mendeley Data, version 4, 2019. https://doi.org/10.17632/kg75m9ydfz.4 (CC BY 4.0)

## Licence

- Code (`*.py`, `extraction_tool/`, `demo/`): MIT licence, see `LICENSE`.
- Data extract, result files and figures: CC BY 4.0. The extract is derived from the Zhu et al. dataset (CC BY 4.0); please cite that dataset as well when you use it.

## How to cite

Please cite the article above (the reference will be updated on publication) and this repository (see `CITATION.cff`).

## Contact

Dounia Tebr, dounia.tebr@usmba.ac.ma
