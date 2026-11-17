# Data

## `zhu_relaxation_extract.csv.gz` (included)

One row per cycle of the 130 cells of Zhu et al. (2022), with these columns:

| Column | Meaning |
|---|---|
| `dataset` | Dataset_1 (NCA), Dataset_2 (NCM) or Dataset_3 (NCM+NCA) |
| `cell` | cell name, e.g. `CY25-05_1-#1` (temperature, charge rate, discharge rate, cell number) |
| `cycle` | cycle number |
| `cap_mAh` | discharge capacity of the cycle (the label) |
| `qc_mAh` | charge capacity of the cycle (never used as a model input) |
| `v_dis_end` | voltage at the end of the discharge (used to drop incomplete discharges) |
| `n_rest` | number of points in the rest curve |
| `rest_t`, `rest_v` | the rest after the full charge: times (s) and voltages (V), separated by `;` |

It was made from the raw CSV files with `extraction_tool/battery_extract_tool.html` and matches `extract.py` exactly on the cells we compared.

Source: Zhu, J. et al. Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation [dataset]. Zenodo, version 2, 2022. https://doi.org/10.5281/zenodo.6405084. Licence: CC BY 4.0. The extract keeps that licence; please cite the dataset when you use it.

To rebuild it yourself: download the dataset from Zenodo, unzip it, open `extraction_tool/battery_extract_tool.html` in a browser, choose the unzipped folder and save the result into this folder.

## Retired notebook cells (not included; needed only for `salinas.py`)

Download the two Excel files `1034 cells.xlsx` and `152 batteries.xlsx` from

Salinas, F. Data for: A second life for Li-ion cells rescued from notebook batteries [dataset]. Mendeley Data, version 4, 2019. https://doi.org/10.17632/kg75m9ydfz.4 (CC BY 4.0)

and put them in a folder `data/salinas/`.

## Files created when you run the scripts

`clean.pkl` and `preds_step*.pkl` (intermediate files) and the `results_*` files are written here. They are not part of the repository; the results reported in the article are in `../results/`.
