# Replication of Fraser & Frazer (2020): GP Prescribing and Deprivation in Northern Ireland

This repository contains a replication of the analysis from:

> Fraser L, Frazer K. *Analysis of general practice prescribing in Northern Ireland using a measure of deprivation.* J Public Health. 2020;42(4):722-727.

The original study used Kendall's tau to correlate ward-level deprivation (NIMDM 2017) with GP prescribing rates across 12 prespecified BNF sections in Northern Ireland. This replication uses **Q4 2025 (October-December) prescribing data** to assess whether the same deprivation-prescribing gradients persist six years later.

## Key Findings

- **9 of 12** prespecified BNF sections showed statistically significant negative correlations with deprivation at the Bonferroni-corrected threshold (p < 0.0005), compared to **8 of 12** in the original study.
- All 12 prespecified sections showed **negative** Kendall's tau values, confirming that more deprived areas consistently have higher prescribing rates.
- Correlation magnitudes were **generally attenuated** compared to 2019, suggesting either narrowing of deprivation gaps or methodological differences.
- Across all 168 BNF sections analysed, **27 showed significant** correlations with deprivation.

## Limitations

- **Practice postcode as proxy for patient deprivation**: Deprivation is assigned based on the GP practice postcode, not where registered patients actually live. This ecological measure likely attenuates the true relationship between individual-level deprivation and prescribing.
- Ward-level aggregation may mask within-ward variation.
- Items dispensed (not DDD or patient counts) are used as the prescribing measure.

## Files

| File | Description |
|------|-------------|
| `analysis.py` | Full analysis script (Python) |
| `fraser_replication_report_v2.html` | Complete HTML report with embedded figures and comparison with original study |
| `correlations_bnf_sections.csv` | Kendall's tau results for all 168 BNF sections |
| `correlations_drugs.csv` | Kendall's tau results for 28 individual drugs |
| `ward_totals.csv` | Ward-level prescribing totals and deprivation ranks |
| `summary_statistics.csv` | Overall summary statistics |
| `postcode_geography.csv` | Practice postcode to ward mapping (from NISRA Central Postcode Directory) |
| `figure1-5_*.png` | Individual figure files |

## Data Sources

- **Prescribing data**: [BSO OpenData NI GP Prescribing](https://www.opendatani.gov.uk/dataset/gp-prescribing-data) (Q4 2025, ~184MB total, not included in repo)
- **Practice reference file**: [BSO GP Practice List](https://www.opendatani.gov.uk/dataset/gp-prescribing-data) (January 2026)
- **Deprivation data**: [NIMDM 2017](https://www.nisra.gov.uk/statistics/deprivation/northern-ireland-multiple-deprivation-measure-2017-nimdm2017) (Small Area level)
- **Geography**: [NISRA Central Postcode Directory](https://www.nisra.gov.uk/support/geography/central-postcode-directory)

## Requirements

```
pandas
scipy
matplotlib
numpy
xlrd
```

## Running the Analysis

1. Download the Q4 2025 prescribing CSVs and the NIMDM 2017 Excel file into the working directory.
2. Ensure `postcode_geography.csv` is present (maps practice postcodes to wards).
3. Run `python analysis.py`.

## Licence

This analysis is shared for academic and educational purposes.
