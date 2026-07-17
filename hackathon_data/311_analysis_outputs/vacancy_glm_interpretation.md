# GLM Interpretation (CategoryName-Only Model)

- **Model**: GLM was skipped -- no non-deterministic `CategoryName` levels remained after the leakage/separation guard.
- **Deterministic categories dropped**: 138.
- **Non-deterministic categories kept**: 0.

## Practical Read
- `is_high_likelihood` is derived directly from `CategoryName` (Board-up/Abandoned categories), so nearly every `CategoryName` level perfectly separates the outcome.
- A GLM using `CategoryName` as the sole predictor is not identifiable under these conditions; no coefficients are reported.