# Statistical methods

## Symbols and direction

For every paired comparison, the difference is defined as:

```text
d_i = BIMODriver_i - baseline_i
```

For the ablation analysis, the left-hand configuration replaces BIMODriver in
this definition and the reference is Original Transductive.

Positive mean/HL differences and positive `r_rb` favor the left-hand method.
Negative values favor the right-hand method.

## Wilcoxon signed-rank test

All analyses are paired. The signed-rank statistic ranks the absolute nonzero
paired differences and then uses their signs.

- For 10 paired values and 15 cancer-type means, the code enumerates the exact
  sign-flip distribution. This supports tied absolute ranks without depending
  on a SciPy version-specific `auto` decision.
- For the direct 50-fold sensitivity analysis, the code uses the asymptotic
  Wilcoxon approximation.
- Clean/Hit and ablation analyses are two-sided.
- The original-paper comparison tables also retain the pre-specified one-sided
  alternative `BIMODriver > baseline`.

## Paired rank-biserial effect size

`r_rb` is the paired rank-biserial correlation:

```text
r_rb = (W_positive - W_negative) / (W_positive + W_negative)
```

It ranges from -1 to 1. Its sign gives the direction of the paired differences;
its absolute magnitude reflects their rank consistency. A value near 1 means
nearly all ranked differences favor the left method, and a value near -1 means
nearly all favor the right method.

## Hodges-Lehmann difference and confidence interval

The paired Hodges-Lehmann estimate is the median of the Walsh averages of the
paired differences. It estimates the typical location shift, using the same
left-minus-right direction as above.

The reported 95% interval is a percentile bootstrap interval for this HL
estimate. Therefore a reported `HL difference = 0.0055` with a 95% interval
`[0.0041, 0.0067]` means that the estimated typical paired improvement is
0.0055 and its bootstrap uncertainty interval is 0.0041 to 0.0067.

## Benjamini-Hochberg families

The BH correction family must be named because a q-value depends on all p-values
included in that family.

| Analysis | Pairing unit | Family size |
|---|---:|---:|
| Clean/Hit fixed test | run | 8 |
| Four-way ablation primary | five-fold run mean | 6 |
| Four-way ablation supplement | fold value | 6 |
| CPDB paper comparison | five-fold run mean | 18 within scope |
| STRING paper comparison | five-fold run mean | 18 within scope |
| 15 cancer-type means | cancer-type mean | 18 within scope |
| Per-cancer run-mean tests | five-fold run mean | 135 per metric / 270 across both metrics when reported globally |

The paper reanalysis exports alternative global and within-scope q-values where
needed. The column name identifies the correction family.

## Why 50-fold SD and 10-run-mean SD differ

The 50-fold SD describes variation among all individual fold scores. The
10-run-mean SD first averages each row's five folds, so fold-level variation is
smoothed before the SD is calculated. The latter is therefore typically much
smaller. These quantities answer different questions and should not be placed
under the same label.
