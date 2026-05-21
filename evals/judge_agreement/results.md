# MTBench Judge-Agreement Study: GPT-4 vs GPT-5.4 Models

**Models evaluated:** qwen3-4b-base, qwen3-4b-instruct-2507  
**Total judgment pairs:** 287  
**Judging mode:** single-answer grading (1–10 scale)

## GPT-4 Self-Agreement Baseline

From the two GPT-4 judging runs on qwen3-4b-instruct-2507 (159 duplicate pairs):

| Metric | Value |
|--------|-------|
| Pearson r | N/A [nan, nan] |
| Spearman ρ | N/A [nan, nan] |
| MAE | 7.827 [7.423, 8.197] |
| Exact match | 0.03 |
| Within ±1 pt | 0.08 |
| Quadratic-weighted κ | 0.000 |

## Overall Agreement Metrics

### gpt-5.4-mini-2026-03-17

| Metric | Value |
|--------|-------|
| n | 287 |
| Pearson r | 0.761 [0.693, 0.820] |
| Spearman ρ | 0.678 [0.598, 0.749] |
| MAE | 1.770 [1.573, 1.960] |
| MSE | 5.929 |
| Bias (new − GPT-4) | -1.003 |
| Exact match | 0.26 |
| Within ±1 pt | 0.51 |
| Quadratic-weighted κ | 0.708 |

### gpt-5.4-nano-2026-03-17

| Metric | Value |
|--------|-------|
| n | 287 |
| Pearson r | 0.760 [0.697, 0.817] |
| Spearman ρ | 0.701 [0.626, 0.766] |
| MAE | 1.854 [1.629, 2.058] |
| MSE | 6.960 |
| Bias (new − GPT-4) | -1.359 |
| Exact match | 0.29 |
| Within ±1 pt | 0.52 |
| Quadratic-weighted κ | 0.696 |

## Aggregate Scores by Judge

|                                            |   overall |   turn1 |   turn2 |
|:-------------------------------------------|----------:|--------:|--------:|
| ('qwen3-4b-base', 'gpt-4')                 |   6.35517 | 7.25625 | 5.24615 |
| ('qwen3-4b-base', 'gpt-5.4-mini')          |   5.58621 | 6.5     | 4.46154 |
| ('qwen3-4b-base', 'gpt-5.4-nano')          |   5.23448 | 6.175   | 4.07692 |
| ('qwen3-4b-instruct-2507', 'gpt-4')        |   8.71479 | 9.04375 | 8.29032 |
| ('qwen3-4b-instruct-2507', 'gpt-5.4-mini') |   7.47183 | 8       | 6.79032 |
| ('qwen3-4b-instruct-2507', 'gpt-5.4-nano') |   7.11268 | 7.7375  | 6.30645 |

## By Turn

| Stratum | Judge | n | Pearson r | Spearman ρ | MAE | Bias | Exact | Within±1 | κ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | mini | 160 | 0.765 | 0.609 | 1.600 | -0.900 | 0.29 | 0.51 | 0.716 |
| 1 | nano | 160 | 0.748 | 0.622 | 1.694 | -1.194 | 0.32 | 0.51 | 0.688 |
| 2 | mini | 127 | 0.735 | 0.703 | 1.984 | -1.134 | 0.22 | 0.51 | 0.666 |
| 2 | nano | 127 | 0.749 | 0.747 | 2.055 | -1.567 | 0.25 | 0.54 | 0.669 |

## By Category

| Stratum | Judge | n | Pearson r | Spearman ρ | MAE | Bias | Exact | Within±1 | κ |
|---|---|---|---|---|---|---|---|---|---|
| coding | mini | 40 | 0.894 | 0.880 | 1.250 | -0.350 | 0.30 | 0.70 | 0.863 |
| coding | nano | 40 | 0.852 | 0.841 | 1.250 | -1.150 | 0.50 | 0.78 | 0.803 |
| extraction | mini | 37 | 0.595 | 0.534 | 2.027 | -0.676 | 0.19 | 0.54 | 0.577 |
| extraction | nano | 37 | 0.721 | 0.663 | 1.811 | -1.162 | 0.24 | 0.54 | 0.674 |
| humanities | mini | 30 | 0.872 | 0.641 | 2.250 | -2.250 | 0.17 | 0.27 | 0.623 |
| humanities | nano | 30 | 0.794 | 0.776 | 2.683 | -2.617 | 0.13 | 0.20 | 0.533 |
| math | mini | 40 | 0.822 | 0.879 | 1.050 | 0.850 | 0.65 | 0.85 | 0.772 |
| math | nano | 40 | 0.817 | 0.880 | 0.925 | 0.525 | 0.72 | 0.85 | 0.801 |
| reasoning | mini | 40 | 0.866 | 0.824 | 1.350 | -0.500 | 0.35 | 0.62 | 0.845 |
| reasoning | nano | 40 | 0.835 | 0.784 | 1.450 | -0.650 | 0.33 | 0.68 | 0.819 |
| roleplay | mini | 29 | 0.859 | 0.725 | 1.914 | -1.638 | 0.14 | 0.41 | 0.695 |
| roleplay | nano | 29 | 0.797 | 0.629 | 2.155 | -1.879 | 0.14 | 0.38 | 0.635 |
| stem | mini | 37 | 0.697 | 0.576 | 2.486 | -2.216 | 0.08 | 0.27 | 0.502 |
| stem | nano | 37 | 0.755 | 0.714 | 2.486 | -2.378 | 0.08 | 0.30 | 0.542 |
| writing | mini | 34 | 0.738 | 0.648 | 2.118 | -1.941 | 0.09 | 0.29 | 0.488 |
| writing | nano | 34 | 0.603 | 0.570 | 2.500 | -2.206 | 0.03 | 0.29 | 0.395 |

## By Judge Template

| Stratum | Judge | n | Pearson r | Spearman ρ | MAE | Bias | Exact | Within±1 | κ |
|---|---|---|---|---|---|---|---|---|---|
| single-math-v1 | mini | 60 | 0.861 | 0.841 | 1.067 | 0.100 | 0.55 | 0.75 | 0.852 |
| single-math-v1 | nano | 60 | 0.810 | 0.848 | 1.117 | -0.217 | 0.65 | 0.77 | 0.807 |
| single-math-v1-multi-turn | mini | 60 | 0.825 | 0.808 | 1.367 | -0.100 | 0.32 | 0.70 | 0.805 |
| single-math-v1-multi-turn | nano | 60 | 0.832 | 0.802 | 1.300 | -0.633 | 0.38 | 0.77 | 0.816 |
| single-v1 | mini | 100 | 0.681 | 0.496 | 1.920 | -1.500 | 0.13 | 0.37 | 0.545 |
| single-v1 | nano | 100 | 0.710 | 0.543 | 2.040 | -1.780 | 0.12 | 0.36 | 0.541 |
| single-v1-multi-turn | mini | 67 | 0.711 | 0.653 | 2.537 | -2.060 | 0.13 | 0.34 | 0.547 |
| single-v1-multi-turn | nano | 67 | 0.711 | 0.743 | 2.731 | -2.403 | 0.13 | 0.33 | 0.539 |

## By Evaluated Model

| Stratum | Judge | n | Pearson r | Spearman ρ | MAE | Bias | Exact | Within±1 | κ |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-4b-base | mini | 145 | 0.757 | 0.738 | 1.831 | -0.769 | 0.29 | 0.52 | 0.717 |
| qwen3-4b-base | nano | 145 | 0.784 | 0.770 | 1.769 | -1.121 | 0.36 | 0.54 | 0.735 |
| qwen3-4b-instruct-2507 | mini | 142 | 0.673 | 0.497 | 1.708 | -1.243 | 0.23 | 0.50 | 0.586 |
| qwen3-4b-instruct-2507 | nano | 142 | 0.646 | 0.544 | 1.940 | -1.602 | 0.22 | 0.51 | 0.541 |

## Top 10 Largest Disagreements

### gpt-5.4-mini-2026-03-17

| model                  |   question_id |   turn | category   |   gpt4_score |   mini_score |   signed_diff |
|:-----------------------|--------------:|-------:|:-----------|-------------:|-------------:|--------------:|
| qwen3-4b-base          |           134 |      1 | extraction |            1 |            9 |             8 |
| qwen3-4b-base          |           136 |      2 | extraction |           10 |            2 |            -8 |
| qwen3-4b-instruct-2507 |           111 |      2 | math       |            1 |            9 |             8 |
| qwen3-4b-base          |           111 |      1 | math       |            2 |            9 |             7 |
| qwen3-4b-instruct-2507 |           144 |      2 | stem       |           10 |            3 |            -7 |
| qwen3-4b-base          |           140 |      2 | extraction |            9 |            3 |            -6 |
| qwen3-4b-base          |           143 |      2 | stem       |            9 |            3 |            -6 |
| qwen3-4b-base          |           150 |      2 | stem       |           10 |            4 |            -6 |
| qwen3-4b-instruct-2507 |           136 |      2 | extraction |           10 |            4 |            -6 |
| qwen3-4b-instruct-2507 |           156 |      2 | humanities |           10 |            4 |            -6 |

### gpt-5.4-nano-2026-03-17

| model                  |   question_id |   turn | category   |   gpt4_score |   nano_score |   signed_diff |
|:-----------------------|--------------:|-------:|:-----------|-------------:|-------------:|--------------:|
| qwen3-4b-base          |           136 |      2 | extraction |           10 |            2 |            -8 |
| qwen3-4b-instruct-2507 |           111 |      1 | math       |            1 |            9 |             8 |
| qwen3-4b-base          |           111 |      1 | math       |            2 |            9 |             7 |
| qwen3-4b-base          |           140 |      2 | extraction |            9 |            2 |            -7 |
| qwen3-4b-base          |           143 |      2 | stem       |            9 |            2 |            -7 |
| qwen3-4b-base          |           126 |      2 | coding     |            9 |            2 |            -7 |
| qwen3-4b-instruct-2507 |           144 |      2 | stem       |           10 |            3 |            -7 |
| qwen3-4b-instruct-2507 |           156 |      2 | humanities |           10 |            3 |            -7 |
| qwen3-4b-instruct-2507 |           101 |      2 | reasoning  |            8 |            1 |            -7 |
| qwen3-4b-instruct-2507 |           122 |      2 | coding     |           10 |            3 |            -7 |

