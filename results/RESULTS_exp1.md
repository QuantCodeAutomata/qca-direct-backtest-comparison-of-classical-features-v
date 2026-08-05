# Experiment: exp1_direct_backtest

## Median AUC by Configuration

| dataset_type   | family   |   training_size |   blinding_day |   median_auc |   std_auc |   count |
|:---------------|:---------|----------------:|---------------:|-------------:|----------:|--------:|
| classical      | LR       |              50 |              0 |          0.5 |    0.415  |      34 |
| classical      | LR       |              50 |              1 |          0.5 |    0.4398 |      34 |
| classical      | LR       |             100 |              0 |          0.5 |    0.4024 |      21 |
| classical      | LR       |             100 |              1 |          0.5 |    0.4024 |      21 |
| classical      | XGB      |              50 |              0 |          0.5 |    0.3586 |      34 |
| classical      | XGB      |              50 |              1 |          0.5 |    0.3881 |      34 |
| classical      | XGB      |             100 |              0 |          0.5 |    0.37   |      21 |
| classical      | XGB      |             100 |              1 |          0.5 |    0.3842 |      21 |
| hardware       | LR       |              50 |              0 |          0.5 |    0.3904 |      34 |
| hardware       | LR       |              50 |              1 |          0.5 |    0.3767 |      34 |
| hardware       | LR       |             100 |              0 |          0.5 |    0.3842 |      21 |
| hardware       | LR       |             100 |              1 |          0.5 |    0.3291 |      21 |
| hardware       | XGB      |              50 |              0 |          0.5 |    0.4254 |      34 |
| hardware       | XGB      |              50 |              1 |          0.5 |    0.4107 |      34 |
| hardware       | XGB      |             100 |              0 |          0.5 |    0.4183 |      21 |
| hardware       | XGB      |             100 |              1 |          0.5 |    0.3964 |      21 |
| sim_long       | LR       |              50 |              0 |          0.5 |    0.3904 |      34 |
| sim_long       | LR       |              50 |              1 |          0.5 |    0.3767 |      34 |
| sim_long       | LR       |             100 |              0 |          0.5 |    0.3842 |      21 |
| sim_long       | LR       |             100 |              1 |          0.5 |    0.3291 |      21 |
| sim_long       | XGB      |              50 |              0 |          0.5 |    0.4254 |      34 |
| sim_long       | XGB      |              50 |              1 |          0.5 |    0.4107 |      34 |
| sim_long       | XGB      |             100 |              0 |          0.5 |    0.4183 |      21 |
| sim_long       | XGB      |             100 |              1 |          0.5 |    0.3964 |      21 |
| sim_short      | LR       |              50 |              0 |          0.5 |    0.3635 |      34 |
| sim_short      | LR       |              50 |              1 |          0.5 |    0.3963 |      34 |
| sim_short      | LR       |             100 |              0 |          0.5 |    0.4155 |      21 |
| sim_short      | LR       |             100 |              1 |          0.5 |    0.4603 |      21 |
| sim_short      | XGB      |              50 |              0 |          0.5 |    0.4093 |      34 |
| sim_short      | XGB      |              50 |              1 |          0.5 |    0.4225 |      34 |
| sim_short      | XGB      |             100 |              0 |          0.5 |    0.3964 |      21 |
| sim_short      | XGB      |             100 |              1 |          0.5 |    0.4068 |      21 |

## Relative Uplift vs Classical Timestamps

| dataset_type   | family   |   training_size |   blinding_day |   median_auc |   std_auc |   count |   baseline_auc |   relative_uplift |
|:---------------|:---------|----------------:|---------------:|-------------:|----------:|--------:|---------------:|------------------:|
| classical      | LR       |              50 |              0 |          0.5 |    0.415  |      34 |            0.5 |                 0 |
| classical      | LR       |              50 |              1 |          0.5 |    0.4398 |      34 |            0.5 |                 0 |
| classical      | LR       |             100 |              0 |          0.5 |    0.4024 |      21 |            0.5 |                 0 |
| classical      | LR       |             100 |              1 |          0.5 |    0.4024 |      21 |            0.5 |                 0 |
| classical      | XGB      |              50 |              0 |          0.5 |    0.3586 |      34 |            0.5 |                 0 |
| classical      | XGB      |              50 |              1 |          0.5 |    0.3881 |      34 |            0.5 |                 0 |
| classical      | XGB      |             100 |              0 |          0.5 |    0.37   |      21 |            0.5 |                 0 |
| classical      | XGB      |             100 |              1 |          0.5 |    0.3842 |      21 |            0.5 |                 0 |
| hardware       | LR       |              50 |              0 |          0.5 |    0.3904 |      34 |            0.5 |                 0 |
| hardware       | LR       |              50 |              1 |          0.5 |    0.3767 |      34 |            0.5 |                 0 |
| hardware       | LR       |             100 |              0 |          0.5 |    0.3842 |      21 |            0.5 |                 0 |
| hardware       | LR       |             100 |              1 |          0.5 |    0.3291 |      21 |            0.5 |                 0 |
| hardware       | XGB      |              50 |              0 |          0.5 |    0.4254 |      34 |            0.5 |                 0 |
| hardware       | XGB      |              50 |              1 |          0.5 |    0.4107 |      34 |            0.5 |                 0 |
| hardware       | XGB      |             100 |              0 |          0.5 |    0.4183 |      21 |            0.5 |                 0 |
| hardware       | XGB      |             100 |              1 |          0.5 |    0.3964 |      21 |            0.5 |                 0 |
| sim_long       | LR       |              50 |              0 |          0.5 |    0.3904 |      34 |            0.5 |                 0 |
| sim_long       | LR       |              50 |              1 |          0.5 |    0.3767 |      34 |            0.5 |                 0 |
| sim_long       | LR       |             100 |              0 |          0.5 |    0.3842 |      21 |            0.5 |                 0 |
| sim_long       | LR       |             100 |              1 |          0.5 |    0.3291 |      21 |            0.5 |                 0 |
| sim_long       | XGB      |              50 |              0 |          0.5 |    0.4254 |      34 |            0.5 |                 0 |
| sim_long       | XGB      |              50 |              1 |          0.5 |    0.4107 |      34 |            0.5 |                 0 |
| sim_long       | XGB      |             100 |              0 |          0.5 |    0.4183 |      21 |            0.5 |                 0 |
| sim_long       | XGB      |             100 |              1 |          0.5 |    0.3964 |      21 |            0.5 |                 0 |
| sim_short      | LR       |              50 |              0 |          0.5 |    0.3635 |      34 |            0.5 |                 0 |
| sim_short      | LR       |              50 |              1 |          0.5 |    0.3963 |      34 |            0.5 |                 0 |
| sim_short      | LR       |             100 |              0 |          0.5 |    0.4155 |      21 |            0.5 |                 0 |
| sim_short      | LR       |             100 |              1 |          0.5 |    0.4603 |      21 |            0.5 |                 0 |
| sim_short      | XGB      |              50 |              0 |          0.5 |    0.4093 |      34 |            0.5 |                 0 |
| sim_short      | XGB      |              50 |              1 |          0.5 |    0.4225 |      34 |            0.5 |                 0 |
| sim_short      | XGB      |             100 |              0 |          0.5 |    0.3964 |      21 |            0.5 |                 0 |
| sim_short      | XGB      |             100 |              1 |          0.5 |    0.4068 |      21 |            0.5 |                 0 |

## Key Findings

- **Best relative uplift**: 0.00% for model family `LR` on dataset `classical` (training_size=50, blinding_day=0).
- **Highest median AUC**: 0.5000 (family=`LR`, training_size=50, blinding_day=0).
