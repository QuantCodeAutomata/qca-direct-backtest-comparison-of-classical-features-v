# Experiment: exp3_robustness

## Median AUC by Configuration

| dataset_type   | family   |   training_size |   blinding_day |   median_auc |   std_auc |   count |
|:---------------|:---------|----------------:|---------------:|-------------:|----------:|--------:|
| idle           | LR       |              50 |              0 |          0.5 |    0      |      37 |
| idle           | LR       |              50 |              1 |          0.5 |    0      |      37 |
| idle           | LR       |             100 |              0 |          0.5 |    0      |      23 |
| idle           | LR       |             100 |              1 |          0.5 |    0      |      23 |
| idle           | XGB      |              50 |              0 |          0.5 |    0      |      37 |
| idle           | XGB      |              50 |              1 |          0.5 |    0      |      37 |
| idle           | XGB      |             100 |              0 |          0.5 |    0      |      23 |
| idle           | XGB      |             100 |              1 |          0.5 |    0      |      23 |
| noiseless      | LR       |              50 |              0 |          0.5 |    0.3885 |      37 |
| noiseless      | LR       |              50 |              1 |          0.5 |    0.3821 |      37 |
| noiseless      | LR       |             100 |              0 |          0.5 |    0.3976 |      23 |
| noiseless      | LR       |             100 |              1 |          0.5 |    0.3515 |      23 |
| noiseless      | XGB      |              50 |              0 |          0.5 |    0.4328 |      37 |
| noiseless      | XGB      |              50 |              1 |          0.5 |    0.4214 |      37 |
| noiseless      | XGB      |             100 |              0 |          0.5 |    0.4264 |      23 |
| noiseless      | XGB      |             100 |              1 |          0.5 |    0.4074 |      23 |
| noisy          | LR       |              50 |              0 |          0.5 |    0.3885 |      37 |
| noisy          | LR       |              50 |              1 |          0.5 |    0.3821 |      37 |
| noisy          | LR       |             100 |              0 |          0.5 |    0.3976 |      23 |
| noisy          | LR       |             100 |              1 |          0.5 |    0.3515 |      23 |
| noisy          | XGB      |              50 |              0 |          0.5 |    0.4328 |      37 |
| noisy          | XGB      |              50 |              1 |          0.5 |    0.4214 |      37 |
| noisy          | XGB      |             100 |              0 |          0.5 |    0.4264 |      23 |
| noisy          | XGB      |             100 |              1 |          0.5 |    0.4074 |      23 |
