Walk-forward backtest over 68 races.

| Method | Spearman | MAE (positions) | Winner | Podium | Top 10 |
|---|---|---|---|---|---|
| Model (position) | 0.691 | 3.31 | 44% | 58% | 79% |
| Model (change from grid) | 0.690 | 3.32 | 40% | 62% | 79% |
| Starting grid | 0.677 | 3.25 | 60% | 68% | 79% |
| Championship order | 0.607 | 3.85 | 34% | 58% | 75% |
| Recent form (last 5) | 0.594 | 3.95 | 22% | 50% | 73% |

| Spearman per year | Model (position) | Model (change from grid) | Starting grid | Championship order | Recent form (last 5) |
|---|---|---|---|---|---|
| 2024 | 0.716 | 0.710 | 0.730 | 0.662 | 0.651 |
| 2025 | 0.669 | 0.669 | 0.651 | 0.532 | 0.519 |
| 2026 | 0.686 | 0.692 | 0.645 | 0.630 | 0.616 |

| Spearman, per race | model better | baseline better | ties | sign test p |
|---|---|---|---|---|
| Model (position) vs starting grid | 36 | 32 | 0 | 0.7163 |
| Model (change from grid) vs starting grid | 31 | 35 | 2 | 0.7122 |
| Model (position) vs championship order | 47 | 21 | 0 | 0.0022 |
| Model (position) vs recent form (last 5) | 54 | 13 | 1 | 0.0 |
