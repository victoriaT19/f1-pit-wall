Walk-forward backtest over 68 races.

| Method | Spearman | MAE (positions) | Winner | Podium | Top 10 |
|---|---|---|---|---|---|
| Ensemble (model + Plackett-Luce) | 0.705 | 3.18 | 46% | 63% | 81% |
| Plackett-Luce | 0.701 | 3.21 | 49% | 68% | 79% |
| Model (position) | 0.691 | 3.31 | 44% | 58% | 79% |
| Model (change from grid) | 0.690 | 3.32 | 40% | 62% | 79% |
| Starting grid | 0.677 | 3.25 | 60% | 68% | 79% |
| Championship order | 0.607 | 3.85 | 34% | 58% | 75% |
| Recent form (last 5) | 0.589 | 3.99 | 21% | 50% | 73% |

| Spearman per year | Ensemble (model + Plackett-Luce) | Plackett-Luce | Model (position) | Model (change from grid) | Starting grid | Championship order | Recent form (last 5) |
|---|---|---|---|---|---|---|---|
| 2024 | 0.744 | 0.750 | 0.716 | 0.710 | 0.730 | 0.662 | 0.646 |
| 2025 | 0.665 | 0.654 | 0.669 | 0.669 | 0.651 | 0.532 | 0.512 |
| 2026 | 0.706 | 0.697 | 0.686 | 0.692 | 0.645 | 0.630 | 0.613 |

| Spearman, per race | model better | baseline better | ties | sign test p |
|---|---|---|---|---|
| Ensemble (model + Plackett-Luce) vs starting grid | 41 | 27 | 0 | 0.1143 |
| Ensemble (model + Plackett-Luce) vs model (position) | 47 | 21 | 0 | 0.0022 |
| Plackett-Luce vs starting grid | 41 | 27 | 0 | 0.1143 |
| Model (position) vs starting grid | 36 | 32 | 0 | 0.7163 |
| Model (change from grid) vs starting grid | 31 | 35 | 2 | 0.7122 |
| Model (position) vs championship order | 47 | 21 | 0 | 0.0022 |
| Model (position) vs recent form (last 5) | 56 | 12 | 0 | 0.0 |

Held out (2025-2026, not used to choose any hyperparameter):

| Method | Spearman | MAE | Winner | Podium | Top 10 |
|---|---|---|---|---|---|
| Ensemble (model + Plackett-Luce) | 0.684 | 3.32 | 52% | 64% | 78% |
| Plackett-Luce | 0.673 | 3.40 | 55% | 67% | 76% |
| Model (position) | 0.677 | 3.39 | 55% | 64% | 77% |
| Model (change from grid) | 0.679 | 3.38 | 55% | 65% | 77% |
| Starting grid | 0.648 | 3.45 | 68% | 68% | 77% |
| Championship order | 0.576 | 4.06 | 32% | 60% | 72% |
| Recent form (last 5) | 0.558 | 4.18 | 20% | 50% | 71% |

Probabilities, Brier score (lower is better):

| Method | win | podium | top 10 | win (2025-26) | podium (2025-26) | top 10 (2025-26) |
|---|---|---|---|---|---|---|
| Ensemble (model + Plackett-Luce) | 0.0297 | 0.0657 | 0.1451 | 0.0267 | 0.0626 | 0.1567 |
| Plackett-Luce | 0.0321 | 0.0698 | 0.1446 | 0.0301 | 0.0671 | 0.1574 |
| Starting grid | 0.0292 | 0.0659 | 0.1560 | 0.0250 | 0.0623 | 0.1648 |
