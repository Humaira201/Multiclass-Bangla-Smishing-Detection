| Model | Accuracy | Macro F1 | Note |
|---|---|---|---|
| TF-IDF + Logistic Regression (text only) | 0.8952 | 0.8933 | This project, 1253-message dataset (1002/125/126 split) |
| TF-IDF + Structured Features | 0.9516 | 0.9500 | This project, 1253-message dataset (1002/125/126 split) |
| BanglaBERT (dual-input: text+structured) | 0.9683 | 0.9668 | This project, 1253-message dataset (1002/125/126 split) |
| BanglaBERT (dual-input) - FINAL leakage-free | 0.9677 | 0.9663 | This project, 1253-message dataset (1002/125/126 split) |
| BERT + Char-CNN Hybrid (published) | 0.9847 | — | 3-class, ~2287 messages, different dataset |
| BanglaBERT + XGBoost / NirapodBarta (published) | — | — | Multiclass, BanglaBarta dataset (Mendeley) — check paper for exact reported metric |
| BiLSTM-based (published) | — | — | Check paper for exact reported metric |