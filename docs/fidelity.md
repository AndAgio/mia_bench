# Differences from the papers

Where our implementations depart from the original papers. Each method's page has the details.

| Method | Difference | Reason |
|---|---|---|
| [HAMP](defenses/hamp.md#differences-from-the-paper) | KL term averaged over classes, not summed | as in the authors' code, which the paper's parameters assume |
| HAMP | random images normalized like real inputs | our models take normalized images |
| HAMP | soft labels stop while the true class leads | the paper's condition cannot be met at gamma = 1 |
| HAMP | no one-point margin when picking the best model | not implemented yet |
| HAMP | `neural_feat` attacks the trained model's features | lets every attack run; the paper only attacks outputs |
