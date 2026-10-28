# Differences from the papers

Where our implementations depart from the original papers. Each method's page has the details.

| Method | Difference | Reason |
|---|---|---|
| [HAMP](defenses/hamp.md#differences-from-the-paper) | KL term averaged over classes, not summed | as in the authors' code, which the paper's parameters assume |
| HAMP | random images normalized like real inputs | our models take normalized images |
| HAMP | soft labels stop while the true class leads | the paper's condition cannot be met at gamma = 1 |
| HAMP | no one-point margin when picking the best model | not implemented yet |
| HAMP | `neural_feat` attacks the trained model's features | lets every attack run; the paper only attacks outputs |
| [LDL](defenses/ldl.md#differences-from-the-paper) | returns the averaged logits, not only the label | lets every attack run; label-only attacks see the paper's defense |
| LDL | 0.02 is the noise std, not its variance | as in the authors' code |
| LDL | copies per query from the authors' code | the paper gives none for images |
| LDL | Gaussian noise on News | the paper does not cover News |
| LDL | non-binary inputs flip as 1 − x | label-only attacks query points between binary inputs |
| LDL | `neural_feat` attacks the undefended features | lets every attack run; the paper only attacks labels |
| [Purifier](defenses/purifier.md#differences-from-the-paper) | batch-normalized latent with fixed-scale noise | without a KL term, training removes the noise |
| Purifier | λ = 0.1 | the paper gives none; λ = 1 gives one-hot outputs |
| Purifier | squared L2 reconstruction loss | the paper only says L2 norm |
| Purifier | latent noise seeded per query | repeated queries get the same output |
| Purifier | acc_test measured on the validation split | the test split holds the audited non-members |
| Purifier | swap set matched by 1-NN on log-confidences, with a threshold | saturated confidences of different inputs collide |
| Purifier | swaps exchange the two largest scores | the paper does not say how |
| Purifier | the reformer outputs K scores | the appendix's 2K outputs cannot be confidences |
| Purifier | CIFAR-100 gets the learning rate listed for CIFAR10 | the appendix names CIFAR10 twice |
| Purifier | `neural_feat` attacks the undefended features | lets every attack run; the paper only attacks outputs |
