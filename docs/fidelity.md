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
| [MemGuard](defenses/memguard.md#differences-from-the-paper) | quantization width of the one-time randomness is an option | the paper gives none |
| MemGuard | defense classifier on as many training members as validation samples | keeps the paper's balance of members and non-members |
| MemGuard | the search on c3 stops at the authors' bound | as in the authors' code |
| MemGuard | a failed search on c3 returns the last noise that succeeded | as in the authors' code; Algorithm 1 as printed returns no noise |
| MemGuard | `neural_feat` attacks the undefended features | lets every attack run; the paper only attacks outputs |
| [MIST](defenses/mist.md#differences-from-the-paper) | ResNet-18's C and λ for every CIFAR model | the benchmark has no AlexNet or DenseNet |
| MIST | mixup in Phase 1 only | the paper does not say where |
| MIST | optimizer state shared by the two phases and averaged over the copies | the paper does not say |
| MIST | the best validation model, not the last epoch's | as for every defense in the benchmark |
| [SELENA](defenses/selena.md#differences-from-the-paper) | the final model is picked on the validation split, not on test accuracy | the test split holds the audited non-members |
| [RelaxLoss](defenses/relax_loss.md#differences-from-the-paper) | α from the official configs, ResNet-20's for every CIFAR model | the paper gives none |
| RelaxLoss | correct predictions get gradient ascent when only incorrect ones are flattened | as in the official code; the paper does not say |
| RelaxLoss | flattened predictions get no gradient ascent | as in the paper; the official code adds it |
| RelaxLoss | epochs count from 1 | as in Algorithm 1; the official code starts at 0 |
| RelaxLoss | soft labels are constants | as in the paper; the official code backpropagates through them |
| RelaxLoss | the best validation model, not the last epoch's | as for every defense in the benchmark |
