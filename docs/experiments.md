# Experiment log

I started by reproducing the saved three-model system. After that, I tested a
small number of changes that the data audit or earlier results gave a reason to
try. I selected epochs with normal-view validation mAP and checked ensemble
contribution after training. I kept the failed results as part of the record.

## Earlier project history

These figures come from the supplied project history. Rounded historical values
are marked as approximate. They are not new reproductions.

| Model or experiment | Observed validation result | What I took from it |
| --- | --- | --- |
| Clean ViT baseline | mAP about 0.9064 | A baseline before EVA |
| EVA Small seed 1 | Best epoch 11, normal mAP about 0.9200 | Saved in the original backup |
| EVA Small seed 2 | Best epoch 8, normal mAP about 0.9238 | Added useful diversity |
| EVA Small seed 3 | Best epoch 9, normal mAP about 0.9109 | The three-model fixed-TTA ensemble reached 0.9472 |
| Extra identical Small seed | Normal mAP about 0.9130; ensemble about 0.9449 | Another seed could hurt the ensemble |
| Seed 1 continuation | Did not beat the saved checkpoint | I did not repeat unchanged continuation |
| ConvNeXt V2 Tiny 384 | mAP about 0.9054 and hurt the EVA ensemble | A different architecture did not automatically help |
| Historical EVA Base 448 | About 0.9355 alone and 0.9482 with the Small models | The exact checkpoint was lost, so this is not a reproducible final model |
| Rebuilt historical Base | About 0.9312 alone and 0.9454 in the ensemble | I did not report the lost successful model as reproduced |

The original V1 holdout report used 670 images: mAP about 0.9091, Top-1 85.52%,
and Top-5 98.51%. That holdout was therefore already exposed before the latest
work. It was not a new blind test for V2.

## Reproduced reference and diagnostics

The three saved Small checkpoints reproduced mAP **0.9471792916**, Top-1
**0.8871794872**, and Top-5 **0.9948717949** on the original 390 validation images.

The initial audit found 21 validation near-duplicate candidates, 95 quality
candidates, and a shared photograph with conflicting bacterial leaf spot and
septoria leaf spot labels. Exact overlap checks alone had missed this kind of
relationship. Of 44 Top-1 mistakes, 31 had the correct class at rank 2, 11 at
ranks 3-5, and two outside the first five. I treated these findings as reasons
to inspect data and error overlap, not as permission to change labels blindly.

## Experiments on the original 390-image validation set

| Experiment | Why I tried it | Outcome |
| --- | --- | --- |
| Small global-weight grid | Check whether one of the three saved Small models deserved more weight | Five-fold cross-fitted mAP 0.9450904096 versus 0.9471792916 for equal weights. Failed. |
| Natural sampling with weighted CE, new Small | Replacement sampling repeatedly sees some images and misses others within an epoch | 21 epochs, best epoch 13. Normal mAP 0.923738148, fixed TTA 0.927202526. Adding it to the original three reached 0.948454882. Small gain. |
| New EVA Base 448 with layer decay 0.8 | Test a larger model and higher resolution with conservative early-layer updates | 12 epochs, best epoch 4. Normal mAP 0.938581295, fixed TTA 0.942226320. Original three plus Base: 0.948423853. With the new Small as well: 0.950680364. |
| Gentler Base, learning rates divided by three | Test whether the first Base was adapting too quickly | 12 epochs, best epoch 4. Normal mAP 0.944735153, fixed TTA 0.946557722. Better standalone, but its predefined replacement ensemble fell to 0.949152620 from 0.951249720. |
| EMA at the same selected epoch | Compare smoothed weights without selecting an extra favorable epoch | Fixed TTA about 0.940549638. It did not beat the raw checkpoint. |
| Targeted ensemble checks | Look for complementary errors instead of adding every model | Both Bases plus the new Small reached 0.953678254. Adding seed 1 produced a four-model result of 0.955080015. |

The first natural-sampling attempt stopped on a numerical check. I restarted
the same recipe from the same initialization with a lower AMP starting scale.
That was a numerical repair, not a separate training hypothesis. A prematurely
queued Base proposal was canceled before training and does not count as a run.

I also inspected all 103 nontrivial subsets of an earlier five-model prediction
pool as an exploratory analysis. Cross-fitting the subset choice gave about
0.944835028 and showed that flexible subset search was unstable. I do not treat
the best exploratory subset as an unbiased estimate.

## Final controlled comparison

After image-group review, I used new manifests with 6,790 training and 388
validation images. Both arms started from the same archived tensor state and
used the same seed, sampler, optimizer, augmentation, learning rates, and
best-epoch rule. The candidate added the combined MixUp/CutMix policy. This does
not isolate MixUp and CutMix individually.

| Arm | Epochs completed | Selected epoch | Normal mAP on 388 | Fixed-TTA mAP on 388 |
| --- | ---: | ---: | ---: | ---: |
| Clean control Base 448 | 15 | 7 | 0.948052793 | 0.949691323 |
| MixUp/CutMix Base 448 | 19 | 11 | 0.960699077 | 0.960542023 |

The training logs contain validation after every full epoch. Early stopping
used patience 8. Both arms had a maximum of 20 epochs and 2.5 hours. The control
recorded one handled AMP overflow at epoch 15; the mixed run recorded none.

## Final ensemble selection on the same 368 images

| System | Models | Macro mAP | Top-1 |
| --- | ---: | ---: | ---: |
| Existing four-model reference | 4 | 0.954284985 | 0.913043478 |
| Same four members with clean control Base | 4 | 0.954301995 | 0.913043478 |
| Same four members with mixed Base | 4 | 0.959897091 | 0.913043478 |
| Clean control alone, fixed TTA | 1 | 0.950930481 | 0.907608696 |
| Mixed Base alone, fixed TTA | 1 | 0.960307638 | 0.918478261 |
| Mixed Base plus clean control | 2 | **0.964825453** | 0.913043478 |
| Mixed Base plus earlier Base | 2 | 0.961753827 | 0.910326087 |
| Mixed Base plus newer Small | 2 | 0.960328618 | 0.910326087 |
| Mixed Base alone, normal view | 1 | 0.961433075 | **0.923913043** |

I chose the two-model pair by the predefined primary metric, macro mAP. A single
mixed model had better Top-1 on this subset. The final choice therefore does not
maximize every metric. The bootstrap intervals include zero, which limits how
strongly I can interpret the differences.

## What I did not run

I did not train new identical seed 5/6 models, rerun the unchanged failed
ConvNeXt setup, reproduce the lost successful historical Base checkpoint, or
train five neural cross-validation folds. Grouped CV was proposed as a stronger
future protocol. I did not run an Optuna study or a broad automated search.
MLflow was not used retroactively; the experiment record consists of the actual
saved configurations, CSV histories, checkpoint hashes, and JSON result files.

## Final recipe

Architecture: `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`. Seed 27182.
Class-balanced replacement sampling, effective batch 16, two T4 GPUs, FP16 AMP,
gradient clipping 1.0, AdamW weight decay 0.05, label smoothing 0.05. Head LR
0.0000333333, last backbone LR 0.0000066667, layer decay 0.8, three warmup epochs
followed by 17-epoch cosine decay. No EMA in the final pair.

The mixed arm used MixUp alpha 0.2, CutMix alpha 1.0, application probability
0.5, and switch probability 0.5. Standard augmentation remained the same in
both arms. `reports/raw/control/experiment_config.json` records the full recipe.

The original starting file has SHA-256
`2ad075bccc2241b2b6a253729a3c63571d9005102e1a80a06bfd1efe0266f810`.
The final eight selected checkpoints are permanently backed up in the owner's
private Kaggle Dataset `armenbars/mroot-final-models-20261024-110313`. Only the
final two exported inference files are needed to run V2.
