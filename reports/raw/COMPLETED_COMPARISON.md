# MROOT — completed Base448 MixUp/CutMix comparison

Both arms used the same 6,790 training images, 388 validation images, saved initial weights, sampler, optimizer, and learning rates. Validation ran after every complete epoch. Only the candidate mixed training examples and labels.

| Model / selected epoch | Normal mAP (388) | Fixed-TTA mAP (388) |
|---|---:|---:|
| control, epoch 7 | 94.8053% | 94.9691% |
| mixup_cutmix, epoch 11 | 96.0699% | 96.0542% |

The 368-image ensemble comparison is a subset of the same 388-image validation set. It excludes detected copies present in the historical training data used by the older ensemble members.

| Validation system | Models | mAP (368) | Top1 (368) | Top5 (368) |
|---|---:|---:|---:|---:|
| Existing four-model reference | 4 | 95.4285% | 91.3043% | 99.7283% |
| Same four members, clean control Base | 4 | 95.4302% | 91.3043% | 100.0000% |
| Same four members, MixUp/CutMix Base | 4 | 95.9897% | 91.3043% | 100.0000% |
| Control Base alone | 1 | 95.0930% | 90.7609% | 99.4565% |
| MixUp/CutMix Base alone | 1 | 96.0308% | 91.8478% | 99.7283% |
| MixUp/CutMix + clean control | 2 | 96.4825% | 91.3043% | 99.4565% |
| MixUp/CutMix + earlier Base | 2 | 96.1754% | 91.0326% | 100.0000% |
| MixUp/CutMix + newer Small | 2 | 96.0329% | 91.0326% | 99.4565% |
| MixUp/CutMix Base alone, normal view | 1 | 96.1433% | 92.3913% | 99.4565% |

Candidate minus control, normal validation mAP: +1.2646 percentage points. Four-model ensemble difference: +0.5595 points.

Neither training arm nor this validation review read or evaluated the 623-image final test. The split quarantine uses the prior image-identity audit. Original data, checkpoints, cells, and deployment are preserved. No additional training is queued.

Conditional confidence intervals and all recipes, class reports, curves, and selected checkpoints are saved in this experiment directory. These validation results do not establish an independent final-test improvement.

Artifacts: `/kaggle/working/MROOT_CLEAN_MIXUP_COMPARISON_20261024T064448_671839Z`.