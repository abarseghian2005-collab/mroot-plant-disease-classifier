# MROOT — final evaluation complete

Frozen system: clean control Base448 epoch 7 + MixUp/CutMix Base448 epoch 11, equal weights, fixed TTA.

**Holdout: 623 images. Macro mAP: 92.9896%. Top1: 89.4061% (557/623). Top5: 99.3579% (619/623).**

The same 623-image subset was used. The 47 previously excluded images were not evaluated. Neither model was trained on these holdout images or their detected training copies.

The system was frozen before this evaluation. No post-holdout model selection, tuning, additional training, or deployment change was made. Original data and checkpoints are preserved.

This is a follow-up evaluation on a previously evaluated holdout, not a first-ever blind test. Class reports, errors, confusion matrix, predictions, checkpoint hashes, and recipes are saved.

**All experiments are finished.**

Artifacts: /kaggle/working/MROOT_CLEAN_MIXUP_COMPARISON_20261024T064448_671839Z/FINAL_HOLDOUT_20261024T103253_167194Z