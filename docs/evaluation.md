# Evaluation protocol

I used macro mAP as the primary development metric because the 39 diseases have
unequal training support. I also recorded Top-1, Top-5, class AP, and confusion
matrices. A ranking improvement does not necessarily improve Top-1 accuracy.

## Which images each number uses

| Name | Images | Purpose |
| --- | ---: | --- |
| Original clean validation | 390 | V1 and early V2 comparisons |
| Final experiment validation | 388 | Best-epoch selection for both final arms |
| Shared final comparison subset | 368 | Compare older/newer models without known historical training copies |
| Historical holdout pool | 670 | Previously evaluated V1 report |
| Fixed final reporting subset | 623 | Final reported comparison after the image-group audit |

The 368 images belong to the 388-image validation set. They are not a new
independent test set. The difference is 20 validation images with image copies
in older models' training history. Epoch selection for the final pair still used
all 388 images. This means the 368 subset does not eliminate selection bias.

The original exact path and byte/pixel overlap checks found no exact
train/validation or train/holdout overlap. Near-duplicate review later found
related photographs that those exact checks missed. The final split is disjoint
for the detected/reviewed groups, not a guarantee against every possible copy.

The final training/validation manifests quarantine 80 training and two validation
rows. Those are experimental manifest exclusions. They do not delete or relabel
the original images.

## Final holdout result

The original 670-image pool included 46 images related to training images and
one related to validation. A fixed mask retained 623 images before the reported
predictions. The final system returned 557 correct Top-1 and 619 correct Top-5
predictions on those 623 images.

| System on the same 623 images | Macro mAP | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| Previous four-model ensemble | 0.9258515190 | 0.8812199037 | 0.9935794543 |
| Final two-model ensemble | 0.9298959361 | 0.8940609952 | 0.9935794543 |

The model choice and TTA weights were frozen before this final run. The holdout
had already been evaluated in earlier project stages, so these results must not
be described as a first untouched-test evaluation. I did not run more experiments
after the final evaluation. The repository update reads the saved reports only.

## Uncertainty

The paired stratified bootstrap is conditional on the saved predictions. It
does not account for all the choices made while reusing validation, best-epoch
selection, or variation across random seeds. The 95% interval for the final pair
minus the mixed model alone on 368 images was approximately [-0.00272, 0.01234]
mAP. It includes zero. I describe the pair as the highest observed validation
mAP in the checked set, rather than a statistically proven generalization gain.

## Deployment check

Deployment verification used one validation image, not a holdout image. It
checked that the live top five match the verified final model and that the
uploaded checkpoint hashes match the local exports. It is an inference parity
check, not another model-selection experiment or a new accuracy estimate.
