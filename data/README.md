# Data and split definitions

The prediction target is the disease, independent of the plant species.
There are 39 labels and no healthy class.

| Stage | Training | Validation | Additional validation subset |
| --- | ---: | ---: | ---: |
| Original clean split | 6,870 | 390 | n/a |
| Final paired experiment | 6,790 | 388 | 368 |

The final experiment quarantined 80 training rows and two validation rows
after image-group and label-conflict review. The original images and
manifests remained unchanged. The 368-image subset excludes 20 additional
validation images that had copies in earlier models' training data. It is
part of the 388 validation images, not another test set.

The committed CSVs in `data/manifests/` are the exact final experiment
manifests. Their SHA-256 values are recorded in the saved configuration.
They contain image paths and labels, not image bytes. The paths refer to the
original Kaggle input layout. The training CLI supports a prefix mapping
when a local data directory differs.

To obtain the images, add the owner's `data_train` and `data_val` datasets to
a Kaggle notebook, together with `mroot-current-best-backup` for the original
manifests. Access to private datasets requires the owner's authorization.
Do not copy credentials into this repository. Dataset paths alone do not
grant a redistribution license. Ask the owner for dataset access if needed.

The 670-image historical holdout came from the source pool but remained
outside the training manifest. A later audit identified 46 images related
to training images and one related to validation. The final reporting mask
retained 623 images. This subset had already been evaluated with an earlier
system, so its final score is not a pristine unseen-test estimate.

Holdout images, labels, per-image predictions, and a holdout evaluation
launcher are intentionally absent here. `reports/raw/` contains the already
saved aggregate result. No notebook in this update reevaluates the holdout.
