# Reproducibility

## Inference

The public model source is the immutable Space commit
`cb7bf0deb1179cd2ce445bf024f355c38660fb9e`. The downloader requests that revision
and verifies both safetensors SHA-256 values from `ensemble.json`.

The inference core comes from the tested deployment. The repository moves it
into `src/mroot/`, adds a downloader, and supplies local Gradio and FastAPI entry
points. It preserves each checkpoint's transform, class order, TTA coefficients,
and equal model weights. The live Space is a separate deployment and is not
automatically changed by GitHub commits.

## Training

The exact train/validation manifest bytes are committed. Images are not. The
saved SHA-256 values are:

- Train: `ac50f2feadafc51f9ccd51a7b78bb2f8c97187f98d882e22f7e41b11cf4c9f1e`.
- Validation: `af565b47a858f4ec66854912158015297e4b900bc115526152286e3988a1a84d`.

Exact replay also needs the trusted original starting checkpoint from the
archived Kaggle notebook output. The expected SHA-256 is
`2ad075bccc2241b2b6a253729a3c63571d9005102e1a80a06bfd1efe0266f810`.
Its recorded source is:

```text
/kaggle/input/notebooks/armenbars/mroot-deploy/
  EVA02_BASE448_LLRD08_REVIEWED_BALANCED_S27182_20261023T232813_645547Z/initial_state.pth
```

That is a historical runtime path, not a promise that every new notebook mounts
it at the same location. Add the appropriate saved notebook output as an input
and locate the file. If it is unavailable, inference still works from the final
public weights, but exact training replay is blocked. Initializing a fresh public
pretrained model would be a new run, not an exact reproduction of the saved one.

To run one arm deliberately on a GPU machine:

```bash
python -m pip install -r requirements-training.txt
python -m src.mroot.train --arm control --initial-state /path/initial_state.pth --output /new/run_control
python -m src.mroot.train --arm mixup_cutmix --initial-state /path/initial_state.pth --output /new/run_mixed
```

Use `--path-prefix OLD=NEW` if the local image directory has a different prefix.
The manifest files themselves remain unchanged. Every output root must be new.
The trainer checks the source initialization hash and manifest hashes, keeps
the fixed class mapping, and saves immutable best checkpoints. It never loads a
holdout manifest. The CLI wrapper is new code around the exported original
training core. Full retraining was not repeated for this repository update.

## Verification scope

- Saved model tensors matched the safetensors exports exactly.
- Both remote model file hashes matched the verified exports.
- CPU and GPU inference agreed on a validation image within about 0.0000006.
- The live Space agreed with the verified GPU output within about 0.0000012.
- Read-only EDA checked all 7,178 final train/validation files. No unreadable or
  missing files, missing disease labels, shared paths, or detected shared groups
  were found. This is not an exhaustive guarantee against undetected duplicates.
- API contract/input validation tests use a fake predictor to avoid large model
  downloads. They are distinct from real-model inference verification.

The Dockerfile is supplied for the local API. A Docker image build requires a
Docker engine and network access; the build has not been verified in this
desktop environment. The GitHub Actions workflow installs the lightweight test
dependencies and runs compilation plus the API/schema tests. Its actual run
status is visible under the repository's Actions tab after publication.

No MLflow tracking run, external-data evaluation, calibration study, or video
recording is claimed by this update.
