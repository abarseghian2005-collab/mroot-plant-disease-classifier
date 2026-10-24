# MROOT speaking notes

The main talk uses slides 1-14, with about one minute for the live demo. Slides 15-20 are for questions. Aim for about ten minutes overall.

## 1. MROOT

Hi, I’m Armen. This is MROOT, my plant disease classification project. I’ll explain how I prepared the data, compared models, and chose the final ensemble. Then I’ll show the app. The model predicts a disease category from a photo. I’ll also point out the limits of the evaluation, because the score only makes sense if the split and the metric are clear.

Source notes: ACA guidelines and project validation example

## 2. The problem I chose

I wanted to build a small tool that someone can actually try, rather than finish with a notebook score. The input is a plant photo, and the output is a disease category. I kept the disease labels independent of the plant species, so the task has 39 labels. It is still a closed-set classifier. There is no healthy class, so I would not present this as a tool that can decide whether any plant is healthy or sick.

Source notes: Original project label definition and class_mapping.json.

## 3. The data and the final split

The original clean split had 6,870 training images and 390 validation images. Exact overlap checks had already run, but a later review found related photographs and one group with conflicting labels. For the final comparison, I used new manifests with 6,790 training images and 388 validation images. I quarantined rows in the experimental manifests and kept the original files. A 368-image subset helped compare the newer models with older models that had seen copies of some validation photos. It is part of the 388, not a separate test set.

Source notes: reports/raw/split_summary.json; exact CSVs in data/manifests; reports/raw/eda_summary.json.

## 4. What I found in the data

The classes are not equally represented. In the final training split, the smallest class has 68 images and the largest has 755. The validation set mostly has ten images per class, so individual photos matter a lot. I also checked image sizes, aspect ratios, missing labels, and file integrity. The read-only audit checked all 7,178 train and validation files and found no missing or unreadable files. That does not prove the labels are correct. It tells me which problems the basic checks did and did not find.

Source notes: reports/raw/class_support.csv; image_metadata.csv; eda_summary.json; notebooks/01_eda.ipynb. The chart shows selected classes, not all 39.

## 5. The starting model

Before trying a new idea, I reproduced the three saved EVA Small models. The fixed-view ensemble reached 94.72 percent macro mAP on the original 390 validation images, which matched the previous result. I used macro mAP as the main metric because every disease should contribute equally, even when training counts differ. Top-1 accuracy was 88.72 percent. Those two numbers answer different questions, so I keep both labels visible throughout the presentation.

Source notes: Supplied MROOT handover text and the verified project experiment log in docs/experiments.md. Historical rounded numbers are approximate. Reproduced reference: macro mAP 0.9471792916, Top-1 0.8871794872, Top-5 0.9948717949 on 390 images.

## 6. The experiments before the final pair

I did not assume that another model would improve the result. First, I tried a small weight grid for the three saved models. Cross-fitting did worse than equal weights, so I kept equal weights. Natural sampling with weighted loss gave only a small ensemble gain. Base models at 448 pixels helped, but a lower learning rate improved the individual model more than its predefined ensemble. That is why I checked how errors combined instead of choosing models only by their standalone scores. The appendix and repository keep the numbers for each run.

Source notes: Supplied MROOT handover text and the verified project experiment log in docs/experiments.md. Historical rounded numbers are approximate. Detailed LR run: reports/raw/gentler_base_result.json.

## 7. The final controlled comparison

For the last experiment, I wanted a comparison I could explain clearly. Both runs started from the same saved weights and used the same seed, clean split, optimizer, learning rates, sampler, and ordinary augmentations. The candidate added a combined MixUp/CutMix policy. MixUp blends two examples and their labels. CutMix replaces a region and adjusts the target labels. The idea was to reduce memorization. Because I tested the combined policy, I cannot say which augmentation helped on its own.

Source notes: reports/raw/control/experiment_config.json; mixup_cutmix/experiment_config.json. MixUp: https://arxiv.org/abs/1710.09412. CutMix: https://arxiv.org/abs/1905.04899.

## 8. Validation after every complete epoch

The control reached its best normal-view mAP at epoch 7, and the mixed run at epoch 11. The control completed 15 epochs and the mixed run 19, because both used early stopping with patience eight. I did not just use the final epoch. Each improved checkpoint had its own filename. The mixed model's best normal validation mAP was 96.07 percent, compared with 94.81 percent for the control. These are scores on the same 388 validation images.

Source notes: Project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/training_history.csv, mixup_cutmix/training_history.csv.

## 9. Why I kept two models

I compared the candidates on the same 368-image subset. The mixed model plus the clean control had the highest observed macro mAP at 96.48 percent. Adding a different Base or a Small model did not beat that pair. A single mixed model with normal inference had higher Top-1 accuracy on this subset, so this choice does not maximize every metric. I kept the predefined primary metric, macro mAP. The uncertainty intervals include zero, so I would not describe the small differences as a proven generalization gain.

Source notes: Project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/training_history.csv, mixup_cutmix/training_history.csv. Bootstrap caveats: reports/raw/paired_validation_intervals.json and docs/evaluation.md.

## 10. The final holdout report

After choosing the model and freezing its inference recipe, I ran the final report on the fixed 623-image subset. Macro mAP was 92.99 percent, Top-1 accuracy was 89.41 percent, and Top-5 was 99.36 percent. That means 557 correct first predictions out of 623. The preceding four-model system had 549 correct first predictions on the same images, so the final pair added eight. This holdout had already been evaluated earlier in the project. I am not calling it a fresh blind test, and I stopped experimentation after this report.

Source notes: reports/raw/FINAL_HOLDOUT_20261024T103253_167194Z/FINAL_HOLDOUT_RESULT.json and the previously saved four-model result on the same fixed 623-image subset. The holdout was previously evaluated.

## 11. The limitations I would keep visible

There are still issues I would not hide behind the aggregate score. The validation set is small and has been reused. Similar disease labels can be hard to separate, and some source photos contain watermarks or other distracting information. The example here is an actual validation photo. There is also no healthy or unknown class, so an unrelated image can receive a confident disease prediction. Before using this outside a demo, I would collect new external images with reviewed labels and test the model once on that new set.

Source notes: Original project validation example bacterial_leaf_spot, with its Shutterstock/Alchemist from India watermark left intact. Displayed to discuss dataset quality, not as original photography. Evaluation limits: docs/evaluation.md.

## 12. How the final app makes a prediction

The app loads the two exported models and the fixed class mapping. Each model gets three views: the saved normal resize, its horizontal flip, and a full-image padded view. The view probabilities use weights 0.375, 0.375, and 0.25. Then I average the two models equally. The app returns the five largest disease scores. I kept the exact saved preprocessing, including the squash resize setting, because changing that during deployment would change what I evaluated.

Source notes: ensemble.json; src/mroot/inference.py; reports/live_deployment_verification.json.

## 13. Live demonstration

I’ll open the V2 Space, upload a plant photo, and click Analyze photo. The interface shows the five likely disease labels. I would explain that these are model scores rather than guaranteed diagnoses. The published checkpoint hashes match the verified local exports, and a live prediction matched the local GPU result within about one millionth in probability. That request took about three seconds including its queue, but it is only a smoke test. If the connection or ZeroGPU queue is unavailable during the talk, I can show this interface and the saved verification record.

Source notes: Live app: https://huggingface.co/spaces/Armennnn/mroot-plant-disease-classifier-v2. reports/live_deployment_verification.json. Screenshot taken from the running app.

## 14. What I learned

The main lesson for me was that adding a model is not enough. I had to check whether its predictions helped the ensemble, and whether the data comparison was fair. I also learned to keep the training records and checkpoints together so I can explain and reproduce a result later. The final result is a working app with a smaller ensemble, but the next useful step would be new external validation and a healthy or unknown option. I would do that before treating the project as more than a demonstration.

Source notes: Project experiment log and evaluation limitations. Future steps are proposals, not completed experiments.

## 15. Earlier experiments (appendix)

These are rounded historical figures from the handover. I did not retrain them during the final update. The successful historical Base checkpoint was lost, so I explicitly separate an observed historical score from the saved reproducible models. The current final checkpoints are backed up and their export hashes are verified.

Source notes: Supplied MROOT handover text and the verified project experiment log in docs/experiments.md. Historical rounded numbers are approximate.

## 16. New training runs and selected epochs (appendix)

The split column matters. The first three new candidates used the original 390-image validation set. The last pair used 388 cleaned validation images. Each selected checkpoint came from normal-view validation mAP. The fixed-TTA score was measured after loading that same selected checkpoint. The natural-sampling Small completed 21 epochs, the first Base and gentler Base completed 12 each, and the clean pair completed 15 and 19.

Source notes: Project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/training_history.csv, mixup_cutmix/training_history.csv. Earlier runs: docs/experiments.md and reports/raw/gentler_base_result.json.

## 17. The shared final training recipe (appendix)

The exact configuration files and original training core are in the repository. Both arms used seed 27182, the same archived starting weights, class-balanced replacement sampling, and an effective batch of 16. AdamW used head learning rate one times ten to the minus four divided by three and final-backbone learning rate two times ten to the minus five divided by three. Earlier layers used decay 0.8. Both used warmup, cosine decay, label smoothing, AMP, and gradient clipping. The added combined mixing policy is the candidate difference.

Source notes: reports/raw/control/experiment_config.json and mixup_cutmix/experiment_config.json; configs/final_training.json.

## 18. Which split each result uses (appendix)

I would use this table if asked why there are several validation counts. The 368 subset is inside 388, and epoch selection still used the 388. The 623-image reporting mask came from the earlier 670 holdout pool after excluding detected related images. It had been evaluated before. I do not describe it as a new external test. These distinctions keep the scores comparable within a row and avoid treating every number as an independent result.

Source notes: reports/raw/split_summary.json; exact CSVs in data/manifests; reports/raw/eda_summary.json. reports/raw/FINAL_HOLDOUT_20261024T103253_167194Z/FINAL_HOLDOUT_RESULT.json and the previously saved four-model result on the same fixed 623-image subset. The holdout was previously evaluated.

## 19. Uncertainty and model selection (appendix)

The bootstrap resamples the saved validation predictions. It does not redo training or checkpoint selection, and one seed per arm does not measure training variation. Both intervals shown here include zero. The scores helped me choose a bounded final system, but they do not prove a small improvement will repeat on new data. The three-model weighting test also used cross-fitting on cached predictions, not five independently retrained neural models.

Source notes: reports/raw/COMPLETED_COMPARISON.json, intervals; configs/ensemble_weight_grid.json; docs/evaluation.md.

## 20. Sources and project links (appendix)

The project builds on publicly released EVA-02 models and the timm implementation. The MixUp and CutMix papers motivated the final policy, but the MROOT experiment is the evidence for its observed behavior here. The repository includes the saved configuration and metric artifacts. The supplied ACA guidelines shaped the EDA, deployment, repository structure, and presentation coverage.

Source notes: Supplied ML Engineer Project Guidelines.pdf, Armenian Code Academy, pp. 1-3. Other references are the linked primary sources.
