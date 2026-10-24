# MROOT detailed speaking notes

This expanded deck contains 30 main slides and six appendix slides. A full explanation takes about 20-25 minutes. For a talk of about ten minutes, use slides 1, 3, 5, 6, 12, 14, 16, 17, 22, 24, 26, 27, 29, 30. The extra slides preserve the experiment details for questions and rehearsal.

Historical motives are reconstructed from the recorded experiment setup where the handover did not state them explicitly. Historical rounded scores are approximate. Proposals are separate from completed experiments.

## 1. MROOT

Hi, I’m Armen. MROOT is my plant disease classification project. I’ll explain the decisions behind it, including experiments that failed. The main question was not just which individual model had the best score. I wanted to find a small group of models that worked well together, while checking whether the data and evaluation were trustworthy. At the end I’ll show the deployed app.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded.

## 2. What the app is meant to do

The user uploads one photo and receives the model’s five highest disease scores. The label is the disease, rather than a plant-and-disease combination. That is the task I kept throughout the project. This is also a limited task: the model has not learned a healthy class or an unknown class. An unrelated photo can still get a confident disease label. For the presentation, I describe it as a classification demo, not a complete diagnosis system.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded.

## 3. How I separated the data

The source folders were called data_train and data_val. The folder name alone does not decide what a model sees. I used CSV manifests that list the allowed files. The original clean training manifest had 6,870 images and validation had 390. The 670-image holdout came physically from the original train source pool, but its files were excluded from the training manifest. Later image-group review produced a new final experimental split of 6,790 training and 388 validation images. The 368 comparison images are a subset of those 388, used when comparing against older models with historical copy exposure. These are not independent test sets.

Sources: reports/raw/split_summary.json, class_support.csv, image_metadata.csv, eda_summary.json, and exact data/manifests CSVs. Original handover explains that the holdout comes from the source train pool but was excluded by manifest.

## 4. Why class balance mattered

The final training data is uneven. Powdery mildew has 755 images, while berry blotch has 68. This is about an eleven-to-one difference. A model that focuses on the most common diseases could look good overall while neglecting small classes. That motivated class-balanced training and macro mAP. I also checked image dimensions, aspect ratios, labels, and whether files could be opened. All 7,178 final train and validation files passed the basic missing/unreadable checks. That tells me the files work. It does not prove the diagnosis labels are correct. The chart shows selected classes from the 39-class table.

Sources: reports/raw/split_summary.json, class_support.csv, image_metadata.csv, eda_summary.json, and exact data/manifests CSVs. Original handover explains that the holdout comes from the source train pool but was excluded by manifest.

## 5. How I read the scores

Top-1 is the most direct measure for a user reading the first label. Top-5 tells me whether the correct answer is still among the suggestions. Macro mAP evaluates how well the model ranks each disease’s images above other images, then gives every class equal weight. It is not a percentage of photos classified correctly. For example, the final 96.48% validation mAP and 91.30% Top-1 describe different things. Macro mAP was the primary metric before the last ensemble selection, so I kept that rule even when another model had higher Top-1.

Sources: Project metric implementations: sklearn average_precision_score over one-hot targets and probability columns; reports/raw/COMPLETED_COMPARISON.json.

## 6. From the baseline to three EVA models

The clean ViT baseline was about 90.64% mAP. The project then used pretrained EVA-02 Small models at 336 pixels and fine-tuned them for the 39 diseases. Different seeds give different training randomness, which can produce different errors. The original seed 1 selected epoch 11 at about 92.00% normal-view mAP. Seed 2 selected epoch 8 at about 92.38%, and seed 3 selected epoch 9 at about 91.09%. Averaging their fixed-view probabilities was more useful than selecting only the best individual model. The two-model result was about 93.53%, and the three-model result was 94.72%. I reproduced the saved three-model result at 0.9471792916 before new experiments. These are historical results on the original 390 validation images, not a comparison with the later 368 subset.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded. EVA-02 model source: https://arxiv.org/abs/2303.11331.

## 7. Why each model sees three views

This is test-time augmentation, or TTA. The model is already trained. I run several views of the same photo and average their probabilities. The padded view preserves the whole image while filling the unused border with the model’s mean RGB color. The horizontal flip changes orientation without introducing a new training run. The three-view recipe came from the existing system. I preserved it so a new result was comparable to the old system. I did not run an isolated ablation proving each view helps every model. In fact, the mixed model’s later normal-view Top-1 was stronger than its fixed-TTA Top-1 on one subset. TTA is a measured recipe, not a guarantee.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded. Frozen V2 implementation: ensemble.json and src/mroot/inference.py.

## 8. Another identical seed did not help

This result was in the original project history. The reason to try another seed is that training randomness can produce complementary predictions. But an extra model can also repeat the same mistakes or add less reliable probabilities. The fourth Small model did not help the existing ensemble. That is why I did not immediately spend the budget on seed 5 and seed 6. The result does not prove seed ensembles never help. The first three clearly did. It shows that each added member needs a contribution check. The historical motive here is reconstructed from the experiment type; the original prompt records the setup and result, rather than a full experiment proposal.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded.

## 9. Continuing seed 1 was not enough

Continuing training is a cheap-looking next step because the model already works. But a lower training loss does not mean better validation performance. The checkpoint can move away from a useful solution or fit the development data more closely. The handover says the continuation did not improve over seed 1. It does not provide enough exact continuation settings to diagnose a single cause. I therefore report the failure without inventing a learning-rate explanation. Later fine-tuning experiments changed a clear part of the recipe, such as layer decay or learning-rate scale, instead of repeating the same continuation and expecting a different outcome.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded.

## 10. A different architecture was not automatically useful

The idea was architectural diversity: a convolutional network and a Transformer do not process images in exactly the same way. That could be useful if the new model corrected mistakes made by EVA. The recorded ConvNeXt V2 Tiny result was about 90.54% mAP and it hurt the EVA ensemble. A different architecture does not guarantee useful diversity. Its predictions also have to be accurate enough and complementary enough. This result applies to the tested recipe, not every possible ConvNeXt model. I did not rerun the unchanged setup during the latest work. The proposed rationale is reconstructed from the historical experiment type.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded.

## 11. The historical Base result and the lost checkpoint

A larger EVA-02 Base at 448 pixels had looked promising earlier. The historical score was about 93.55% alone and 94.82% with the three Small models. However, that exact checkpoint was lost. A later rebuild reached about 93.12% alone and only 94.54% with the Small models. I could not use the stronger historical number as a current saved system. The reproducible reference remained the three Small models at 94.72%. This history gave a reason to investigate a larger model and higher resolution, but it also made saving checkpoints and configuration a practical requirement. Changing the backbone and the input size together does not isolate which change helps.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded.

## 12. Why I needed to investigate generalization

The original three-model system had about 94.72% macro mAP on 390 validation photos and 90.91% on the old 670-image holdout. The difference was about 3.81 percentage points. These sets contain different images, so this is not a controlled measurement of a single cause. The gap could reflect source differences, class difficulty, label noise, near duplicates, or repeated selection on a small validation set. The old holdout Top-1 was 85.52% and Top-5 was 98.51%, which also suggested many errors were ranking issues among plausible diseases. I treated this report as a warning about generalization, not as a tuning target. The next experiments used train and validation evidence. The 670-image holdout was already exposed, so a genuinely fresh test would require new images.

Sources: Original MROOT handover prompt supplied by the student, sections CURRENT REPRODUCIBLE MODEL, IMPORTANT PREVIOUS FINDINGS, TRAINING RECIPE, CURRENT VALIDATION RESULTS, and FINAL HOLDOUT RESULT. Historical values are approximate. Motives for old runs are reconstructed from their setup unless explicitly recorded.

## 13. Why I inspected mistakes before training more

The reproduced reference made 44 Top-1 mistakes on the original 390 validation images. In 31 of those cases, the correct label was ranked second. Another 11 had it at ranks three to five, and two were outside the first five. This suggested that many mistakes were close ranking decisions among plausible labels. It motivated checking model error overlap and simple ensemble weighting. It did not prove calibration or a specialist classifier would solve the problem. I also reviewed per-class performance, confusion patterns, high-confidence mistakes, and suspicious photos. The historical holdout gap gave a reason to investigate generalization, but I did not treat its old weak-class scores as targets for tuning.

Sources: Verified initial V2 diagnostics summarized in docs/experiments.md. Original holdout context in supplied handover.

## 14. Why exact duplicate checks were not enough

An identical file hash detects the same bytes, but resized, cropped, or compressed copies can have different hashes. Initial review produced 21 validation near-duplicate candidates and 95 quality candidates. These were review candidates, not 116 confirmed bad images. A related photograph also appeared with conflicting bacterial leaf spot and septoria leaf spot labels. That makes evaluation harder to interpret. Later image-group review quarantined 80 training rows and two validation rows in new experiment manifests. I kept the original files and manifests. The final review found no detected shared image groups in the 6,790/388 split, but no near-duplicate method proves every relationship is gone. The watermarked example also shows that source cues can exist independently of disease symptoms.

Sources: reports/raw/split_summary.json, class_support.csv, image_metadata.csv, eda_summary.json, and exact data/manifests CSVs. Original handover explains that the holdout comes from the source train pool but was excluded by manifest. Initial audit summary in docs/experiments.md. Image is the unaltered bacterial_leaf_spot source photo with Shutterstock/Alchemist from India credit retained.

## 15. A small ensemble-weight experiment

This experiment reused saved validation predictions, so it did not train five new neural networks. I divided validation into five stratified folds. For each fold, the other four selected one of four simple weight combinations. I then applied the selected weights to the held-out fold. Pooling those held-out predictions gave 94.5090% mAP, below the 94.7179% equal-weight reference. Choosing and reporting weights on all the same images could look better just by fitting validation noise. Cross-fitting was a check against that problem, although it is still based on reused validation data. I kept the grid small because roughly ten validation photos per class cannot support very flexible per-class weights.

Sources: configs/ensemble_weight_grid.json; scripts/crossfit_weights.py; original global-weight experiment result in the notebook. Original 390-image validation.

## 16. Natural sampling with a weighted loss

The original model used class-balanced replacement sampling. That gives rare classes more chances to appear, but can repeatedly show the same rare photo. I tested natural shuffled sampling so each training row is used once per epoch, with class weighting in the loss to retain pressure on rare diseases. Sampling and loss changed as a pair, so this is a test of that alternative balancing strategy. Normal-view validation selected epoch 13 at 92.3738% mAP, and fixed TTA reached 92.7203%. Adding the checkpoint to the original three Small models reached 94.8455%, only about 0.13 percentage points above the reference. The first attempt stopped on a numerical check and restarted from the same initialization with a lower AMP scale. That repair was not a second scientific experiment.

Sources: docs/experiments.md and saved natural-sampling result summarized from the Kaggle notebook. Same original 6,870 training / 390 validation manifests.

## 17. A new Base model at 448 pixels

The earlier Base result made 448 pixels worth investigating. Moving from 336 to 448 gives about 1.78 times as many input pixels, but it also costs more computation, and the Base architecture itself is larger. I cannot say the gain came only from resolution. I used layer-wise learning-rate decay: earlier backbone layers get smaller updates than later layers, with factor 0.8 between groups. The hypothesis was to adapt the task-specific features while disturbing earlier pretrained features less. The model selected epoch 4 at 93.8581% normal mAP and 94.2226% fixed-TTA mAP. With the original three Small models it reached 94.8424%. With the newer Small added as well, it reached 95.0680%. The useful result was the combination, not an assumption that a bigger model always wins.

Sources: docs/experiments.md; original Base experiment record in the Kaggle notebook. Architecture source https://arxiv.org/abs/2303.11331.

## 18. Why I reduced the learning rates

The first Base selected epoch 4, which made a gentler adaptation schedule a reasonable hypothesis. Early peaking alone does not prove the learning rate is too high, so I tested it. The gentler run kept the initial state and layer decay, and divided learning rates by three. It selected epoch 4 of 12 at 94.4735% normal mAP and 94.6558% fixed-TTA mAP. That was a clear standalone increase over the first Base. However, the predefined replacement ensemble fell from 95.1250% to 94.9153%. The new model may overlap more with the other members or change the probability average unfavorably. I therefore kept evaluating contribution rather than sorting checkpoints by standalone score alone.

Sources: reports/raw/gentler_base_result.json; docs/experiments.md. Original 390-image validation.

## 19. Why I checked averaged training weights

EMA maintains a running average of the training parameters, giving more weight to recent values. It is different from averaging the predictions of independently trained models. The idea was that a smoothed set of parameters might validate better than the current training weights. To keep this comparison narrow, I evaluated EMA at the raw model’s already selected best epoch. It reached about 94.0550% fixed-TTA mAP, below 94.6558% for the raw checkpoint. This does not show that every EMA setting fails. It shows that the matched EMA checkpoint was not a useful replacement in this experiment. The final two-model system therefore uses raw checkpoints.

Sources: Saved matched-epoch EMA result in reports/raw/gentler_base_result.json and docs/experiments.md. EMA implementation reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py.

## 20. Why I compared selected combinations

The question changed from which model is best alone to which predictions work well together. Among targeted checks on the original 390 validation images, the two Base models plus the newer Small reached 95.3678% mAP. Including original seed 1 gave 95.5080%. These were useful candidate systems at that stage. I also inspected 103 nontrivial subsets of an earlier five-model prediction pool. That broad search was exploratory. Cross-fitting subset selection gave about 94.4835%, which showed the choice was unstable. I did not treat the most favorable full-validation subset as an unbiased gain. The user also canceled a prematurely queued proposal before training, so it does not count as a completed run.

Sources: docs/experiments.md; recorded targeted-ensemble and exploratory-subset results in the Kaggle notebook.

## 21. Why I trained a fresh clean control

Cleaning the split changes the experiment. If I compared a new model trained on cleaned data against an old model trained with historical related images, I would mix several explanations together. I therefore trained a clean control under the final split, then a candidate with the extra mixing policy. Both started from the same archived initial tensors and seed 27182. Both used the balanced sampler, the gentler learning rates, layer decay, and the same ordinary augmentations. The candidate changed the mixing policy. This gives a much clearer comparison than comparing a final mixed model directly with the earliest Small model. It is still one training seed per arm, so it does not measure training-seed variability.

Sources: Saved project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, control/training_history.csv, mixup_cutmix/experiment_config.json, mixup_cutmix/training_history.csv. See docs/experiments.md for the earlier V2 runs. Split source: reports/raw/split_summary.json, class_support.csv, image_metadata.csv, eda_summary.json, and exact data/manifests CSVs. Original handover explains that the holdout comes from the source train pool but was excluded by manifest.

## 22. MixUp and CutMix in plain language

MixUp blends two training images and blends their target labels using the same proportions. For example, a purely illustrative 70/30 mixture uses a target of 70% one disease and 30% the other. CutMix replaces a patch of one image with a patch from another and mixes the target labels according to the patch area. The hypothesis was to discourage memorizing individual photos and to make training less dependent on one small cue. For plant diseases, mixing can also hide useful lesion detail, so it needed an actual validation check. The final candidate used MixUp alpha 0.2, CutMix alpha 1.0, application probability 0.5, and switch probability 0.5. Because both methods formed one policy, I cannot identify which one caused the measured change.

Sources: Zhang et al., mixup, https://arxiv.org/abs/1710.09412. Yun et al., CutMix, https://arxiv.org/abs/1905.04899. Project effect measured in the final paired experiment, not inferred from the papers. Exact settings: configs/final_training.json and mixup_cutmix/experiment_config.json.

## 23. How I chose the best training epoch

An epoch is one complete pass through the defined training draws. For the balanced replacement sampler, that means 6,790 draws, rather than a guarantee that every distinct photo appears once. After each full epoch, I ran validation on the same 388 images. The best normal-view mAP selected the checkpoint, not the final training epoch and not the holdout. I saved each new best under a unique filename. Early stopping patience was eight. The control completed 15 epochs and selected epoch 7. The mixed run completed 19 and selected epoch 11. Only after selecting those checkpoints did I compare their fixed-TTA predictions and ensemble combinations.

Sources: Saved project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, control/training_history.csv, mixup_cutmix/experiment_config.json, mixup_cutmix/training_history.csv. See docs/experiments.md for the earlier V2 runs.

## 24. The final per-epoch validation curves

Both curves use normal-view macro mAP on the same 388 validation images. The control reached its best score of 94.8053% at epoch 7. The mixed run reached 96.0699% at epoch 11, a difference of about 1.26 percentage points in this run. The control stopped at epoch 15 and the mixed run at epoch 19. A slightly better final training loss would not have changed the already selected checkpoint. The control recorded one handled AMP overflow late in training, while the mixed run recorded none. The saved logs make that visible. These curves support the observed candidate choice, but one pair of curves cannot establish that the policy will win for every random seed or external dataset.

Sources: Saved project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, control/training_history.csv, mixup_cutmix/experiment_config.json, mixup_cutmix/training_history.csv. See docs/experiments.md for the earlier V2 runs.

## 25. The mixed run helped some classes and hurt others

The overall score can hide uneven class behavior. This table compares the two selected checkpoints using fixed TTA on all 388 validation images. It includes several weaker or changing classes rather than only favorable examples. Bacterial leaf spot remains difficult even after improvement. With around ten validation images for many classes, an AP change can be sensitive to a few photos. I use this as a diagnosis of what happened on validation, not a claim that each disease has a stable measured improvement. The final pair’s full per-class and confusion records remain in the reports. This table compares individual checkpoints, rather than the final two-model pair.

Sources: reports/raw/validation_class_comparison.csv; support in class_support.csv. Table values are from the saved selected checkpoints.

## 26. Why the final ensemble has two models

I compared meaningful candidates on the same shared 368 images. The mixed model alone reached 96.0308% mAP with fixed TTA. Adding the clean control reached 96.4825%. Adding an earlier Base reached 96.1754%, and adding the newer Small reached 96.0329%. The four-model replacement using the mixed Base reached 95.9897%. The clean pair therefore had the highest observed macro mAP among these comparisons, and it used fewer models than the earlier four-model system. However, the mixed model alone with normal inference had higher Top-1 at 92.3913%, while the final pair’s Top-1 was 91.3043%. I selected the pair by the previously chosen primary metric. The confidence interval for its small mAP advantage over the mixed model includes zero.

Sources: Saved project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, control/training_history.csv, mixup_cutmix/experiment_config.json, mixup_cutmix/training_history.csv. See docs/experiments.md for the earlier V2 runs. Conditional intervals: reports/raw/paired_validation_intervals.json.

## 27. What the final holdout report tells us

After freezing the final selection, I evaluated it on the fixed 623-image reporting subset. The historical holdout pool contained 670 images. The fixed mask excluded 46 detected training-related images and one validation-related image, leaving 623. The final pair reached 92.9896% macro mAP, 89.4061% Top-1, and 99.3579% Top-5. That is 557 correct first predictions out of 623, with 619 correct labels present in the first five. The previous four-model system on these same 623 images had 92.5852% mAP and 549 correct first predictions. The pair added eight correct predictions. This is not directly comparable to the old V1 report on 670 images, and the holdout had already been evaluated earlier. A new untouched external set is still needed for a stronger final estimate.

Sources: reports/raw/FINAL_HOLDOUT_20261024T103253_167194Z/FINAL_HOLDOUT_RESULT.json and FROZEN_SELECTION.json. Earlier four-model result uses the same fixed 623-image subset. This holdout was evaluated previously.

## 28. What I saved and deployed

A model is only useful later if the exact checkpoint, class order, and transforms are available. I backed up all eight selected checkpoints in a private Kaggle Dataset rather than relying on a running session’s temporary files. The final two models were exported to safetensors with tensor equality checks and SHA-256 verification. The separate V2 Space loads those two models and preserves the frozen three-view preprocessing. The original three-Small Space remains unchanged. One live request matched the reference within about 1.2 times ten to the minus six in probability and took about 3.09 seconds including the queue. That was a smoke test, not a performance benchmark. The repository includes modular inference code, a local FastAPI entry point, analysis notebooks, and the experiment records.

Sources: reports/live_deployment_verification.json; ensemble.json; verified Kaggle Dataset armenbars/mroot-final-models-20261024-110313; docs/reproducibility.md.

## 29. Live demonstration

For the demo I open the V2 Space, upload one plant photo, click Analyze photo, and read the five ranked disease names. I explain the difference between the highest model score and diagnostic certainty. The app does not contain a healthy class. If ZeroGPU is sleeping or the queue is unavailable, I use the saved interface screenshot and verification record as the fallback, and say clearly that a new live request did not complete. I would not start training or reevaluate the holdout during the presentation.

Sources: https://huggingface.co/spaces/Armennnn/mroot-plant-disease-classifier-v2 and reports/live_deployment_verification.json.

## 30. What I would improve next

The most valuable next step is stronger evidence, rather than automatically training another large model. The validation set is small and reused, the holdout was exposed earlier, and some disease names remain ambiguous. New independently collected photos would test whether the gain holds outside the development sources. I would also add reviewed healthy and unrelated examples before designing a rejection rule. A probability threshold chosen on the current validation data alone would need a separate check. The project taught me to distinguish a good individual model from a useful ensemble member, to inspect data relationships, and to save the exact artifacts needed to explain a result later. These are future proposals, not completed experiments.

Sources: Project evaluation limitations in docs/evaluation.md and original handover. Future steps are proposals.

## 31. All selected new training epochs (appendix)

The split column is essential. The first three new candidates use the original 390 validation images, while the final clean pair uses 388. Each best epoch came from normal-view macro mAP. Fixed TTA was evaluated after loading that checkpoint. The natural-sampling Small ran 21 epochs, the first Base 12, the gentler Base 12, the clean control 15, and the mixed candidate 19. Historical Small seed 1, 2, and 3 selected epochs 11, 8, and 9. No canceled queued cell should be counted as a completed training experiment.

Sources: Saved project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, control/training_history.csv, mixup_cutmix/experiment_config.json, mixup_cutmix/training_history.csv. See docs/experiments.md for the earlier V2 runs. Earlier runs: docs/experiments.md and original handover.

## 32. Which images each score uses (appendix)

The original validation set contains 390 images. Image-group review removes two rows for the final 388-image epoch-selection set. The 368 is a subset of those 388 used when comparing with older checkpoints that had historical copies. The holdout pool has 670 images, and the fixed final reporting subset has 623. The 47 excluded rows have detected relationships to training or validation. I keep each number next to its evaluation result so a change in the sample set cannot be mistaken for a model improvement.

Sources: reports/raw/split_summary.json, class_support.csv, image_metadata.csv, eda_summary.json, and exact data/manifests CSVs. Original handover explains that the holdout comes from the source train pool but was excluded by manifest. reports/raw/FINAL_HOLDOUT_20261024T103253_167194Z/FINAL_HOLDOUT_RESULT.json and FROZEN_SELECTION.json. Earlier four-model result uses the same fixed 623-image subset. This holdout was evaluated previously.

## 33. The exact shared final recipe (appendix)

Both final arms used EVA-02 Base at 448 pixels, seed 27182, and identical archived starting weights. They used class-balanced replacement sampling with 6,790 draws, effective batch size 16, two T4 GPUs, AdamW with weight decay 0.05, label smoothing 0.05, and layer decay 0.8. The head learning rate was 3.3333333e-5 and the last backbone rate 6.6666667e-6. There were three warmup epochs and cosine decay, with maximum 20 epochs and patience eight. AMP and gradient clipping 1.0 were common to both. The mixed candidate added its combined policy, with probability 0.5 and switch probability 0.5. The complete ordinary augmentations are in the saved configuration, rather than hidden in the presentation.

Sources: Saved project artifacts: reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, control/training_history.csv, mixup_cutmix/experiment_config.json, mixup_cutmix/training_history.csv. See docs/experiments.md for the earlier V2 runs. configs/final_training.json.

## 34. How certain are the small gains? (appendix)

The saved conditional bootstrap intervals include zero for both the mixed-versus-control normal-view difference and the final-pair-versus-mixed-alone fixed-view difference. In plain language, these resamples do not rule out no improvement. They resample the saved predictions, rather than rerunning training, selecting epochs again, or repeating the full search. They therefore do not capture every source of uncertainty. One seed per final arm also cannot tell me training-seed variability. The point estimates guided a bounded final choice, but I do not present a small observed gap as proof of generalization.

Sources: reports/raw/paired_validation_intervals.json and docs/evaluation.md.

## 35. Ideas from the original prompt that I did not test (appendix)

The original prompt contained a menu of possible directions. It did not say every one had already been run. Focal loss, balanced softmax, logit adjustment, SWA, progressive unfreezing, separate augmentation ablations, temperature scaling, and specialist reranking remained proposals. Group-aware neural cross-validation was discussed, but five separate neural training folds were not completed. The small cross-fitted weight experiment is different from retraining five models. I also did not repeat the unchanged failed seed-1 continuation or ConvNeXt setup, and I did not reproduce the lost strong Base checkpoint. Distinguishing proposals from measured experiments is part of the experiment record.

Sources: Original handover, PHASE 4 and PHASE 5; docs/experiments.md.

## 36. Sources and project links (appendix)

The original long MROOT handover provides the historical experiments and approximate scores. The new experiments use saved configurations, per-epoch CSVs, validation predictions, checkpoint hashes, and result JSONs from the Kaggle notebook. EVA-02, MixUp, and CutMix papers explain the methods, while the project’s own validation gives the results reported here. The supplied ACA project guidelines shaped the EDA, experiments, deployment, repository, and presentation coverage.

Sources: Supplied MROOT handover and ML Engineer Project Guidelines.pdf. Linked primary method references and project artifacts.
