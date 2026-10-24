# MROOT presenter guide

Read the explanations before rehearsing the slides. You do not need to recite every setting in the notebook. You should be able to explain the purpose of a method and what the comparison actually showed.

The audience deck has 25 main slides and five optional slides. A full explanation is likely to take 20-25 minutes. For an overview of about ten minutes, use slides 1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 15, 17, 18, 19, 21, 22, and 23, keeping the explanations brief. Rehearse once with a timer and shorten examples if necessary.

## The project in one minute

“MROOT suggests plant disease categories from photos. I started with image models that had already learned general visual patterns, then adapted them using labeled plant photos. I kept separate photos for learning and for comparing my experiments. Because some diseases had far fewer examples, I measured performance in a way that gave each disease equal importance. I tested how to balance the training data, how much to adjust a larger model, and whether mixing training examples helped. I also checked which models worked well together. The final app combines two models. Its first suggestion was correct for 557 of the 623 photos in our final report. That reporting set had already been evaluated earlier, so the next strong check would be an independently collected new set.”

## What the model is actually doing

A photo is a grid of pixels. The model performs many numerical operations on those pixels and produces a score for each of the 39 categories. Its layers contain learned numbers called **parameters** or **weights**.

During **training**, a photo comes with a **label**, meaning the category recorded for that example. A **loss** gives a numerical penalty for a poor prediction. The training procedure adjusts the parameters to reduce that penalty. A lower training loss alone does not prove the model will work better on new photos.

During **inference**, the model uses its trained parameters to score a new image. A user uploading a photo is requesting inference. The app does not retrain itself on that upload.

There are two different uses of the word “weights”:

- **Model weights** are the internal numbers learned in training.
- **Ensemble weights** decide how much each finished model contributes to the combined prediction. Our final models contribute equally.

## Training, validation, and testing

Think of training as learning from worked examples. Validation is a set of practice questions used to choose between approaches. A test is ideally a final exam kept away from those choices.

| Role | What happens | MROOT context |
| --- | --- | --- |
| Training | Parameters change using labeled examples | Final paired runs used 6,790 photos |
| Validation | Compare variants and select saved models | Final pair used 388 photos |
| Shared validation subset | Compare old and new models on the same eligible photos | 368 of those 388 |
| Final reporting | Report a frozen model choice | 623 photos from the historical holdout pool |

The earlier clean experiments used 6,870 training and 390 validation photos. After related-image review, new manifests excluded 80 training rows and two validation rows. A **manifest** is a file list used to decide which images belong to a split.

Why compare on 368 instead of all 388? Twenty validation photos had related copies in older models’ training history. I excluded those 20 when comparing the old and new models, so the comparison used the same eligible photos. The final pair still used all 388 validation photos to select its saved versions during training. The 368 photos are part of validation, and this extra comparison does not create a new independent test.

The original 670 holdout photos physically came from a folder named `data_train`, but their paths were excluded from the training list. The folder name does not mean a model trained on every file in it. The final fixed reporting mask excluded 47 related rows and left 623.

The holdout had already been evaluated before the final model choice. Freezing the last choice before its last evaluation does not make that set pristine again. Say this plainly if asked. We need a new external set for a stronger independent estimate.

## Accuracy, Top-5, and macro mAP

**Top-1 accuracy** checks whether the first suggestion is correct. **Top-5 accuracy** checks whether the correct category is in the first five. If a photo’s correct label is scab and scab appears second, Top-1 is wrong and Top-5 is right.

**Average precision (AP)** checks ranking for one disease across photos. An illustrative rust ranking is:

| Rank | Photo | Rust score | Actual label |
| ---: | --- | ---: | --- |
| 1 | A | 0.90 | rust |
| 2 | B | 0.80 | another disease |
| 3 | C | 0.70 | rust |

There are two rust photos. At the first rust photo, precision is 1/1. At the second, it is 2/3. With distinct scores in this example, AP is their average, about 83.3%. **mAP** stands for mean average precision. **Macro mAP** averages the APs of all 39 diseases equally.

This is a toy calculation, not a MROOT result. A 96.48% macro mAP does not mean 96.48% of photos got the correct first suggestion. The final pair’s validation Top-1 was 91.30% on the shared subset. Its reporting Top-1 was 89.41% on 623 photos.

**Precision** asks how many of the selected disease predictions are correct. **Recall** asks how many true examples of that disease were recovered. The AP calculation summarizes precision as more true examples are recovered down the score ranking.

## Pretraining, fine-tuning, and the model names

**Pretraining** learns general visual patterns from a separate large image collection. **Fine-tuning** adapts the pretrained parameters to our disease categories. This is useful when we have too few task photos to justify training a large image model from zero.

**ViT** means Vision Transformer. It represents an image using patches and combines information across them. **EVA-02** is the pretrained Transformer family used in this project. **Small** and **Base** indicate model sizes. Base has more parameters and greater compute cost. Greater size does not guarantee a better result.

**ConvNeXt V2** is a different architecture that uses learned local visual filters. It was a reasonable diversity candidate, but the specific Tiny 384 setup did not help the EVA ensemble.

## Why combining models can help

An **ensemble** combines predictions from several models. Imagine two students checking the same questions. Combining their answers is useful if they catch different mistakes. If the second student repeats the same errors or adds poor answers, the combination can get worse.

Our models output scores, so we average those scores rather than take a literal vote on the first label. A **seed** is a number used to control random choices, such as data order and some training operations. Changing a seed can give a separately trained model, but it does not guarantee useful differences.

The first three EVA Small models helped together. A fourth identical Small training run hurt the ensemble. The final clean pair was chosen because its measured combined mAP was highest among the final comparisons, not because two is always the ideal number.

## The experiments in ordinary language

| Experiment | Question behind it | What happened and what it means |
| --- | --- | --- |
| Clean ViT baseline | What does an initial image-model approach achieve? | Historical validation mAP about 90.64%; a reference for later work. |
| Three EVA Small seeds | Can separately trained models correct one another’s errors? | The saved three-model prediction recipe reproduced 94.72% mAP on 390 photos. |
| Fourth Small seed | Does another similar model add anything? | The ensemble fell to about 94.49%; another member can hurt. |
| Continue seed 1 | Will training an existing model longer improve it? | It did not beat the saved model. The exact historical continuation settings were not supplied. |
| ConvNeXt V2 Tiny 384 | Can a different architecture add useful predictions? | About 90.54% alone and a worse EVA ensemble. This tested recipe failed. |
| Historical Base 448 | Could a larger model and larger input help? | About 94.82% in an old combination, but the exact checkpoint was lost. A later rebuild gave about 94.54%. Neither is the current reproduced final system. |
| Simple voting weights | Should one existing Small model count more? | A check that chose weights on one validation part and checked another scored 94.51%, below equal weighting at 94.72%. |
| Natural sampling with weighted loss | Can we avoid repeatedly showing the same rare photos while still emphasizing rare diseases? | The new Small gave a small ensemble increase to about 94.85% on 390 photos. |
| New Base at 448 pixels | Can larger input and a larger pretrained model contribute? | A five-model combination reached about 95.07% on 390 photos. Size and resolution changed together, so their effects are not isolated. |
| Gentler learning rates | Can smaller training adjustments help the Base model? | Its individual score improved, but a planned ensemble replacement worsened. |
| EMA parameters | Could smoothing learned parameters improve the selected model? | The matched average scored about 94.05%, below the ordinary model at 94.66%. |
| Targeted combinations | Which predictions actually help together? | A four-model candidate reached about 95.51% on 390 photos. A broad subset search was exploratory and unstable under a held-out-part selection check. |
| Clean control and mixed candidate | Does adding image mixing help when the other conditions match? | One-view mAP was 94.81% for control and 96.07% for the mixed candidate on the same 388 photos. |
| Final two-model combination | Is the new model stronger alone or with another useful model? | Pair mAP was 96.48% on the same 368-photo comparison subset, versus 96.03% for the mixed model alone with multiple views. |

Historical motives are reconstructed from the supplied setup where no original hypothesis was recorded. Historical rounded figures are approximate. A comparison across different sample sets is not a clean measurement of a model-only change.

## Sampling and weighted loss

**Sampling** decides which training photos are drawn. A balanced sampler gives rare classes more chances. With **replacement**, a photo can be selected again, so some photos repeat and others may not appear during a round of draws.

**Weighted loss** changes the penalty used for training. A mistake on a rarer class can contribute more than a mistake on a common one. It does not collect new information or create new real examples.

The tested alternative used ordinary shuffled sampling and weighted loss. Because both parts changed together, the result applies to that alternative combination. Do not claim that we isolated a pure sampler effect.

## Learning rate and preserving pretrained information

The **learning rate** scales each training adjustment. A very large step can move away from a useful solution. Very small steps can make adaptation slow or ineffective. Smaller is not automatically better.

**Layer-wise learning-rate decay** gives earlier model layers smaller rates than later ones. A layer is a stage of the network’s computation. The idea was to disturb earlier pretrained representations less while adapting later parts more. Our layer decay was 0.8. That was a chosen recipe, not a separately proven effect.

The gentler Base experiment divided the rates by three and restarted from the same initial model. Its individual recorded multiple-view mAP increased from 94.22% to 94.66%. Its predefined replacement ensemble fell from about 95.12% to 94.92%. The individual and ensemble questions have different answers.

## MixUp and CutMix

**MixUp** blends two training photos and blends their label targets in matching proportions. **CutMix** replaces a patch using another photo and mixes targets according to the patch area.

The aim was to make reliance on one familiar photo less useful. It was a hypothesis, not a direct measurement of memorization. Mixing may also hide disease cues. Our final test added both methods as one policy, so it cannot identify which method helped individually. Uploaded user photos are ordinary inference inputs; the app does not mix them with a training photo.

## Several views during prediction

**Test-time augmentation (TTA)** means predicting several transformed views of the same input and averaging the scores. Each MROOT model uses its saved normal preprocessing, a horizontal flip, and a whole-image padded view.

**Padding** adds a border so the full image can fit a square without cutting away parts. The view weights are 0.375, 0.375, and 0.25. The two models then contribute 0.5 each. These are implementation details you can explain if asked; they do not need to be recited in the main talk.

This recipe did not win on every metric. The single mixed model with one view had higher first-choice accuracy on the final comparison subset, while the chosen pair had the higher primary mAP score.

## Other notebook terms, if someone asks

| Term | Plain meaning |
| --- | --- |
| Baseline | The reference approach used before testing a change. |
| Control | A matched reference run used to judge the added change. |
| Architecture | The structure of a model’s computation. |
| Parameter | An internal numerical setting learned during training. |
| Hyperparameter | A setting chosen for training, such as learning rate or batch size. |
| Batch | A group of examples processed together for training. |
| Gradient | Information about how changing parameters would change the loss. |
| Optimizer | The rule that turns gradients into parameter adjustments. AdamW was our optimizer. |
| Weight decay | Shrinks parameters during training. AdamW applies this separately from its gradient-based update. It does not guarantee less overfitting. |
| Label smoothing | Replaces a completely certain training target with a slightly softened one. |
| Epoch | A round of the defined training draws. With our replacement sampler, it need not visit every distinct photo exactly once. |
| Scheduler | Changes the learning rate over training. |
| Warmup | Starts with smaller rates and gradually increases them at the beginning. |
| Cosine decay | Reduces the rate using a smooth cosine-shaped schedule. |
| Gradient clipping | Limits the gradient size to reduce excessively large updates. |
| AMP / FP16 | Mixed-precision computation using lower precision for some operations, mainly for speed and memory. |
| Checkpoint | A saved model state that can be loaded later. |
| Early stopping | Stops training after validation has failed to improve for a chosen period. |
| EMA | A running average of learned parameters. This differs from averaging model predictions. |
| Fold | One part of a dataset used in a rotating selection/evaluation scheme. |
| Cross-fitting | Choose an option on some data parts and evaluate its predictions on another part. Our weight test reused saved predictions. |
| Cross-validation training | Repeating model training across different train/validation partitions. We did not train five final neural folds. |
| Ablation | A controlled comparison that removes or changes a component to examine its effect. |
| Confusion matrix | A table counting which true categories were predicted as which categories. |
| Calibration | Whether prediction scores agree with observed correctness rates. We did not establish calibrated diagnostic probabilities. |
| Bootstrap interval | An uncertainty estimate made by repeatedly resampling the available examples or predictions. |
| Hash / SHA-256 | A fingerprint used to check a file’s identity. Different resized copies can still have different hashes. |
| Deployment | Putting the saved model and prediction code into a usable application. |

## Questions you should be ready to answer

**Why not just report accuracy?** Accuracy is useful, so we report it. Macro mAP gives equal importance to each disease and also measures ranking. It was the chosen primary development metric.

**Why not use one huge model?** A larger model costs more and may still make the same mistakes. We compared both standalone and combined predictions. Two chosen models had the highest observed final mAP among the tested candidates.

**Does 96.48% mean the app is correct 96.48% of the time?** No. That is macro mAP on a reused validation subset. The first-choice reporting accuracy was 89.41% on 623 photos.

**Did the model train on validation or test photos?** The final runs trained on their training manifest, not those validation or reporting rows. However, historical related-image exposure and repeated evaluation matter. We reviewed detected image groups, and the old holdout was already exposed. Do not turn a clean manifest check into a claim of completely independent evaluation.

**Why was the test in a folder named data_train?** The source pool and the training manifest are different things. The manifest excluded the held-out paths. A folder name is not proof of training use.

**Did MixUp or CutMix cause the improvement?** The combined policy had a higher observed score than its matched control. We did not isolate the two methods or repeat the full pair over many seeds. We also did not prove a particular causal explanation such as reduced background reliance.

**Is the improvement certain?** No. The saved conditional intervals include zero for the relevant differences, and they do not repeat all training and selection steps. The result is an observed candidate comparison with limitations.

**Can it detect a healthy plant or reject an unrelated image?** That was not part of the trained label set. It may give an incorrect disease label confidently. Healthy/unknown data and a separately checked rejection rule are future work.

**What should I say if the live app is slow?** Explain that the hosted GPU service may have a queue. Show the interface and recorded verification if needed, and say clearly that the new request did not finish.

## Further reading

- [Vision Transformer paper](https://arxiv.org/abs/2010.11929)
- [EVA-02 paper](https://arxiv.org/abs/2303.11331)
- [MixUp paper](https://arxiv.org/abs/1710.09412)
- [CutMix paper](https://arxiv.org/abs/1905.04899)
- [Average precision definition used by the project](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [PyTorch training overview](https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [PyTorch data loading and samplers](https://docs.pytorch.org/docs/stable/data.html)
- [PyTorch randomness and seeds](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Google explanation of overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting)

## Slide-by-slide speaking notes

### 1. MROOT

Hi, I’m Armen. MROOT is my plant disease image classification project. I will explain how a model learns from photos, why I tried different training methods, and how I chose the final system. The result is a web app that suggests disease categories from an uploaded photo.

Source notes: Original student-supplied MROOT handover, including the clean ViT baseline, EVA Small seeds, previous failed experiments, historical Base results, data manifests, and V1 evaluation. Rounded historical scores are approximate. Reasons for old runs are reconstructed from the recorded setup where no original proposal exists.

### 2. A photo becomes a disease suggestion

The task is image classification: assigning a picture to one of a fixed set of categories. MROOT has 39 disease labels. A label is the name attached to a training photo, such as rust or septoria leaf spot. The app ranks five suggestions. It predicts the disease category, independently of the plant species. It has no healthy category, so it cannot reliably answer the broader question of whether any plant is healthy.

Source notes: Original student-supplied MROOT handover, including the clean ViT baseline, EVA Small seeds, previous failed experiments, historical Base results, data manifests, and V1 evaluation. Rounded historical scores are approximate. Reasons for old runs are reconstructed from the recorded setup where no original proposal exists. The photo is an original validation example labeled septoria_leaf_spot. It is not a newly generated image or a new model prediction.

### 3. What training a model means

A neural network is a model whose connected layers transform an image into scores. Its parameters, often called weights, are the internal numbers learned during training. A loss is a numerical penalty for a wrong or poor prediction. Training adjusts the parameters to reduce that penalty. Prediction uses the trained parameters to score a new photo. The app does not learn a new model every time someone uploads an image.

Source notes: Project training configuration and original recorded recipes. General training terminology: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html.

### 4. Why I kept separate photo sets

Training photos teach the model. Validation photos help me compare training choices without directly training on those photos. A test or holdout set is normally reserved for reporting after choices are fixed. Our final paired experiment used 6,790 training and 388 validation photos. Comparisons with older checkpoints also used a shared 368-image subset of validation. The final report used 623 held-out photos. That holdout had already been evaluated earlier, so it is not a completely new blind test. The original clean split had 6,870 training and 390 validation photos; I keep those counts with the earlier results.

Source notes: Project reports/raw/class_support.csv, eda_summary.json, split_summary.json, and exact data/manifests CSVs. Original handover explains the source-folder versus manifest distinction. Project reports/raw/FINAL_HOLDOUT_20261024T103253_167194Z/FINAL_HOLDOUT_RESULT.json and FROZEN_SELECTION.json. Comparison with the previous four-model result uses the same 623 images. This holdout was previously evaluated.

### 5. What Top-1 and Top-5 accuracy mean

These measures answer a direct user question. Did the model’s first answer match the label, or was the label at least included among its five suggestions? The example is only an illustration. A model can have strong Top-5 and still make noticeably more first-choice errors. That pattern motivated checking which diseases the model confused and whether combining models improved their ranking.

Source notes: Original student-supplied MROOT handover, including the clean ViT baseline, EVA Small seeds, previous failed experiments, historical Base results, data manifests, and V1 evaluation. Rounded historical scores are approximate. Reasons for old runs are reconstructed from the recorded setup where no original proposal exists. Definitions follow the project evaluation code. The second-place example is illustrative.

### 6. Macro mAP gives each disease equal importance

Average precision, or AP, measures ranking quality for one disease. Macro mAP averages AP over all 39 diseases with equal importance. In the toy table, a non-rust photo ranks above a rust photo, reducing the rust AP. This metric is not the fraction of photos with a correct first suggestion. I chose it because common classes should not dominate the score.

Source notes: Project metric code uses sklearn.metrics.average_precision_score. Definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html. Any three-photo example in this presentation is invented only to explain the metric.

### 7. Some diseases had far fewer training examples

A class is one of the categories the model can predict. In our final training data, powdery mildew had 755 photos while berry blotch had 68. A model could pay much more attention to common classes unless the training method accounts for this difference. That is called class imbalance. It explains why I used a metric that treats classes equally and tested different ways of balancing training. The chart shows a few examples from all 39 classes.

Source notes: Project reports/raw/class_support.csv, eda_summary.json, split_summary.json, and exact data/manifests CSVs. Original handover explains the source-folder versus manifest distinction.

### 8. Related photos can make an evaluation too easy

Matching file hashes finds identical bytes, but a resized copy can have a different hash. If a training photo and its close copy appear in validation, the score may look better than performance on genuinely new photos. Review also found a related photo with conflicting disease labels. We used new manifests, meaning CSV lists of the files allowed in each split, to quarantine detected groups. The final pair used 80 fewer training rows and two fewer validation rows. The original files remained intact. The checks reduced detected overlap; they cannot prove that every possible relationship was removed.

Source notes: Project reports/raw/class_support.csv, eda_summary.json, split_summary.json, and exact data/manifests CSVs. Original handover explains the source-folder versus manifest distinction. Initial and reviewed audit findings in docs/experiments.md.

### 9. I adapted models that already knew visual patterns

I did not train a large image model from zero using only a few thousand photos. I started with pretrained models and fine-tuned them. ViT stands for Vision Transformer. It represents an image through patches and combines information across those patches. EVA-02 is the particular pretrained Transformer family used in MROOT. Small and Base are different model sizes. A larger model has more learned parameters and costs more to run, but it still needs a measured benefit on our task.

Source notes: Project handover and model identifiers. ViT: https://arxiv.org/abs/2010.11929. EVA-02: https://arxiv.org/abs/2303.11331.

### 10. Why combining models helped at first

An ensemble combines predictions from several trained models. A random seed controls random choices in training, so the same general recipe can produce models with different errors. The original project trained three separate EVA Small models. Their disease probabilities were averaged. The historical clean ViT baseline was about 90.64% macro mAP, two EVA models reached about 93.53%, and the three-model result was 94.72% on the original 390 validation photos. I reproduced that saved reference before trying changes. Averaging helps only when the predictions combine usefully.

Source notes: Original student-supplied MROOT handover, including the clean ViT baseline, EVA Small seeds, previous failed experiments, historical Base results, data manifests, and V1 evaluation. Rounded historical scores are approximate. Reasons for old runs are reconstructed from the recorded setup where no original proposal exists.

### 11. More training and more models did not always help

A fourth Small model used the same general recipe with different training randomness, but it lowered the ensemble score to about 94.49%. Continuing the first model’s training did not beat its earlier saved result. ConvNeXt V2 Tiny was another image-model architecture, using learned local visual filters instead of the Transformer approach. It reached about 90.54% alone and hurt the EVA combination. These failures kept me from treating more training or more models as automatic progress. The motivations for these historical runs are reconstructed from the setup in the handover.

Source notes: Original student-supplied MROOT handover, including the clean ViT baseline, EVA Small seeds, previous failed experiments, historical Base results, data manifests, and V1 evaluation. Rounded historical scores are approximate. Reasons for old runs are reconstructed from the recorded setup where no original proposal exists.

### 12. Two ways to give rare diseases more attention

The sampler decides which training examples are drawn. With replacement sampling, the same rare photo may appear several times while another photo is skipped in that round. A loss tells training how costly its mistakes are. I tested ordinary shuffled sampling and moved the class balancing into the loss penalty. This was a combined sampling-and-loss change, not two separately isolated effects. The new Small model added only about 0.13 percentage points to the original ensemble on the same 390 validation photos. That was a small candidate improvement, not evidence that balancing was solved.

Source notes: docs/experiments.md and the saved natural-sampling experiment. Original 390-photo validation.

### 13. Could a larger model use finer image detail?

Resolution is the number of input pixels across and down. Increasing it may preserve detail that was harder to see at a smaller input size. I also changed the model size, so this experiment cannot identify a resolution-only effect. Earlier parts of the pretrained model received smaller updates than later parts, to limit how much previous learning was disturbed. This is called layer-wise learning-rate decay. The larger Base contributed usefully in some combinations. Its cost and its contribution both mattered, rather than its size alone.

Source notes: docs/experiments.md and original Base configuration. EVA-02: https://arxiv.org/abs/2303.11331.

### 14. Smaller training adjustments helped one model

A learning rate controls how large an adjustment training makes to its parameters. Large adjustments may move a useful pretrained solution too much, while smaller ones can adapt more gently. I tested smaller rates rather than assuming they would help. The gentler Base improved by itself under the recorded three-view prediction recipe. However, replacing the earlier Base in a predefined ensemble reduced that combination from about 95.12% to 94.92% on the same 390 validation photos. I therefore kept checking how predictions worked together, not just which model had the highest individual score.

Source notes: reports/raw/gentler_base_result.json and docs/experiments.md. The standalone scores shown use the recorded three-view prediction recipe.

### 15. A fair comparison needs a matched reference

After the image-group review changed the split, I needed a fresh reference trained on that same split. Otherwise, a difference could come from cleaner data rather than the proposed training method. A control is the matched reference for this comparison. Both final models used the same 6,790 training and 388 validation photos and the same saved starting parameters. The candidate added MixUp and CutMix as one policy. I can compare that combined policy with the control, but I cannot claim which mixing method caused the difference on its own. I ran one seed per arm, so repeated-seed variation remains unknown.

Source notes: Project reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, mixup_cutmix/experiment_config.json, paired_validation_intervals.json, and docs/experiments.md. These are saved results, not new evaluations.

### 16. Augmentation creates varied training examples

Data augmentation changes training inputs so the model sees variation. A flip or a modest crop can change the picture while keeping the labeled disease visible. The changes need care: an aggressive crop can remove a lesion, and large color changes can remove useful symptoms. Our ordinary augmentations were already in the shared recipe. The last experiment added a mixing policy to test whether stronger variation could help. It did not prove that the model stopped using backgrounds or memorizing photos; that was the hypothesis being tested.

Source notes: Recorded original and final augmentation configurations. MixUp/CutMix motivation: Zhang et al., https://arxiv.org/abs/1710.09412; Yun et al., https://arxiv.org/abs/1905.04899. Actual MROOT outcome is from the saved final controlled comparison.

### 17. MixUp blends two examples and their labels

MixUp uses a blended pair of images and correspondingly blended targets. The 70/30 example is invented to explain the idea, not an actual evaluation result. The model must learn from combinations instead of relying only on complete familiar photos. During normal prediction, the user still uploads an ordinary photo. The public app does not blend the user’s photo with another image.

Source notes: Zhang et al., https://arxiv.org/abs/1710.09412; Yun et al., https://arxiv.org/abs/1905.04899. Actual MROOT outcome is from the saved final controlled comparison.

### 18. CutMix replaces a region with part of another photo

CutMix changes a region rather than blending every pixel. Its targets are mixed according to patch area. For disease photos this might encourage the model to use more than one local cue, but it can also hide important symptom detail. That is why it needed a validation comparison. Our experiment tested a combined MixUp/CutMix policy. It was not a separate proof of MixUp’s effect and another proof of CutMix’s effect.

Source notes: Zhang et al., https://arxiv.org/abs/1710.09412; Yun et al., https://arxiv.org/abs/1905.04899. Actual MROOT outcome is from the saved final controlled comparison.

### 19. Results of the matched comparison

On the same 388 validation photos, the control scored 94.81% macro mAP and the mixed-image candidate scored 96.07% with one standard prediction view. This was an observed result from a matched comparison. Both models trained only on the training split. I saved the best version using validation rather than simply using the last training version. The result supports keeping the candidate for ensemble checks. It does not prove that the policy will improve every seed or a completely new data source.

Source notes: Project reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, mixup_cutmix/experiment_config.json, paired_validation_intervals.json, and docs/experiments.md. These are saved results, not new evaluations.

### 20. At prediction time, I also average several views

Inference means using a trained model to make a prediction. Test-time augmentation repeats inference for different versions of the same input. It is separate from creating varied examples during training. The padded view preserves the full image and adds a plain border to make it square. We used the recorded normal, flipped, and padded recipe so the deployment matched evaluation. It can help some models, but it did not improve every metric in every comparison. The exact view weights are in the implementation rather than on the public slides.

Source notes: ensemble.json; src/mroot/inference.py; saved normal-view and fixed-view comparisons.

### 21. Why I chose a two-model ensemble

For comparison with older models, I used the same shared 368-photo subset of validation. The mixed-image model alone scored 96.03% macro mAP with the recorded multiple-view recipe. Combining it with the clean control scored 96.48%. Adding an earlier Base or a newer Small did not beat that pair. I chose by macro mAP, the primary metric. A single mixed model with standard single-view prediction had higher Top-1, so the pair was not best on every metric. An uncertainty range estimated by resampling these validation photos includes no improvement, so the small pair-versus-single advantage is uncertain. This is the highest observed mAP among the final comparisons, not a guarantee of an improvement on unseen sources.

Source notes: Project reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, mixup_cutmix/experiment_config.json, paired_validation_intervals.json, and docs/experiments.md. These are saved results, not new evaluations.

### 22. Final reporting on 623 held-out photos

The final pair classified 557 of 623 photos correctly on its first choice, which is 89.41% Top-1 accuracy. It included the true disease in the first five suggestions for 619 photos, or 99.36% Top-5. Its macro mAP was 92.99%. On the same 623 photos, the previous four-model system had 549 correct first choices, so the final pair added eight. This holdout had already been evaluated earlier. We froze the final selection before its last evaluation, but that does not make it a completely new independent test. It also cannot be compared directly with the old V1 score on 670 photos as if the sample sets were identical.

Source notes: Project reports/raw/FINAL_HOLDOUT_20261024T103253_167194Z/FINAL_HOLDOUT_RESULT.json and FROZEN_SELECTION.json. Comparison with the previous four-model result uses the same 623 images. This holdout was previously evaluated.

### 23. Using the trained model in a web app

Deployment means putting the trained model into an application other people can use. In the V2 Hugging Face Space, the user uploads a plant photo and clicks Analyze photo. The app returns the five ranked disease scores. The original V1 app remains available. The V2 model files and prediction recipe were checked against the saved versions. If the online service is unavailable during the demonstration, I can show the interface and saved verification, while saying clearly that the live request did not complete.

Source notes: https://huggingface.co/spaces/Armennnn/mroot-plant-disease-classifier-v2 and reports/live_deployment_verification.json.

### 24. What these results still cannot establish

Generalization is the ability to work on new examples from the intended use setting. Overfitting is one way that ability can fail. Our validation set is small, and repeated experiment decisions can adapt to its quirks even when validation photos are not used in gradient updates. The old holdout was also exposed earlier. Some class labels overlap visually or remain ambiguous. The model’s output score is not a verified probability that a diagnosis is correct. These limitations are why I would next collect independently reviewed external photos and add healthy or unrelated examples before considering a rejection rule.

Source notes: Project evaluation history and limitations in docs/evaluation.md and original handover.

### 25. What I learned from the project

My main lesson was to turn each training idea into a question I could actually test. Balancing rare classes, increasing the model and image size, reducing adjustment size, and mixing training examples were different hypotheses. Some helped a model alone but not the ensemble. Some failed. The final result is a working app and a recorded experiment history. The next useful improvement is stronger evaluation evidence, rather than simply making the ensemble larger.

Source notes: Project experiment record. Proposed future work is not completed work.

### 26. Earlier experiments from the project history (optional)

This slide keeps the earlier history available for questions. It separates measured historical outcomes from possible explanations. The original successful Base checkpoint was lost, and a later rebuild did not reproduce its score. A checkpoint is simply a file containing a saved model state. The loss of that file means the old score cannot serve as a currently reloadable final model. The present selected checkpoints are backed up. None of these historical figures are new reproductions in this presentation.

Source notes: Original student-supplied MROOT handover, including the clean ViT baseline, EVA Small seeds, previous failed experiments, historical Base results, data manifests, and V1 evaluation. Rounded historical scores are approximate. Reasons for old runs are reconstructed from the recorded setup where no original proposal exists.

### 27. A small test of ensemble voting weights (optional)

These voting weights are different from a model’s learned internal weights. I tried only four choices: equal thirds and each model getting half while the others got one quarter. Five-fold cross-fitting means choosing a candidate using four validation parts and evaluating it on the fifth, then rotating that held-out part. It reused saved predictions and did not train five neural networks. The pooled score was worse than equal weights. Broadly searching 103 subsets of an earlier prediction pool was also exploratory, and cross-fitting that selection was unstable. A favorable full-validation search result is not automatically an independent improvement.

Source notes: configs/ensemble_weight_grid.json; scripts/crossfit_weights.py; docs/experiments.md. Original 390-photo validation.

### 28. Averaging parameters is different from averaging predictions (optional)

EMA is an average of parameters within a training run. A prediction ensemble combines outputs from trained models. They are different operations. I checked the EMA version at the same validation-selected training point as the raw gentler Base model. It did not beat the raw model in that check, so it was not used in the final pair. This applies to the tested setup and does not establish that EMA always fails.

Source notes: reports/raw/gentler_base_result.json; docs/experiments.md. Method implementation: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py.

### 29. The new training comparisons in one place (optional)

The validation count is essential when comparing these values. The first three candidates use the original 390-photo validation set. The last pair uses 388. The single-view column uses one standard input view. The multiple-view column uses the recorded normal, flipped, and padded prediction average. These scores refer to each selected model, not the final ensemble. Exact training steps, optimizer settings, and saved file identifiers remain in the reports. Only measured completed experiments appear here. Other proposed methods remain future ideas.

Source notes: Project reports/raw/COMPLETED_COMPARISON.json, control/experiment_config.json, mixup_cutmix/experiment_config.json, paired_validation_intervals.json, and docs/experiments.md. These are saved results, not new evaluations. docs/experiments.md.

### 30. Project evidence and method sources (optional)

The experiment outcomes come from the original project handover and saved MROOT reports. The method references explain the model family and training ideas. Their published results do not replace our own measurements. The supplied ACA guidelines shaped the data exploration, experimentation, deployment, repository, and presentation coverage.

Source notes: Original MROOT handover; ML Engineer Project Guidelines.pdf; linked primary references and project artifacts.
