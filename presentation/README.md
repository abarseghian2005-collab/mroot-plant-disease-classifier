# ACA presentation and demo plan

Use slides 1-14 for the talk. Slides 15-20 are backup material for questions.
The PowerPoint includes editable charts/tables and speaking notes; the separate
notes file is easier to rehearse from.

Suggested timing:

- Problem, data, and starting model: about 3 minutes.
- Experiments, final pair, and results: about 4 minutes.
- Live app: about 1 minute.
- Limits and what I learned: about 2 minutes.

Before presenting, open the V2 Space and keep one suitable plant photo ready.
Upload the image, click Analyze photo, and explain the five ranked diseases.
Explain that a softmax score is not a calibrated diagnosis and the model has no
healthy or unknown category. If the service is sleeping or the queue is long,
use the app screenshot and the recorded single-image verification result rather
than claiming that the live request succeeded.

The EDA and experiment notebooks already contain executed analysis outputs.
The training and inference notebooks deliberately leave their optional run
switches disabled. A presentation should not start training or rerun the holdout.

Use the 96.48% number as validation macro mAP and the 92.99% number as macro mAP
on the fixed 623-image reporting subset. Do not call either number accuracy.
The 623-image subset was evaluated earlier; a new untouched external set would
be a stronger generalization check.
