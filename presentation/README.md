# ACA presentation and demo plan

Use MROOT_ACA_Detailed_Presentation.pptx for the expanded explanation. It contains
30 main slides and six appendix slides, with a full talk of about 20-25 minutes.
For a talk of about ten minutes, use slides 1, 3, 5, 6, 12, 14, 16, 17, 22, 24,
26, 27, 29, and 30. The detailed_speaking_notes.md file explains each slide.
The earlier 20-slide compact deck and its notes remain in this folder.

The PowerPoint includes editable charts/tables and speaking notes.

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
