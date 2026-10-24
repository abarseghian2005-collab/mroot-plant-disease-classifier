# ACA capstone checklist

This maps the supplied ML Engineer Project Guidelines to the project artifacts.
It describes what exists without assigning a grade.

| Requirement | Project evidence | Qualification |
| --- | --- | --- |
| Dedicated EDA | `notebooks/01_eda.ipynb`, image metadata and split checks | No holdout reads in the EDA update |
| Baseline and model comparisons | Experiment log, original configs/histories, `02_experiment_review.ipynb` | Historical rounded results are labeled separately |
| Hyperparameter tuning | Controlled LR comparison, small cross-fitted ensemble-weight grid, configs and utility | No broad automated neural-network search or five-fold retraining claimed |
| Working UI/API | Live Gradio Space, local FastAPI entry point and tests | Live inference tested separately from mock API tests |
| Modular public repository | `src/`, `scripts/`, `data/`, `notebooks/`, README and requirements | Model weights/images stay outside Git |
| Presentation and sources | ACA slide deck, speaking notes, experiment references | Main talk is planned for about ten minutes |
| Docker bonus | Dockerfile and run instructions | Build not verified in this desktop environment |
| CI bonus | GitHub Actions workflow | Check its real run status after publication |
| Data validation bonus | Pydantic manifest validation and image/API validation | Tests cover invalid input and split integrity |
| MLflow bonus | Not implemented | Existing runs use CSV/JSON records |
| Ten-minute video bonus | Speaking plan and demo checklist | Recording still needs the student |

The rubric asks for an untouched test evaluated exactly once. This project's
historical holdout was evaluated earlier, so it does not satisfy that ideal
protocol. The presentation and README disclose this limitation. A new external
test set would be needed to close that gap.
