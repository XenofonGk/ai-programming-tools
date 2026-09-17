# ai-programming-tools

A small collection from working through applied-AI coursework and exercises: one working ML project, plus lab notes on prompting and licensing.

## AI Resume Classifier

The main piece here. A logistic regression model that sorts resume text into one of five job families (Cybersecurity, Data Science, IT Support, Marketing, Software Engineering) using TF-IDF features, trained on a synthetic dataset built specifically to avoid being trivially separable by keyword.

```bash
cd "AI Resume Classifier (NLP + ML)"
pip install -r requirements.txt
python data/generate_dataset.py
python train.py
python predict.py
```

`train.py` gets 97.3% holdout accuracy and 98.3% 5-fold cross-validated accuracy on the generated data — see [the project's own README](<AI Resume Classifier (NLP + ML)/README.md>) for what that number does and doesn't mean, and for an honest list of what it would take to make this work on real resumes.

## Other notes

- [`Effective AI Prompting.md`](Effective%20AI%20Prompting.md) — exercises comparing vague vs. specific prompts across a few scenarios.
- [`How a Neural Network Learns...md`](<How a Neural Network Learns: Simulating Weight Adjustment and Decision Making.md>) — a walkthrough of manually adjusting weights over a small set of training rounds.
- [`SoftwareLicensing.md`](SoftwareLicensing.md) — lab notes on identifying and comparing open-source license terms.

These are coursework notes, not polished guides — kept here because they're a reasonably useful reference for myself later.

---
*Xenofon Gkioka*
