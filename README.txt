FAKE NEWS CLASSIFIER
====================

Project Overview
----------------
This repository contains a complete, end-to-end pipeline for classifying news articles as **fake** or **real** using classical machine-learning techniques in **scikit-learn**. The project is intended as a compact, educational example that demonstrates:

* data acquisition
* data cleaning and preprocessing
* feature engineering with TF-IDF
* model selection with cross-validation
* final evaluation on a held-out test set
* model serialisation for later inference

Dataset
-------
**Fake and Real News Dataset** – Kaggle: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

```
data/raw/Fake.csv   # fake news articles
data/raw/True.csv   # real news articles
```
The CSV files are **not** tracked in Git (see `.gitignore`). Download them manually from Kaggle and place them in `data/raw/`.

Requirements
------------
* Python ≥ 3.9
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* jupyterlab

Installation
------------
```bash
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate.bat      # Windows
pip install -r requirements.txt
```

Running the notebook
--------------------
```bash
jupyter lab
```
Then open **`notebooks/fakenewsclassifier.ipynb`** and run all cells.

Quick CLI demo
--------------
```bash
python src/train.py                       # trains the model and saves model.joblib
python src/predict.py "Some news text…"    # prints the predicted label + probability
```

Repository layout
-----------------
```
├── data/
│   └── raw/               # original CSVs from Kaggle (not committed)
├── notebooks/
│   └── fakenewsclassifier.ipynb
├── src/
│   ├── train.py           # training script
│   └── predict.py         # simple inference script
├── requirements.txt
├── .gitignore
└── README.txt
```

Results
-------
Best model (Gradient Boosting, **n_estimators=150**, **learning_rate=0.05**, **max_depth=3**):

* Cross-validation (5-fold, F1-macro): **0.93 ± 0.01**
* Test set F1-macro: **0.92**

Future Work
-----------
* Grid search for TF-IDF n-gram ranges
* Experiments with transformer-based models (e.g. BERT)
* Tracking experiments and metrics in MLflow

License
-------
Code – MIT License  •  Dataset – see Kaggle page for terms.

Author
------
Your Name Here  (2025)
