# Development-of-a-Deep-Learning-Based-Classification-ADHD-
AI system for early ADHD detection in children by combining neurophysiological signals and eye-tracking data. Applied machine learning and deep learning models with multimodal fusion to classify attention patterns and improve early, non-clinical detection accuracy.

Multimodal AI for ADHD Detection

This repository contains all files and scripts developed for my master’s dissertation project, titled “Multimodal AI for ADHD Detection.”
The project explores how machine learning and deep learning techniques can be applied to multimodal datasets (e.g., EEG, EYE, EDA) to improve detection accuracy and understanding of ADHD.

Project Structure

Deep Learning ADHD.py

Python script implementing the deep learning pipeline for ADHD detection.
Key features:

Loads multimodal datasets.

Builds and trains deep neural networks (LSTM architecture).

Evaluates model performance.
Saves trained models and visualizes results.

EDA.Rmd

R Markdown file performing exploratory data analysis (EDA).
Includes:

Data loading, cleaning, and structure inspection.

Visualization of feature distributions and correlations.

Statistical summaries comparing ADHD and control groups.

Identification of outliers and missing data patterns.

Feature Engineering.

Merge Data.Rmd

R Markdown file handling data integration and preprocessing.
Functions:

Loads datasets from multiple modalities (EEG, EYE, EDA).

Aligns and merges them by participant ID.

Normalizes and encodes variables.

Exports the unified dataset for both R- and Python-based modeling.

Labels ADHD and Control participants

ML Models.Rmd

R Markdown file implementing traditional machine learning models for baseline comparison.
Includes:

Training and evaluation of SVM, Random Forest and XGBoost.

K-fold cross-validation and hyperparameter tuning.

Feature importance analysis.


Multimodal AI ADHD.docx

Microsoft Word document — the main dissertation report.
Contains:

Literature review and background research.

Methodology (data, models, preprocessing).

Experimental results from both R and Python analyses.

Discussion, ethical considerations, and future work.
