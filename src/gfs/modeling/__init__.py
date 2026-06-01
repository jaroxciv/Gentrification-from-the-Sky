"""Modeling — Predicting gentrification.

Aggregates change-detection outputs to LSOA-level feature vectors (paper §3.5),
binarizes the gentrification score into a classification target (§4.1), and
trains/evaluates baseline and satellite-enhanced classifiers — Logistic
Regression, Linear SVC, XGBoost — plus the thresholding ablation (§4–5).
"""
