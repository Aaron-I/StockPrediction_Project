# src/evaluation/evaluate.py
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def evaluate_model(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    return metrics
