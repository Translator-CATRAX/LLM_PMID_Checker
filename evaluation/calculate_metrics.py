#!/usr/bin/env python3
"""
Calculate evaluation metrics from the results file.
Metrics: Accuracy, F1 Score, AUROC, AUPRC
"""

import sys
import json
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report


def calculate_metrics(input_file, output_file):
    """Calculate and save evaluation metrics."""
    
    df = pl.read_csv(input_file, separator='\t', infer_schema_length=0)
    
    if 'is_supported_from_llm' in df.columns:
        predicted_col = 'is_supported_from_llm'
    else:
        predicted_col = 'predicted'
    
    if 'ground_truth' not in df.columns:
        print("Error: 'ground_truth' column not found!")
        return
    
    df_valid = df.filter(~pl.col(predicted_col).is_in(['Unknown', 'Error']))
    
    if df_valid.height == 0:
        print("Error: No valid predictions found!")
        return
    
    gt_str = df_valid['ground_truth'].cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    pred_str = df_valid[predicted_col].cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    
    gt_binary = gt_str.replace_strict({'true': '1', 'false': '0'}, default=None).cast(pl.Int8, strict=False)
    pred_binary = pred_str.replace_strict({'true': '1', 'false': '0'}, default=None).cast(pl.Int8, strict=False)
    
    print(f"Unique ground_truth values: {df_valid['ground_truth'].unique().to_list()}")
    print(f"Unique {predicted_col} values: {df_valid[predicted_col].unique().to_list()}")
    
    mask = gt_binary.is_not_null() & pred_binary.is_not_null()
    gt_binary = gt_binary.filter(mask)
    pred_binary = pred_binary.filter(mask)
    
    if gt_binary.len() == 0:
        print("Error: No valid binary predictions after conversion!")
        print("This usually means the True/False values are not being recognized.")
        return
    
    y_true = gt_binary.to_numpy()
    y_pred = pred_binary.to_numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = None
        
    try:
        auprc = average_precision_score(y_true, y_pred)
    except ValueError:
        auprc = None
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    valid_count = int(gt_binary.len())
    total_count = df.height
    
    report = f"""
========================================
Evaluation Metrics Report
========================================

Dataset Information:
-------------------
Total samples: {total_count}
Valid predictions: {valid_count}
Unknown predictions: {total_count - valid_count}

Class Distribution (Ground Truth):
---------------------------------
True (Supported): {int(sum(y_true == 1))} ({sum(y_true == 1)/len(y_true)*100:.2f}%)
False (Not Supported): {int(sum(y_true == 0))} ({sum(y_true == 0)/len(y_true)*100:.2f}%)

Performance Metrics:
-------------------
Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
F1 Score:  {f1:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
Specificity: {specificity:.4f}
"""
    
    if auroc is not None:
        report += f"AUROC:     {auroc:.4f}\n"
    else:
        report += "AUROC:     N/A (requires varying prediction scores)\n"
        
    if auprc is not None:
        report += f"AUPRC:     {auprc:.4f}\n"
    else:
        report += "AUPRC:     N/A (requires varying prediction scores)\n"
    
    report += f"""
Confusion Matrix:
----------------
                Predicted Negative  Predicted Positive
Actual Negative        {tn:6d}              {fp:6d}
Actual Positive        {fn:6d}              {tp:6d}

Detailed Classification Report:
------------------------------
"""
    
    report += classification_report(y_true, y_pred, target_names=['Not Supported', 'Supported'])
    
    report += """
========================================
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(report)
    
    metrics_dict = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'auroc': float(auroc) if auroc is not None else None,
        'auprc': float(auprc) if auprc is not None else None,
        'total_samples': total_count,
        'valid_predictions': valid_count,
        'unknown_predictions': total_count - valid_count,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nMetrics also saved to {json_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_metrics.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    calculate_metrics(input_file, output_file)
