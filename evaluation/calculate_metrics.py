#!/usr/bin/env python3
"""
Calculate evaluation metrics from the results file.
Metrics: Accuracy, F1 Score, AUROC, AUPRC
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report


def calculate_metrics(input_file, output_file):
    """Calculate and save evaluation metrics."""
    
    # Read results
    df = pd.read_csv(input_file, sep='\t')
    
    # Handle both old and new column names
    if 'is_supported_from_llm' in df.columns:
        predicted_col = 'is_supported_from_llm'
    else:
        predicted_col = 'predicted'
    
    if 'ground_truth' not in df.columns:
        print("Error: 'ground_truth' column not found!")
        return
    
    # Filter out rows with Unknown or Error predictions
    df_valid = df[~df[predicted_col].isin(['Unknown', 'Error'])].copy()
    
    if len(df_valid) == 0:
        print("Error: No valid predictions found!")
        return
    
    # Convert string True/False to binary (handle various formats)
    # Normalize to handle True/TRUE/true and False/FALSE/false
    df_valid['ground_truth_str'] = df_valid['ground_truth'].astype(str).str.strip().str.lower()
    df_valid['predicted_str'] = df_valid[predicted_col].astype(str).str.strip().str.lower()
    
    # Map to binary
    df_valid['ground_truth_binary'] = df_valid['ground_truth_str'].map({'true': 1, 'false': 0})
    df_valid['predicted_binary'] = df_valid['predicted_str'].map({'true': 1, 'false': 0})
    
    # Debug: Show unique values before filtering
    print(f"Unique ground_truth values: {df_valid['ground_truth'].unique()}")
    print(f"Unique {predicted_col} values: {df_valid[predicted_col].unique()}")
    
    # Remove any rows with NaN values
    df_valid = df_valid.dropna(subset=['ground_truth_binary', 'predicted_binary'])
    
    if len(df_valid) == 0:
        print("Error: No valid binary predictions after conversion!")
        print("This usually means the True/False values are not being recognized.")
        return
    
    # Extract ground truth and predictions
    y_true = df_valid['ground_truth_binary'].values
    y_pred = df_valid['predicted_binary'].values
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # AUROC and AUPRC require probability scores, but we only have binary predictions
    # So we'll use the binary predictions as proxy scores
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auroc = None
        
    try:
        auprc = average_precision_score(y_true, y_pred)
    except ValueError:
        auprc = None
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Create metrics report
    report = f"""
========================================
Evaluation Metrics Report
========================================

Dataset Information:
-------------------
Total samples: {len(df)}
Valid predictions: {len(df_valid)}
Unknown predictions: {len(df) - len(df_valid)}

Class Distribution (Ground Truth):
---------------------------------
True (Supported): {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.2f}%)
False (Not Supported): {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.2f}%)

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
    
    # Add sklearn classification report
    report += classification_report(y_true, y_pred, target_names=['Not Supported', 'Supported'])
    
    report += """
========================================
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(report)
    
    # Also print to console
    print(report)
    
    # Save metrics as JSON for easier parsing
    import json
    metrics_dict = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'auroc': float(auroc) if auroc is not None else None,
        'auprc': float(auprc) if auprc is not None else None,
        'total_samples': int(len(df)),
        'valid_predictions': int(len(df_valid)),
        'unknown_predictions': int(len(df) - len(df_valid)),
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

