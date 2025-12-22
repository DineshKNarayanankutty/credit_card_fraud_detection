"""
Reporting layer.
Responsibility: Generate human-readable summaries.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def generate_classification_report(metrics: Dict[str, float]) -> str:
    """
    Generate classification metrics report.
    
    Args:
        metrics: Dict from compute_classification_metrics()
    
    Returns:
        Formatted string report
    """
    report = """
====================================
CLASSIFICATION METRICS
====================================
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}
MCC:       {mcc:.4f}
====================================
""".format(**metrics)
    
    return report


def generate_confusion_matrix_report(cm_metrics: Dict[str, Any]) -> str:
    """
    Generate confusion matrix report.
    
    Args:
        cm_metrics: Dict from compute_confusion_matrix_metrics()
    
    Returns:
        Formatted string report
    """
    report = """
====================================
CONFUSION MATRIX & DERIVED METRICS
====================================
True Positives:   {tp:>6}
True Negatives:   {tn:>6}
False Positives:  {fp:>6}
False Negatives:  {fn:>6}

Sensitivity (Recall): {sensitivity:.4f}
Specificity:          {specificity:.4f}
False Positive Rate:  {fpr:.4f}
False Negative Rate:  {fnr:.4f}
====================================
""".format(**cm_metrics)
    
    return report


def generate_probabilistic_report(metrics: Dict[str, float]) -> str:
    """
    Generate probabilistic metrics report.
    
    Args:
        metrics: Dict with roc_auc and pr_auc
    
    Returns:
        Formatted string report
    """
    report = """
====================================
PROBABILISTIC METRICS
====================================
ROC-AUC:  {roc_auc:.4f}
PR-AUC:   {pr_auc:.4f}
====================================
""".format(**metrics)
    
    return report


def generate_threshold_report(threshold_analysis: Dict) -> str:
    """
    Generate threshold analysis report.
    
    Args:
        threshold_analysis: Dict from get_threshold_analysis()
    
    Returns:
        Formatted string report
    """
    report = """
====================================
THRESHOLD ANALYSIS
====================================
"""
    
    report += f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n"
    report += "─" * 50 + "\n"
    
    for threshold, metrics in sorted(threshold_analysis.items()):
        report += f"{threshold:<12.2f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}\n"
    
    report += "====================================\n"
    
    return report


def generate_full_report(
    classification_metrics: Dict[str, float],
    cm_metrics: Dict[str, Any],
    prob_metrics: Dict[str, float] = None,
    threshold_analysis: Dict = None
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        classification_metrics: Classification metrics dict
        cm_metrics: Confusion matrix metrics dict
        prob_metrics: Probabilistic metrics dict (optional)
        threshold_analysis: Threshold analysis dict (optional)
    
    Returns:
        Formatted string report
    """
    report = """
====================================
FRAUD DETECTION MODEL REPORT
====================================
"""
    
    report += generate_classification_report(classification_metrics)
    report += generate_confusion_matrix_report(cm_metrics)
    
    if prob_metrics:
        report += generate_probabilistic_report(prob_metrics)
    
    if threshold_analysis:
        report += generate_threshold_report(threshold_analysis)
    
    report += """
====================================
END OF REPORT
====================================
"""
    
    return report


def log_report(report: str, log_level: str = "INFO"):
    """
    Log report to logger.
    
    Args:
        report: Report string
        log_level: Logging level
    """
    log_func = getattr(logger, log_level.lower(), logger.info)
    for line in report.split("\n"):
        log_func(line)
