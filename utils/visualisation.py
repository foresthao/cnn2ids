from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrix(y_true, y_pred, labels, save=False, save_dir=None, filename=None):
    """Plot normalised confusion matrix"""

    # Set font sizes for better readability
    plt.rcParams.update({'font.size': 14})
    
    confusion_mtx = confusion_matrix(y_true, y_pred)
    precision_confusion_mtx = confusion_mtx.T / (confusion_mtx.sum(axis=1)).T
    recall_confusion_mtx = confusion_mtx / confusion_mtx.sum(axis=0)

    fig = plt.figure(figsize=(24, 8))

    plt.subplot(1, 3, 1)
    _ = sns.heatmap(confusion_mtx, annot=True, cmap="Blues", fmt="", xticklabels=labels, yticklabels=labels, 
                   annot_kws={'size': 12})
    plt.xlabel("Predicted label", fontsize=16, fontweight='bold')
    plt.ylabel("True label", fontsize=16, fontweight='bold')
    plt.title("Confusion Matrix", fontsize=18, fontweight='bold')

    plt.subplot(1, 3, 2)
    _ = sns.heatmap(precision_confusion_mtx, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels,
                   annot_kws={'size': 12})
    plt.xlabel("Predicted label", fontsize=16, fontweight='bold')
    plt.ylabel("True label", fontsize=16, fontweight='bold')
    plt.title("Precision Matrix", fontsize=18, fontweight='bold')

    plt.subplot(1, 3, 3)
    _ = sns.heatmap(recall_confusion_mtx, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels,
                   annot_kws={'size': 12})
    plt.xlabel("Predicted label", fontsize=16, fontweight='bold')
    plt.ylabel("True label", fontsize=16, fontweight='bold')
    plt.title("Recall Matrix", fontsize=18, fontweight='bold')

    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')


def plot_roc_curve(y_test, y_score, labels, save=False, save_dir=None, filename=None):

    n_classes = y_score.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # 设置标签的字体大小
    fig = plt.figure(figsize=(16, 12))
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.4f})'.format(roc_auc["micro"]), linewidth=3)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'.format(labels[i], roc_auc[i]), linewidth=2.5)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    plt.title('ROC Curve', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')


def plot_precision_recall_curve(y_test, y_score, labels, save=False, save_dir=None, filename=None):

    n_classes = y_score.shape[1]

    precision = dict()
    recall = dict()

    # Plot Precision-Recall curve
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(16, 12))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2.5, label='Precision-Recall for {} class'.format(labels[i]))

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=18, fontweight='bold')
    plt.ylabel('Precision', fontsize=18, fontweight='bold')
    plt.title('Precision vs. Recall Curve', fontsize=20, fontweight='bold')
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
