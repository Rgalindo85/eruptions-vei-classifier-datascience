import matplotlib.pyplot as plt
import seaborn as sns



def classification_report_plot(report, title='Classification Report', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(report, annot=True, fmt='.2f', cmap=cmap)
    
    plt.title(title)
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.tight_layout()
    
    return plt

def confusion_matrix_plot(cm, class_names, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    return plt