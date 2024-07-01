import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from itertools import cycle

def load_spectrogram_files(spectrogram_path):
    file_paths = []
    labels = []
    for root, dirs, files in os.walk(spectrogram_path):
        for file in files:
            if file.endswith(".png"):
                label = os.path.basename(root)
                spectrogram_file_path = os.path.join(root, file)
                file_paths.append(spectrogram_file_path)
                labels.append(label)
                print(f"Found file: {spectrogram_file_path} with label: {label}")

    print(f"Total files loaded: {len(file_paths)}")
    return file_paths, labels

def plot_class_distribution(labels, unique_labels):
    label_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=unique_labels, y=label_counts)
    plt.xticks(rotation=90)
    plt.title("Class Distribution in Dataset")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

def plot_roc_curves(true_labels, predictions, n_classes, unique_labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    valid_classes = []

    for i in range(n_classes):
        if np.sum(true_labels == i) > 0:
            fpr[i], tpr[i], _ = roc_curve(true_labels == i, predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            valid_classes.append(i)

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'purple'])
    for i, color in zip(valid_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {unique_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Each Class')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("roc_curves.png")
    plt.show()

def main():
    spectrogram_path = "C:/Users/tring/Desktop/archive/spectrograms"
    model_save_path = "C:/Users/tring/Desktop/archive/fine_tuned_model_resnet.h5"
    
    spectrogram_files, labels = load_spectrogram_files(spectrogram_path)

    if not spectrogram_files:
        print("No spectrogram files found.")
        return

    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indexed_labels = [label_to_index[label] for label in labels]

    if len(unique_labels) <= 1:
        print("Error: The number of unique labels must be greater than 1 for multi-class classification.")
        return

    _, test_files, _, test_labels = train_test_split(spectrogram_files, indexed_labels, test_size=0.2, random_state=42)

    img_size = (224, 224)
    batch_size = 32

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_df = pd.DataFrame({'filename': test_files, 'class': test_labels})

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )

    print(f"Loading the model from {model_save_path}...")
    model = tf.keras.models.load_model(model_save_path)
    print("Model loaded successfully.")

    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    print("Making predictions on the test set...")
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_df['class'].values

    # Plot class distribution
    plot_class_distribution(labels, unique_labels)

    # Ensure the labels parameter is passed to confusion_matrix and classification_report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=unique_labels, labels=range(len(unique_labels)), zero_division=0))

    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(unique_labels)))
    plot_confusion_matrix(cm, unique_labels)

    # Plot ROC Curves
    plot_roc_curves(true_labels, predictions, len(unique_labels), unique_labels)

if __name__ == "__main__":
    main()
