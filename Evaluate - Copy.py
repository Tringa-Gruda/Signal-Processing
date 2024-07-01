import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, img_size=(224, 224), **kwargs):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        super().__init__(**kwargs)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_file_paths = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        for file_path in batch_file_paths:
            try:
                img = tf.keras.preprocessing.image.load_img(file_path, target_size=self.img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
        
        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        self.file_paths, self.labels = shuffle(self.file_paths, self.labels)

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
                print(f"Found file: {spectrogram_file_path} with label: {label}")  # Debugging statement
    
    return file_paths, labels

def main():
    spectrogram_path = "C:/Users/tring/Desktop/archive/spectrograms"
    model_save_path = "C:/Users/tring/Desktop/archive/best_model.keras"
    
    spectrogram_files, labels = load_spectrogram_files(spectrogram_path)

    if not spectrogram_files:
        print("No spectrogram files found.")
        return

    print(f"Labels: {labels}")  # Debugging statement to print labels

    unique_labels = sorted(list(set(labels)))
    print(f"Unique Labels: {unique_labels}")  # Debugging statement to print unique labels
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indexed_labels = [label_to_index[label] for label in labels]

    if len(unique_labels) <= 1:
        print("Error: The number of unique labels must be greater than 1 for multi-class classification.")
        return

    train_files, test_files, train_labels, test_labels = train_test_split(spectrogram_files, indexed_labels, test_size=0.2, random_state=42)

    batch_size = 32

    train_gen = AudioDataGenerator(train_files, train_labels, batch_size=batch_size)
    test_gen = AudioDataGenerator(test_files, test_labels, batch_size=batch_size)

    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Starting model training...")
    model.fit(train_gen, validation_data=test_gen, epochs=5)  # Increased epochs for better training
    print("Model training finished.")

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    # Make predictions on the test set
    print("Making predictions on the test set...")
    predictions = model.predict(test_gen)
    predicted_labels = np.argmax(predictions, axis=1)

    # Generate classification report and confusion matrix
    true_labels = np.concatenate([y for _, y in test_gen], axis=0)
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=[unique_labels[i] for i in sorted(label_to_index.values())], zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

if __name__ == "__main__":
    main()
