import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



def create_spectrogram(audio_path, output_path, sr=22050, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_and_save_spectrograms(dataset_path, spectrogram_path):
    labels = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".mp3"):
                label = os.path.basename(root)
                labels.append(label)
                
                audio_path = os.path.join(root, file)
                spectrogram_file_path = os.path.join(spectrogram_path, label + "_" + file.replace(".mp3", ".png"))
                
                try:
                    create_spectrogram(audio_path, spectrogram_file_path)
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
    
    return labels

class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, img_size=(224, 224)):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_file_paths = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        for file_path in batch_file_paths:
            img = tf.keras.preprocessing.image.load_img(file_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
        
        return np.array(images), np.array(batch_labels)

    def on_epoch_end(self):
        self.file_paths, self.labels = shuffle(self.file_paths, self.labels)

def main():
    dataset_path = "C:/Users/tring/Desktop/archive/Voice of Birds"
    spectrogram_path = "C:/Users/tring/Desktop/archive/spectrograms"
    
    os.makedirs(spectrogram_path, exist_ok=True)

    labels = generate_and_save_spectrograms(dataset_path, spectrogram_path)

    spectrogram_files = [os.path.join(spectrogram_path, f) for f in os.listdir(spectrogram_path) if f.endswith(".png")]
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indexed_labels = [label_to_index[label] for label in labels]


if __name__ == "__main__":
    main()
