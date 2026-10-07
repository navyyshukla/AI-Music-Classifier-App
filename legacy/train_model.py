import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress TensorFlow and other warnings for a cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CNNModelTrainer:
    """
    An advanced pipeline to train and evaluate multiple CNN architectures
    for music genre classification from spectrogram images.
    """
    def __init__(self, data_dir, image_size=(128, 431), batch_size=32, epochs=50):
        self.data_dir = data_dir
        self.img_height, self.img_width = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        # This automatically detects genre names from the folder structure, making it scalable
        self.class_names = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
        self.num_classes = len(self.class_names)
        self.history = {}
        self.results = {}

    def load_and_prepare_datasets(self):
        """Loads image data and splits it into training, validation, and test sets."""
        print("\n--- Loading and Preparing Image Datasets ---")
        
        # Keras utility to automatically load images and infer labels from folder names.
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            labels='inferred',
            label_mode='categorical', # Use 'categorical' for multi-class classification
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            validation_split=0.2, # Reserve 20% of data for validation and testing
            subset='both'
        )
        
        self.train_ds, self.val_ds = dataset
        
        # Further split the validation set into a dedicated test set
        val_batches = tf.data.experimental.cardinality(self.val_ds)
        self.test_ds = self.val_ds.take(val_batches // 2)
        self.val_ds = self.val_ds.skip(val_batches // 2)
        
        print(f"Found {self.num_classes} genres: {self.class_names}")
        print(f"Training batches: {tf.data.experimental.cardinality(self.train_ds)}")
        print(f"Validation batches: {tf.data.experimental.cardinality(self.val_ds)}")
        print(f"Test batches: {tf.data.experimental.cardinality(self.test_ds)}")

        # Optimize dataset performance by caching and prefetching
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    def define_models(self):
        """
        Defines a dictionary of different CNN architectures to be trained and compared.
        This fulfills your requirement of using multiple algorithms.
        """
        print("\n--- Defining Model Architectures ---")
        
        models_dict = {}

        # Model 1: A simple but effective baseline CNN
        models_dict['Baseline_CNN'] = models.Sequential([
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Model 2: A deeper CNN with Dropout for regularization (to prevent overfitting)
        models_dict['Deeper_CNN_with_Dropout'] = models.Sequential([
            layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Model 3: Transfer Learning with a pre-trained model (MobileNetV2)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False, # We will add our own classification head
            weights='imagenet' # Use weights pre-trained on the ImageNet dataset
        )
        base_model.trainable = False # Freeze the pre-trained layers
        
        models_dict['Transfer_Learning_MobileNetV2'] = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return models_dict

    def train_and_evaluate(self):
        """Trains each defined model and stores its performance."""
        all_models = self.define_models()
        
        for model_name, model in all_models.items():
            print(f"\n{'='*60}\nTraining Model: {model_name}\n{'='*60}")
            
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks for more efficient training
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

            history = model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=self.epochs,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            self.history[model_name] = history
            
            print(f"\nEvaluating {model_name} on the unseen test set...")
            loss, accuracy = model.evaluate(self.test_ds)
            self.results[model_name] = {'model': model, 'accuracy': accuracy, 'loss': loss}
            print(f"Test Accuracy for {model_name}: {accuracy * 100:.2f}%")

    def find_and_save_best_model(self):
        """Compares results, identifies the best model, and saves it."""
        if not self.results:
            print("No models were trained. Exiting.")
            return

        best_model_name = max(self.results, key=lambda name: self.results[name]['accuracy'])
        best_model = self.results[best_model_name]['model']
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print(f"\n--- Model Comparison Summary ---")
        for name, result in self.results.items():
            print(f"  - {name}: {result['accuracy']*100:.2f}% Test Accuracy")
        
        print(f"\nBest model is '{best_model_name}' with {best_accuracy*100:.2f}% accuracy.")
        
        # Save the best model in the recommended .keras format for TensorFlow
        model_save_path = 'best_cnn_model.keras'
        best_model.save(model_save_path)
        print(f"Best model saved to: {model_save_path}")
        
        # Also save the class names (genre list) for the app to use
        np.save('class_names.npy', self.class_names)
        print("Class names saved to: class_names.npy")

        self.plot_training_history(best_model_name)
    
    def plot_training_history(self, model_name):
        """Plots the accuracy and loss curves for the best model."""
        history = self.history[model_name]
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title(f'Accuracy for {model_name}')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title(f'Loss for {model_name}')
        
        plt.savefig('training_history.png')
        print("Training history plot saved to: training_history.png")
        plt.show()

if __name__ == "__main__":
    # The script will look for your augmented image data in this folder
    DATA_DIR = 'spectrograms_augmented'
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        print("Please run the 'feature_extractor.py' script first to generate the spectrogram images.")
    else:
        trainer = CNNModelTrainer(data_dir=DATA_DIR)
        trainer.load_and_prepare_datasets()
        trainer.train_and_evaluate()
        trainer.find_and_save_best_model()