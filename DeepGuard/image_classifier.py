import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ImageClassifier:
    def __init__(self, data_path="Data", img_size=(128, 128)):
        """
        Initialize the image classifier
        
        Args:
            data_path (str): Path to the data directory
            img_size (tuple): Target image size (width, height)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.model = None
        self.history = None
        self.class_names = ['Fake', 'Real']
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess images from the data directory
        
        Returns:
            tuple: (X, y) where X is the image data and y is the labels
        """
        print("Loading and preprocessing images...")
        
        images = []
        labels = []
        
        # Load fake images
        fake_path = os.path.join(self.data_path, "Fake")
        for img_name in os.listdir(fake_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(fake_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0  # Normalize to [0, 1]
                    images.append(img)
                    labels.append(0)  # 0 for fake
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Load real images
        real_path = os.path.join(self.data_path, "Real")
        for img_name in os.listdir(real_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(real_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0  # Normalize to [0, 1]
                    images.append(img)
                    labels.append(1)  # 1 for real
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images")
        print(f"Fake images: {np.sum(y == 0)}")
        print(f"Real images: {np.sum(y == 1)}")
        
        return X, y
    
    def create_model(self):
        """
        Create a CNN model for image classification
        """
        print("Creating model...")
        
        # Use transfer learning with MobileNetV2 as base
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[1], self.img_size[0], 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model created successfully!")
        self.model.summary()
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """
        Create data generators for training and validation
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=8,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=8,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=30):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
        """
        print("Training model...")
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators(
            X_train, y_train, X_val, y_val
        )
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduce learning rate when plateau is reached
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=max(1, len(X_train) // 8),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=max(1, len(X_val) // 8),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test, y_test: Test data
        """
        print("Evaluating model...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        return accuracy, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_image(self, image_path):
        """
        Predict class for a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (prediction, confidence, class_name)
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)[0][0]
        class_name = self.class_names[int(prediction > 0.5)]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return prediction, confidence, class_name
    
    def save_model(self, model_path="deepguard_model.h5"):
        """
        Save the trained model
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save!")

def main():
    """
    Main function to run the complete pipeline
    """
    print("=== DeepGuard Image Classification Model ===")
    print("Classifying between Real and Fake Images\n")
    
    # Initialize classifier
    classifier = ImageClassifier()
    
    # Load and preprocess data
    X, y = classifier.load_and_preprocess_data()
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create and train model
    classifier.create_model()
    classifier.train_model(X_train, y_train, X_val, y_val, epochs=30)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = classifier.evaluate_model(X_test, y_test)
    
    # Save model
    classifier.save_model()
    
    # Test on a few sample images
    print("\n=== Sample Predictions ===")
    
    # Test on specific fake image
    sample_fake = r"F:\DeepGuard\Data\Fake\fake_10.jpg"
    if os.path.exists(sample_fake):
        pred, conf, class_name = classifier.predict_single_image(sample_fake)
        print(f"Sample Fake Image (fake_10.jpg):")
        print(f"  Prediction: {class_name}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Raw Score: {pred:.4f}")
    else:
        print(f"Fake image not found at: {sample_fake}")
    
    # Test on specific real image
    sample_real = r"F:\DeepGuard\Data\Real\real_1006.jpg"
    if os.path.exists(sample_real):
        pred, conf, class_name = classifier.predict_single_image(sample_real)
        print(f"\nSample Real Image (real_1006.jpg):")
        print(f"  Prediction: {class_name}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Raw Score: {pred:.4f}")
    else:
        print(f"Real image not found at: {sample_real}")
    
    print(f"\n=== Training Complete ===")
    print(f"Final Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 