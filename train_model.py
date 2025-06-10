import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load and preprocess MNIST dataset
def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Reshape images to include channel dimension
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    
    return (train_images, train_labels), (test_images, test_labels)

# Create and compile the model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def main():
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    # Create and train model
    model = create_model()
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    
    # Save the model
    model.save('mnist_model.h5')
    print("Model saved as 'mnist_model.h5'")

if __name__ == '__main__':
    main() 