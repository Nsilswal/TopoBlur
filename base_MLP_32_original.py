import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Lambda

# Model configuration
input_shape = (28, 28)  # MNIST images size
num_classes = 10        # Digits 0-9

# Model definition
model = Sequential([
    Flatten(input_shape=input_shape),  # Flattens the input
    Lambda(lambda x: x / 255.0),  # Add this line to manually scale inputs
    Dense(32, activation='relu'),     # First hidden layer with 32 neurons and ReLU activation
    Dense(32, activation='relu'),     # Second hidden layer with 32 neurons and ReLU activation
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer=SGD(),  # Using Stochastic Gradient Descent as the optimizer
              loss='sparse_categorical_crossentropy',  # Loss function for integers target
              metrics=['accuracy'])  # Metric to monitor for training

# Model summary
model.summary()


# Define the path to the training and testing directories
train_dir = 'DatasetOrg/MNIST/mnist_png/train'
test_dir = 'DatasetOrg/MNIST/mnist_png/test'

# Load the data
train_dataset = image_dataset_from_directory(
    train_dir,
    label_mode='int',  # Labels are returned as integers
    image_size=(28, 28),  # Resize images to 28x28
    batch_size=32,  # Number of samples per batch
    color_mode='grayscale',  # MNIST is typically gray scale so ensure images are loaded as such
    seed=123       # Seed for reproducibility
)

test_dataset = image_dataset_from_directory(
    test_dir,
    label_mode='int',
    image_size=(28, 28),
    batch_size=32,
    color_mode='grayscale',
    shuffle=False  # No need to shuffle the test data
)

history = model.fit(
    train_dataset,
    epochs=10,  # Number of iterations over the entire dataset
    validation_data=test_dataset  # Optionally add validation data
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
