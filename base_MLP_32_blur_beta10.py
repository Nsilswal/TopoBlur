import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Model configuration
input_shape = (28, 28)  # MNIST images size
num_classes = 10        # Digits 0-9

# Model definition
model = Sequential([
    Flatten(input_shape=input_shape),  # Flattens the input
    Lambda(lambda x: x / 255.0),  # Normalize inputs
    Dense(32, activation='relu'),     # First hidden layer
    Dense(32, activation='relu'),     # Second hidden layer
    Dense(num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=SGD(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Define the path to the dataset directory
dataset_dir = 'MNIST_blur_beta10'

# Load the data
full_dataset = image_dataset_from_directory(
    dataset_dir,
    label_mode='int',
    image_size=(28, 28),
    batch_size=32,
    color_mode='grayscale',
    shuffle=True,
    seed=123
)

# Determine the total number of batches in the dataset
total_batches = tf.data.experimental.cardinality(full_dataset).numpy()

# Calculate split sizes for 80-20 train-test split
train_size = int(0.8 * total_batches)
test_size = total_batches - train_size

# Split the dataset into training and testing
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy*100:.2f}%")
