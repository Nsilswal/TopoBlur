import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load the entire dataset
data_dir = 'MNIST_modified_no_padding'
batch_size = 32
img_height = 28
img_width = 28

all_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Normalize pixel values
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

all_ds = all_ds.map(lambda x, y: (normalization_layer(x), y))

# Split the dataset into train, validation, and test
def split_dataset(dataset, train_split=0.7, test_split=0.15, shuffle=True, shuffle_size=10000):
    if shuffle:
        # Shuffle the dataset
        dataset = dataset.shuffle(shuffle_size, seed=12)
    
    dataset_size = dataset.cardinality().numpy()
    train_size = int(train_split * dataset_size)
    test_size = int(test_split * dataset_size)
    
    train_ds = dataset.take(train_size)
    test_ds = dataset.skip(train_size).take(test_size)
    val_ds = dataset.skip(train_size + test_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = split_dataset(all_ds)

# Configure the dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standard LeNet5 Model
def build_lenet5_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential()
    
    # First convolutional layer with 32 filters and a kernel size of 5x5
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(img_height, img_width, 3)))
    # Pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional layer with 64 filters and a kernel size of 5x5
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # Pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening the 3D output to 1D
    model.add(layers.Flatten())
    
    # Fully connected layer
    model.add(layers.Dense(120, activation='relu'))
    
    # Another fully connected layer
    model.add(layers.Dense(84, activation='relu'))
    
    # Output layer with softmax activation
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

model = build_lenet5_model()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

# Evaluate the model
model.evaluate(test_ds)
