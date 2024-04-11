import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
data_dir = 'MNIST_blur_beta2'  # Adjusted to your new dataset path
img_height, img_width = 28, 28
batch_size = 32
num_classes = 10
validation_split = 0.2  # Reserve 20% of the data for validation

# Create ImageDataGenerators for training and validation, with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split  # Specify the validation split
)

# Generate batches of augmented data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    subset='training'  # Specify this is the training subset
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    subset='validation'  # Specify this is the validation subset
)

# Define the MLP model (same as before)
model = models.Sequential([
    layers.Flatten(input_shape=(img_height, img_width, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)
