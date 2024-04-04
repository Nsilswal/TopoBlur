import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold

import tensorflow as tf
import numpy as np

data_dir = 'MNIST_modified_no_padding'
img_height = 28
img_width = 28
batch_size = 32  # You can adjust this depending on your memory capacity

all_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Normalize pixel values
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
all_ds = all_ds.map(lambda x, y: (normalization_layer(x), y))

# Initialize lists to hold the images and labels
images = []
labels = []

# Iterate over the dataset and collect all data into lists
for imgs, lbls in all_ds:
    images.append(imgs.numpy())
    labels.append(lbls.numpy())

# Concatenate lists to form arrays
X = np.concatenate(images, axis=0)
y = np.concatenate(labels, axis=0)

def create_model(input_shape=(28, 28, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Assume X and y are preloaded datasets
# You will need to load and preprocess your dataset here

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train_index, val_index in kf.split(X):
    X_train_fold, X_val_fold = X[train_index], X[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    # Create a new model for this fold
    model = create_model(input_shape=X_train_fold.shape[1:], num_classes=10)
    
    # Fit the model
    history = model.fit(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold))
    
    # Evaluate the model
    scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    fold_no += 1

# Print the average scores
print('----------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('----------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('----------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('----------------------------------------------------------------')
