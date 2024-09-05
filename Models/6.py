import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, TimeDistributed, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
frame_size = (128, 128)
batch_size = 32
sequence_length = 10

# Directory paths
train_dir = 'idd_dataset_extracted/train'
val_dir = 'idd_dataset_extracted/val'
test_dir = 'idd_dataset_extracted/test'

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=frame_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=frame_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=frame_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Image classification model
image_model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(frame_size[0], frame_size[1], 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

image_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('image_classification_model.keras', save_best_only=True)

# Train image model
image_history = image_model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate image model
test_loss, test_acc = image_model.evaluate(test_generator)
print(f"Test accuracy for image model: {test_acc:.2f}")

# Save the image model
image_model.save('image_classification_model.keras')

# Video classification model
video_model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(sequence_length, frame_size[0], frame_size[1], 3)),
    TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(128, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    ConvLSTM2D(128, (3, 3), activation='relu', return_sequences=False),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

video_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
video_early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
video_model_checkpoint = ModelCheckpoint('video_classification_model.keras', save_best_only=True)

# Function to generate sequences from image generators
def generate_sequences(generator, sequence_length):
    while True:
        X_batch, y_batch = next(generator)
        X_seq_batch = []
        y_seq_batch = []
        for i in range(0, len(X_batch), sequence_length):
            if i + sequence_length <= len(X_batch):
                X_seq_batch.append(X_batch[i:i+sequence_length])
                y_seq_batch.append(y_batch[i])
        yield np.array(X_seq_batch), np.array(y_seq_batch)

train_sequence_generator = generate_sequences(train_generator, sequence_length)
val_sequence_generator = generate_sequences(val_generator, sequence_length)

# Calculate steps per epoch
train_steps_per_epoch = train_generator.samples // (batch_size * sequence_length)
val_steps_per_epoch = val_generator.samples // (batch_size * sequence_length)

# Train video model
video_history = video_model.fit(
    train_sequence_generator,
    steps_per_epoch=train_steps_per_epoch,
    epochs=50,
    validation_data=val_sequence_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[video_early_stopping, video_model_checkpoint]
)

# Save the video model
video_model.save('video_classification_model.keras')

# Evaluate video model
test_sequence_generator = generate_sequences(test_generator, sequence_length)
test_steps = test_generator.samples // (batch_size * sequence_length)
test_loss, test_acc = video_model.evaluate(test_sequence_generator, steps=test_steps)
print(f"Test accuracy for video model: {test_acc:.2f}")
