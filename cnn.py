import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ensure TensorFlow is using GPU
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Data augmentation and loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training set
training_set = train_datagen.flow_from_directory(
    './train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Load test set
test_set = test_datagen.flow_from_directory(
    './test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Automatically determine the number of classes
num_classes = training_set.num_classes
print("Number of classes:", num_classes)

# Define the model
classifier = Sequential()

# Add layers
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=50,  # Increase epochs
    validation_data=test_set,
    validation_steps=len(test_set)
)

# Save the model
model_path = 'model.h5'
classifier.save(model_path)
print(f"Model saved to {model_path}")

# Evaluate the model on the test set
test_loss, test_acc = classifier.evaluate(test_set, steps=len(test_set))
print(f'Test accuracy: {test_acc * 100:.2f}%')
