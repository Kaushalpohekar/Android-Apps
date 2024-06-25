import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
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
    'images/train',
    target_size=(64, 64),
    batch_size=12,
    class_mode='categorical'
)

# Load test set
test_set = test_datagen.flow_from_directory(
    'images/test',
    target_size=(64, 64),
    batch_size=12,
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
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=20,
    validation_data=test_set,
    validation_steps=len(test_set)
)
