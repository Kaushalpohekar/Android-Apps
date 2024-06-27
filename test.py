import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Is GPU available?", "Yes" if tf.config.list_physical_devices('GPU') else "No")
