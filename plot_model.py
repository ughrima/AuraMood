import keras
from keras.utils import plot_model
import os  # Make sure to import os for path handling

from config import MODEL_DIR_PATH

# Correct the path using os.path.join
model_path = os.path.join(MODEL_DIR_PATH, 'Emotion_Voice_Detection_Model.h5')

# Load the model
restored_keras_model = keras.models.load_model(model_path)

# Generate and save the model plot
plot_model(restored_keras_model, to_file='media/model.png')
