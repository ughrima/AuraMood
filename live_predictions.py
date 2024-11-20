import keras
import librosa
import numpy as np
import os

# Ensure these paths are correct in config.py
from config import EXAMPLES_PATH, MODEL_DIR_PATH


class LivePredictions:
    """
    Main class for emotion prediction from audio files.
    """

    def __init__(self, file):
        """
        Initializes the model and loads the specified audio file.
        """
        self.file = file
        self.path = os.path.join(
            MODEL_DIR_PATH, 'Emotion_Voice_Detection_Model.h5')

        # Check if the model file exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model file not found: {self.path}")

        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Processes the audio file and makes predictions using the loaded model.
        """
        # Load the audio file
        data, sampling_rate = librosa.load(self.file)

        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(
            y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)

        # Reshape the MFCCs to match the input shape expected by the model
        x = np.expand_dims(mfccs, axis=-1)  # Shape becomes (40, 1)
        x = np.expand_dims(x, axis=0)       # Shape becomes (1, 40, 1)

        # Make a prediction
        prediction = self.loaded_model.predict(x)

        # Convert prediction to emotion label
        emotion = self.convert_class_to_emotion(prediction)

        print(f"Predicted Emotion: {emotion}")

    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Converts model output to human-readable emotion label.
        """
        label_conversion = {
            '0': 'neutral', '1': 'calm', '2': 'happy', '3': 'sad',
            '4': 'angry', '5': 'fearful', '6': 'disgust', '7': 'surprised'
        }
        # Convert predicted class to human-readable emotion
        return label_conversion.get(str(np.argmax(pred)), 'unknown')


if __name__ == '__main__':
    # Define the example file paths
    example_files = [
        os.path.join(
            EXAMPLES_PATH, '/Users/bajajnidhi/Desktop/predictive prject/AuraMood/examples/03-01-01-01-01-02-05.wav'),
        os.path.join(
            EXAMPLES_PATH, '/Users/bajajnidhi/Desktop/predictive prject/AuraMood/examples/10-16-07-29-82-30-63.wav')
    ]

    # Iterate over example files and make predictions
    for file in example_files:
        if not os.path.exists(file):
            print(f"Example file not found: {file}")
        else:
            print(f"Processing file: {file}")
            live_prediction = LivePredictions(file=file)
            live_prediction.loaded_model.summary()  # Print model summary for verification
            live_prediction.make_predictions()
