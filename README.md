AuraMood: Audio Emotion Classification from Multiple Datasets
AuraMood is a deep learning-based emotion classification system that predicts the emotional state of a speaker from audio files. By leveraging two different emotion-labeled datasets, RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) and TESS (Toronto Emotional Speech Set), this project aims to identify emotions in spoken language with an overall F1 score of 80% across 8 distinct emotional classes: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

Dataset Information
This system is built using two key datasets:

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Includes 1440 speech and 1012 song audio files with emotional expressions from 24 actors (12 male, 12 female).
Emotions covered: Calm, Happy, Sad, Angry, Fearful, Surprise, Disgust (for speech), and Calm, Happy, Sad, Angry, Fearful (for songs).
Available at: RAVDESS Dataset.
TESS (Toronto Emotional Speech Set)

Contains 2800 speech files across seven emotions: Anger, Disgust, Fear, Happiness, Pleasant Surprise, Sadness, and Neutral.
Available at: TESS Dataset.
Emotions to Classify
0 = Neutral
1 = Calm
2 = Happy
3 = Sad
4 = Angry
5 = Fearful
6 = Disgust
7 = Surprised
Due to the skewed nature of the TESS dataset (absence of a calm class), this model may perform with slightly less accuracy in predicting the calm emotion.

Model Performance
The model is a deep learning classifier designed to predict the emotional state of a speaker. It achieves an F1 score of 80% on the 8-class emotion classification task.

Model Summary

Loss and Accuracy Plots

Classification Report

Confusion Matrix

Setup & Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/AuraMood.git
cd AuraMood
Optional: Download Datasets:

RAVDESS: Download the audio files and unzip them into the features folder.
TESS: Download and unzip the dataset into the TESS_Toronto_emotional_speech_set_data folder.
Prepare Data (Optional):

Run tess_pipeline.py to organize the TESS dataset into the correct folder structure.
If you want to create new features, run create_features.py. However, the pre-generated features are available in the features folder.
Train the Model (Optional):

If you want to train a new model, run:
bash
Copy code
python neural_network.py
Pre-trained models are available in the model folder, so training may not be necessary.
How to Test the Model
To test the model with new audio files:

Place your test audio file in the examples folder.
Run the prediction script to classify the emotion:
bash
Copy code
python predict.py --audio_file path/to/audio_file.wav
The model will predict one of the following emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, or Surprised.
Note: The classes in the code are encoded from 0 to 7, corresponding to the emotion labels in the dataset (01 to 08 in the original dataset). So, if the model predicts 0, it corresponds to the Neutral class.

Example Predictions
03-01-01-01-01-02-05.wav: This is a neutral audio file, but the model predicts calm. Listen to the audio and verify if the emotion prediction seems reasonable.
10-16-07-29-82-30-63.wav: A disgust file, where the model performs as expected.
Contributing
Feel free to contribute to AuraMood by submitting pull requests or reporting issues. Any improvements or additions are welcome, especially related to handling more diverse emotion categories or adding additional datasets.
