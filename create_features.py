import os
import time
import joblib
import librosa
import numpy as np

# Update your config paths properly
from config import SAVE_DIR_PATH
from config import TRAINING_FILES_PATH

class CreateFeatures:

    @staticmethod
    def features_creator(path, save_dir) -> str:
        """
        Extracts MFCCs from the audio files located in `path` and saves
        the features and labels as joblib files in `save_dir`.
        """
        lst = []
        start_time = time.time()

        for subdir, dirs, files in os.walk(path):
            for file in files:
                try:
                    # Load audio file using librosa
                    X, sample_rate = librosa.load(os.path.join(subdir, file),
                                                  res_type='kaiser_fast')
                    # Extract MFCC features
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                    
                    # Convert labels from 1-8 to 0-7 for compatibility with the model
                    file_label = int(file[7:8]) - 1
                    lst.append((mfccs, file_label))

                except ValueError as err:
                    print(f"Error processing {file}: {err}")
                    continue

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        # Unzipping list to features (X) and labels (y)
        X, y = zip(*lst)

        # Convert to numpy arrays
        X, y = np.asarray(X), np.asarray(y)

        # Print the shapes of the arrays for verification
        print(f"Feature shape: {X.shape}, Label shape: {y.shape}")

        # Save features and labels as joblib files
        joblib.dump(X, os.path.join(save_dir, 'X.joblib'))
        joblib.dump(y, os.path.join(save_dir, 'y.joblib'))

        return "Feature extraction completed."


if __name__ == '__main__':
    print('Feature extraction routine started')
    result = CreateFeatures.features_creator(path=TRAINING_FILES_PATH, save_dir=SAVE_DIR_PATH)
    print(result)
