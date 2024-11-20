import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv1D, Flatten, Dropout, Activation
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from config import SAVE_DIR_PATH, MODEL_DIR_PATH


class TrainModel:

    @staticmethod
    def train_neural_network(X, y) -> None:
        """
        This function trains the neural network.
        """

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)

        # Reshape the data for Conv1D
        x_traincnn = np.expand_dims(X_train, axis=2)
        x_testcnn = np.expand_dims(X_test, axis=2)

        print(
            f"Train shape: {x_traincnn.shape}, Test shape: {x_testcnn.shape}")

        # Build the model
        model = Sequential()
        model.add(Conv1D(64, 5, padding='same', input_shape=(40, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(8))
        model.add(Activation('softmax'))

        print(model.summary())  # Corrected to call model.summary()

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])

        # Train the model
        cnn_history = model.fit(
            x_traincnn, y_train, batch_size=16, epochs=50, validation_data=(x_testcnn, y_test))

        # Loss plotting
        plt.plot(cnn_history.history['loss'])
        plt.plot(cnn_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('loss.png')
        plt.close()

        # Accuracy plotting
        plt.plot(cnn_history.history['accuracy'])
        plt.plot(cnn_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('accuracy.png')

        # Get predictions
        predictions = model.predict(x_testcnn)

        # Convert the predicted probabilities to class labels
        predicted_classes = np.argmax(predictions, axis=1)

        # Ensure that y_test is of integer type
        new_y_test = y_test.astype(int)

        # Confusion Matrix
        matrix = confusion_matrix(new_y_test, predicted_classes)
        print("Confusion Matrix:")
        print(matrix)

        # Classification Report
        print("Classification Report:")
        print(classification_report(new_y_test, predicted_classes))

        # Save the model
        model_name = 'Emotion_Voice_Detection_Model.h5'

        if not os.path.isdir(MODEL_DIR_PATH):
            os.makedirs(MODEL_DIR_PATH)
        model_path = os.path.join(MODEL_DIR_PATH, model_name)
        model.save(model_path.replace('.h5', '.keras'))
        print(f'Saved trained model at {model_path}')


if __name__ == '__main__':
    print('Training started')

    # Load data
    X = joblib.load(SAVE_DIR_PATH + '/X.joblib')
    y = joblib.load(SAVE_DIR_PATH + '/y.joblib')

    # Train the neural network
    TrainModel.train_neural_network(X=X, y=y)
