# AuraMood: Audio Emotion Classification from Multiple Datasets

AuraMood is a deep learning-based emotion classification system designed to predict the emotional state of a speaker from audio files. By leveraging two distinct emotion-labeled datasets—RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) and TESS (Toronto Emotional Speech Set)—AuraMood achieves an **F1 score of 80%** across 8 distinct emotional classes: **Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised**.

---

## **Features**
- **Emotion Classification**: Detects emotions in speech and song audio files.
- **Multi-Dataset Integration**: Combines RAVDESS and TESS datasets for robust training.
- **8-Class Classification**: Covers a wide range of emotions.
- **Pre-trained Models**: Ready-to-use models for prediction.
- **Custom Predictions**: Test with your own audio files.

---

## **Datasets**

### **1. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
- **Overview**:
  - 1440 speech and 1012 song audio files.
  - Performed by 24 actors (12 male, 12 female).
- **Emotions Covered**:
  - Speech: Calm, Happy, Sad, Angry, Fearful, Surprise, Disgust.
  - Songs: Calm, Happy, Sad, Angry, Fearful.
- **Download**: [RAVDESS Dataset](https://zenodo.org/record/1188976)

### **2. TESS (Toronto Emotional Speech Set)**
- **Overview**:
  - 2800 speech files.
  - Spoken by 2 female actors.
- **Emotions Covered**:
  - Neutral, Anger, Disgust, Fear, Happiness, Pleasant Surprise, Sadness.
- **Download**: [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487)

**Note**: The TESS dataset lacks a Calm emotion class, which may affect accuracy for this category.

---

## **Emotions to Classify**
| **Label** | **Emotion**      |
|-----------|------------------|
| 0         | Neutral          |
| 1         | Calm             |
| 2         | Happy            |
| 3         | Sad              |
| 4         | Angry            |
| 5         | Fearful          |
| 6         | Disgust          |
| 7         | Surprised        |

---

## **Model Performance**
- **F1 Score**: **80%** for the 8-class emotion classification task.
- Includes **Loss and Accuracy Plots**, **Classification Report**, and **Confusion Matrix** for detailed performance metrics.

---

## **Setup and Installation**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/AuraMood.git
cd AuraMood
```

### **Step 2: Download Datasets (Optional)**
- **RAVDESS**: Download and unzip audio files into the `features` folder.
- **TESS**: Download and unzip into the `TESS_Toronto_emotional_speech_set_data` folder.

### **Step 3: Prepare Data (Optional)**
- To organize the TESS dataset:
  ```bash
  python tess_pipeline.py
  ```
- To create new features (optional, pre-generated features are included):
  ```bash
  python create_features.py
  ```

### **Step 4: Train the Model (Optional)**
- To train a new model:
  ```bash
  python neural_network.py
  ```
- Pre-trained models are available in the `model` folder, so training is optional.

---

## **Testing the Model**

### **Step 1: Test with Custom Audio Files**
1. Place your test audio file in the `examples` folder.
2. Run the prediction script:
   ```bash
   python predict.py --audio_file path/to/audio_file.wav
   ```

### **Output**
The model predicts one of the following emotions:
- **Neutral**
- **Calm**
- **Happy**
- **Sad**
- **Angry**
- **Fearful**
- **Disgust**
- **Surprised**

The prediction classes correspond to the encoded labels (0-7). For example:
- **0 = Neutral**
- **1 = Calm**

### **Example Predictions**
- **File**: `03-01-01-01-01-02-05.wav`
  - Expected Emotion: Neutral
  - Model Prediction: Calm (check audio to verify reasonableness).
- **File**: `10-16-07-29-82-30-63.wav`
  - Expected Emotion: Disgust
  - Model Prediction: Disgust (accurate).

---

## **Contributing**
We welcome contributions to improve AuraMood! Suggestions for:
- Enhancing model performance.
- Adding new emotion categories.
- Integrating additional datasets.

To contribute:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request with a clear description of changes.

---

## **Future Enhancements**
- Expanding to multilingual datasets.
- Supporting real-time emotion detection.
- Addressing dataset imbalance for Calm and other under-represented emotions.

