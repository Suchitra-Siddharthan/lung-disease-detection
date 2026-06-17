# Lung Disease Detection System

A machine learning-based web application for classifying chest X-ray images into three categories: **COVID**, **Normal**, and **Pneumonia**. The system uses handcrafted feature extraction techniques (GLCM, LBP, and HOG) combined with classical machine learning models including Support Vector Machine (SVM), Random Forest (RF), and K-Nearest Neighbors (KNN).

The application provides an interactive React frontend, a Flask backend, confidence score visualization, feature contribution analysis, and rule-based diagnostic explanations.

---

## Features

* Chest X-ray image upload and analysis
* Classification into:

  * COVID
  * Normal
  * Pneumonia
* Multiple ML models:

  * Support Vector Machine (SVM)
  * Random Forest (RF)
  * K-Nearest Neighbors (KNN)
* Soft-voting ensemble prediction
* Confidence score visualization
* Feature contribution analysis
* Rule-based prediction explanation
* REST API-based architecture
* Interactive React + TypeScript frontend

---

## Tech Stack

### Frontend

* ReactJS
* TypeScript
* Vite
* Recharts

### Backend

* Flask
* Flask-CORS

### Machine Learning & Image Processing

* Scikit-learn
* OpenCV
* NumPy
* Scikit-image
* Joblib

---

## Dataset

The project uses chest X-ray images belonging to three categories:

* COVID
* Normal
* Pneumonia

The images are organized into class-specific directories and processed into numerical feature vectors before model training.

---

## Project Workflow

```text
Chest X-ray Image
        │
        ▼
Image Preprocessing
        │
        ▼
Feature Extraction
(GLCM + LBP + HOG)
        │
        ▼
Feature Scaling
(StandardScaler)
        │
        ▼
Model Prediction
(SVM / RF / KNN)
        │
        ▼
Soft Voting Ensemble
        │
        ▼
Prediction + Confidence Scores
        │
        ▼
Feature Analysis + Explanation
```

---

## Data Preprocessing

Each image undergoes the following preprocessing pipeline:

### 1. Image Resizing

All images are resized to:

```text
224 × 224 pixels
```

This ensures a consistent input size for feature extraction.

### 2. Grayscale Conversion

Images are converted to grayscale because chest X-rays primarily contain diagnostic information in intensity patterns rather than color information.

### 3. Normalization

Pixel values are normalized to:

```text
0 – 1
```

This improves numerical stability during feature extraction and model training.

---

## Feature Extraction

Instead of using deep learning-based feature learning, the project uses handcrafted image descriptors.

### GLCM (Gray Level Co-occurrence Matrix)

GLCM captures texture characteristics of lung tissue.

Extracted properties:

* Contrast
* Dissimilarity
* Homogeneity
* Energy
* Correlation
* ASM (Angular Second Moment)

For each property:

* Mean
* Standard Deviation

are computed and added to the feature vector.

Purpose:

* Detect texture abnormalities
* Capture intensity variations in lung regions

---

### LBP (Local Binary Pattern)

Configuration:

```text
Radius = 2
P = 16
Method = Uniform
```

LBP captures local texture patterns and micro-level image variations.

Purpose:

* Detect fine-grained texture changes
* Capture local abnormalities in lung tissue

---

### HOG (Histogram of Oriented Gradients)

Configuration:

```text
Orientations = 9
Pixels Per Cell = 8 × 8
Cells Per Block = 2 × 2
```

Purpose:

* Capture edge information
* Capture structural patterns
* Detect shape variations and lung boundary distortions

---

## Feature Vector Construction

The final feature vector is created by concatenating:

```text
GLCM Features + LBP Features + HOG Features
```

This combined representation is used for training and inference.

---

## Feature Scaling

A StandardScaler is applied before model training.

Purpose:

* Standardize feature distributions
* Prevent large-valued features from dominating
* Improve SVM and KNN performance

The trained scaler is saved and reused during inference to ensure consistent preprocessing.

---

## Dataset Splitting

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

Parameters:

```python
test_size = 0.2
random_state = 42
stratify = labels
```

Stratification ensures balanced class distribution across training and testing sets.

---

## Machine Learning Models

### Support Vector Machine (SVM)

Configuration:

```python
kernel='rbf'
probability=True
class_weight='balanced'
```

Purpose:

* Handles high-dimensional feature spaces effectively
* Captures nonlinear class boundaries
* Generates probability scores for confidence visualization

---

### Random Forest

Configuration:

```python
n_estimators=100
```

Purpose:

* Ensemble tree-based classifier
* Reduces overfitting
* Provides stable predictions

---

### K-Nearest Neighbors (KNN)

Configuration:

```python
n_neighbors=5
```

Purpose:

* Distance-based classification
* Baseline model for comparison
* Captures local neighborhood information

---

## Model Performance

### Support Vector Machine (SVM)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 94%   |
| Precision | 94%   |
| Recall    | 94%   |
| F1-Score  | 94%   |

#### Class-wise Results

| Class     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| COVID     | 0.90      | 0.92   | 0.91     |
| Normal    | 0.92      | 0.90   | 0.91     |
| Pneumonia | 1.00      | 1.00   | 1.00     |

---

### Random Forest

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 88%   |
| Precision | 88%   |
| Recall    | 88%   |
| F1-Score  | 88%   |

#### Class-wise Results

| Class     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| COVID     | 0.83      | 0.84   | 0.83     |
| Normal    | 0.86      | 0.82   | 0.84     |
| Pneumonia | 0.95      | 0.98   | 0.97     |

---

### K-Nearest Neighbors

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 69%   |
| Precision | 72%   |
| Recall    | 69%   |
| F1-Score  | 68%   |

#### Class-wise Results

| Class     | Precision | Recall | F1-Score |
| --------- | --------- | ------ | -------- |
| COVID     | 0.70      | 0.46   | 0.55     |
| Normal    | 0.82      | 0.63   | 0.71     |
| Pneumonia | 0.63      | 1.00   | 0.78     |

---

## Ensemble Prediction

The deployed system supports soft-voting ensemble inference.

Process:

1. Obtain probability predictions from:

   * SVM
   * Random Forest
   * KNN

2. Average probabilities class-wise

3. Select the class with the highest average probability

This improves robustness by reducing dependence on a single classifier.

---

## Confidence Scores

Confidence scores are generated from model probabilities.

Example:

```json
{
  "COVID": 0.85,
  "Normal": 0.10,
  "Pneumonia": 0.05
}
```

These values are visualized in the frontend using pie charts.

---

## Feature Contribution Analysis

The system computes normalized contribution scores for:

* Texture (GLCM)
* Pattern (LBP)
* Structure (HOG)

These contributions are displayed using bar charts and are used to generate explanations.

---

## Explanation Generation

A rule-based explanation engine generates human-readable summaries using:

* Predicted disease class
* Dominant feature category

Examples:

* Texture-dominant prediction
* Pattern-dominant prediction
* Structure-dominant prediction

This improves interpretability of model predictions.

---

## API Endpoints

### Health Check

#### Request

```http
GET http://localhost:5000/health
```

#### Response

```json
{
  "status": "ok"
}
```

---

### Disease Prediction

#### Request

```http
POST http://localhost:5000/predict
```

#### Form Data

| Key   | Description          |
| ----- | -------------------- |
| file  | Chest X-ray image    |
| model | svm / rf / knn / all |

Example:

```text
file = chest_xray.png
model = svm
```

#### Response

```json
{
  "results": {
    "svm": {
      "class_name": "COVID"
    }
  },
  "features": {
    "texture": 0.72,
    "pattern": 0.15,
    "structure": 0.13
  },
  "explanation": "The model predicts COVID..."
}
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Suchitra-Siddharthan/lung-disease-detection.git
cd lung-disease-detection
```

---

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend runs at:

```text
http://localhost:5000
```

---

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

---

## Saved Models

```text
model/
├── svm_model.pkl
├── rf_model.pkl
├── knn_model.pkl
├── scaler.pkl
```

The saved StandardScaler is reused during inference to ensure the same preprocessing pipeline used during training is applied during prediction.

---

## Future Improvements

* Weighted ensemble learning
* Stacking-based ensemble models
* CNN-based feature extraction
* Additional lung disease classes
* Integration with radiography systems
* Cloud deployment
* Hospital information system integration
* Explainable AI methods such as SHAP and Grad-CAM

---

## Learning Outcomes

This project provided practical experience in:

* Medical image processing
* Feature engineering
* Machine learning model development
* Ensemble learning
* Model evaluation
* REST API development
* Full-stack application development
* Explainable AI concepts
* Healthcare-focused AI systems
