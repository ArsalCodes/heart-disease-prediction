# ❤️ Heart Disease Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to predict the likelihood of heart disease based on patient medical information.

A Logistic Regression classification model is trained using a heart disease dataset. The model is integrated into an interactive web application built with Streamlit, allowing users to enter patient information and receive a prediction.

**Disclaimer:** This application is an educational Machine Learning demonstration and is not a substitute for professional medical diagnosis or advice.

## 🎯 Project Objectives

- Develop a Machine Learning classification model to predict heart disease.
- Apply feature scaling using StandardScaler.
- Build a Scikit-learn Pipeline for preprocessing and model training.
- Evaluate model performance using accuracy, confusion matrix, and classification report.
- Deploy an interactive prediction interface using Streamlit.

## 🛠️ Technologies & Libraries

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- StandardScaler
- Streamlit
- Pickle

## 📊 Machine Learning Workflow

1. Load and explore the heart disease dataset.
2. Separate independent features and target variable.
3. Split the dataset into training and testing sets (80:20).
4. Apply StandardScaler to standardize numerical features.
5. Train a Logistic Regression classification model.
6. Evaluate the model using training accuracy, testing accuracy, confusion matrix, and classification report.
7. Save the trained pipeline using Pickle.
8. Integrate the trained model into a Streamlit web application.

## 📈 Model Performance

The model achieved the following initial testing results:

| Evaluation Metric | Result |
|---|---:|
| Testing Accuracy | 79% |
| Test Samples | 61 |
| Macro Average F1-score | 0.79 |
| Weighted Average F1-score | 0.79 |

*Note: These are results from the current train-test split. Performance should be further validated before any real-world application.*

## 🖥️ Streamlit Web Application

The interactive application allows users to enter patient information, including:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- Oldpeak
- Slope
- Number of major vessels
- Thal

After entering the information, users can click the **Predict** button to receive the model's classification output.

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── train_model.py
├── heart_disease_pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/ArsalCodes/heart-disease-prediction
```

### 2. Navigate to the project directory

```bash
cd heart-disease-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 📦 Saved Model

The trained Machine Learning pipeline is saved as:

`heart_disease_pipeline.pkl`

The pipeline includes feature scaling and Logistic Regression, ensuring that new input data receives the same preprocessing during prediction as it did during model training.

## 🚀 Future Improvements

- Improve model evaluation and validation.
- Explore other classification algorithms.
- Add interactive visualizations.
- Improve the user interface.
- Deploy the application using Streamlit Community Cloud.

## 👨‍💻 Author

**Muhammad Arsalan Khan**

Electronics Engineer | Aspiring Data Analyst & Data Scientist

GitHub: https://github.com/ArsalCodes

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.
