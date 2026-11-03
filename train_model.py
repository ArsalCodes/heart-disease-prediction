
import pandas as pd
import pickle

# Load dataset
df12 = pd.read_csv("heart-disease.csv")

# Check dataset
df12.head()
df12["target"].value_counts()

# Separate features and target
X = df12.drop(columns=["target"])
Y = df12["target"]

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)

# Build ML pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(X_train, Y_train)
print("Model training is completed")

# Model evaluation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Training accuracy
train_predictions = pipeline.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)

print(f"Training accuracy: {train_accuracy*100:.2f}%")

# Testing predictions
test_predictions = pipeline.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)

print(f"Testing accuracy: {test_accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(Y_test, test_predictions)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, test_predictions))

# Save trained pipeline
with open("heart_disease_pipeline.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Pipeline saved as heart_disease_pipeline.pkl")
