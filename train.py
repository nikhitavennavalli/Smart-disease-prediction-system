import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

def preprocess_and_train(dataset_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    df = df.dropna()

    # Binary/Categorical Encoding
    binary_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Gender Encoding
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Ordinal Encoding for Blood Pressure & Cholesterol
    ordinal_map = {'Low': 0, 'Normal': 1, 'High': 2}
    df['Blood Pressure'] = df['Blood Pressure'].map(ordinal_map)
    df['Cholesterol Level'] = df['Cholesterol Level'].map(ordinal_map)

    # Encode Target Outcome (Positive = 1, Negative = 0)
    df['Outcome Variable'] = df['Outcome Variable'].map({'Positive': 1, 'Negative': 0})

    # Encode Multi-Disease Category Label
    disease_encoder = LabelEncoder()
    df['Disease_Label'] = disease_encoder.fit_transform(df['Disease'])

    # Feature Matrix & Target Setup
    feature_cols = [
        'Age', 'Gender', 'Fever', 'Cough', 'Fatigue', 
        'Difficulty Breathing', 'Blood Pressure', 'Cholesterol Level', 'Disease_Label'
    ]
    
    X = df[feature_cols]
    y = df['Outcome Variable']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Base Models
    svm_model = SVC(probability=True, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Soft Voting Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('knn', knn_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft'
    )

    ensemble.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    print("=== MODEL EVALUATION METRICS ===")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # Feature Importance (Explainability)
    rf_model.fit(X_train_scaled, y_train)
    importances = rf_model.feature_importances_
    print("\n=== FEATURE IMPORTANCE (XAI) ===")
    for feature, imp in zip(feature_cols, importances):
        print(f"{feature}: {imp:.4f}")

    # Save artifacts
    joblib.dump(ensemble, 'models/ensemble_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(disease_encoder, 'models/disease_encoder.pkl')
    print("\nAll trained artifacts saved successfully to models/")

if __name__ == "__main__":
    preprocess_and_train('data/Disease_symptom_and_patient_profile_dataset.csv')