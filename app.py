import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import hashlib
import os
from datetime import date

# Set page layout
st.set_page_config(
    page_title="Smart Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------------
# AUTHENTICATION FUNCTIONS
# ---------------------------------------------------------
USER_DB_FILE = "users.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USER_DB_FILE):
        default_users = {
            "admin": {
                "password": hash_password("adminpassword"),
                "full_name": "System Administrator",
                "dob": "2000-01-01",
                "email": "admin@example.com"
            }
        }
        with open(USER_DB_FILE, "w") as f:
            json.dump(default_users, f)
        return default_users
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        default_users = {
            "admin": {
                "password": hash_password("adminpassword"),
                "full_name": "System Administrator",
                "dob": "2000-01-01",
                "email": "admin@example.com"
            }
        }
        with open(USER_DB_FILE, "w") as f:
            json.dump(default_users, f)
        return default_users

def save_user(username, password, full_name, dob, email):
    users = load_users()
    users[username] = {
        "password": hash_password(password),
        "full_name": full_name,
        "dob": str(dob),
        "email": email
    }
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def authenticate_user(username, password):
    users = load_users()
    if username in users:
        user_data = users[username]
        stored_hash = user_data["password"] if isinstance(user_data, dict) else user_data
        if stored_hash == hash_password(password):
            return True
    return False

# Initialize Session State Variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# ---------------------------------------------------------
# SPLIT-SCREEN LOGIN & REGISTER INTERFACE
# ---------------------------------------------------------
def render_login_page():
    left_col, right_col = st.columns([1, 1], gap="large")

    # LEFT SIDE: Welcome & Information Banner
    with left_col:
        st.markdown("## 🩺 Smart Disease Prediction System")
        st.caption("Clinical Diagnostic & Intelligence Portal")
        st.write("---")
        st.markdown("""
        ### Welcome to the Clinical Diagnostic Platform! 👋
        
        Our platform empowers users and healthcare professionals with **early disease risk assessments** powered by advanced ensemble machine learning algorithms.

        #### Key Features:
        * 📊 **Multi-Model Ensemble Intelligence:** Integrates SVM, k-Nearest Neighbors, Random Forest, and Gradient Boosting.
        * 💡 **Clinical Feature Analysis:** Real-time feature importance charts providing clear clinical metric insights.
        * 🔒 **Encrypted Security:** SHA-256 password hashing for secure user access.
        * 📋 **Comprehensive Symptom & Vital Metrics:** Evaluates age, blood pressure, cholesterol levels, and acute symptoms.
        """)

    # RIGHT SIDE: Interactive Login & Registration Portal
    with right_col:
        tab1, tab2 = st.tabs(["🔒 Sign In", "📝 Register New Account"])

        with tab1:
            st.subheader("Account Login")
            with st.form("login_form"):
                login_user = st.text_input("Username", key="login_user_input").strip()
                login_pass = st.text_input("Password", type="password", key="login_pass_input").strip()
                submit_login = st.form_submit_button("Log In", type="primary", use_container_width=True)

                if submit_login:
                    if authenticate_user(login_user, login_pass):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = login_user
                        st.success("Login successful! Loading dashboard...")
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password.")

        with tab2:
            st.subheader("Create a New Account")
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", key="reg_name_input").strip()
                reg_dob = st.date_input("Date of Birth (DOB)", min_value=date(1920, 1, 1), max_value=date.today(), value=date(2000, 1, 1))
                reg_email = st.text_input("Gmail / Email Address", key="reg_email_input").strip()
                reg_user = st.text_input("Choose Username", key="reg_user_input").strip()
                reg_pass = st.text_input("Choose Password", type="password", key="reg_pass_input").strip()
                reg_pass_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm_input").strip()
                
                submit_reg = st.form_submit_button("Register Account", use_container_width=True)

                if submit_reg:
                    users = load_users()
                    if not reg_name or not reg_email or not reg_user or not reg_pass:
                        st.warning("Please fill out all required fields.")
                    elif not reg_email.endswith("@gmail.com") and "@" not in reg_email:
                        st.error("Please enter a valid Gmail / Email address.")
                    elif reg_user in users:
                        st.error("Username already exists. Please pick a different username.")
                    elif reg_pass != reg_pass_confirm:
                        st.error("Passwords do not match. Please verify your password.")
                    else:
                        save_user(reg_user, reg_pass, reg_name, reg_dob, reg_email)
                        st.success("Account created successfully! You can now log in using the Sign In tab.")

# ---------------------------------------------------------
# MAIN DISEASE PREDICTION DASHBOARD
# ---------------------------------------------------------
def render_dashboard():
    # Sidebar Profile & Logout Section
    users = load_users()
    user_info = users.get(st.session_state['username'], {})
    display_name = user_info.get('full_name', st.session_state['username']) if isinstance(user_info, dict) else st.session_state['username']

    st.sidebar.header("👤 User Session")
    st.sidebar.info(f"Welcome, **{display_name}**!\n\nUsername: `{st.session_state['username']}`")
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.title("🩺 Smart Disease Prediction System")
    st.write("Enter patient demographics and health metrics to analyze diagnostic risk using an ensemble machine learning pipeline.")
    st.write("---")

    # Load Trained Model Artifacts
    try:
        ensemble_model = joblib.load("models/ensemble_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        disease_encoder = joblib.load("models/disease_encoder.pkl")
    except Exception as e:
        st.error("Model artifacts not found in `models/` directory. Please run `python train.py` first.")
        return

    st.subheader("📋 Patient Clinical Profile")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        fever = st.selectbox("Fever", ["No", "Yes"])

    with col2:
        cough = st.selectbox("Cough", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        breathing = st.selectbox("Difficulty Breathing", ["No", "Yes"])

    with col3:
        bp = st.selectbox("Blood Pressure Level", ["Low", "Normal", "High"])
        cholesterol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"])
        selected_disease = st.selectbox("Suspected Disease", disease_encoder.classes_)

    st.write("")
    if st.button("🚀 Run Diagnostic Analysis", type="primary", use_container_width=True):
        # Preprocess Input Data
        gender_val = 1 if gender == "Male" else 0
        fever_val = 1 if fever == "Yes" else 0
        cough_val = 1 if cough == "Yes" else 0
        fatigue_val = 1 if fatigue == "Yes" else 0
        breathing_val = 1 if breathing == "Yes" else 0

        bp_map = {"Low": 0, "Normal": 1, "High": 2}
        chol_map = {"Low": 0, "Normal": 1, "High": 2}

        bp_val = bp_map[bp]
        chol_val = chol_map[cholesterol]
        disease_val = disease_encoder.transform([selected_disease])[0]

        input_df = pd.DataFrame([[
            age, gender_val, fever_val, cough_val, fatigue_val,
            breathing_val, bp_val, chol_val, disease_val
        ]], columns=[
            'Age', 'Gender', 'Fever', 'Cough', 'Fatigue', 
            'Difficulty Breathing', 'Blood Pressure', 'Cholesterol Level', 'Disease_Label'
        ])

        scaled_input = scaler.transform(input_df)
        prediction = ensemble_model.predict(scaled_input)[0]
        prediction_proba = ensemble_model.predict_proba(scaled_input)[0]

        st.write("---")
        st.subheader("📊 Diagnostic Assessment Results")
        
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if prediction == 1:
                st.error(f"⚠️ **High Risk Detected** for **{selected_disease}**")
            else:
                st.success(f"✅ **Low Risk / Negative Result** for **{selected_disease}**")

        with res_col2:
            confidence_val = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            st.metric("Model Confidence Level", f"{confidence_val * 100:.2f}%")

        # Clinical Feature Importance Visualization
        st.write("---")
        st.subheader("📊 Clinical Feature Importance")

        with st.spinner("Calculating feature contributions..."):
            try:
                rf_model = None
                if hasattr(ensemble_model, 'named_estimators_'):
                    rf_model = ensemble_model.named_estimators_.get('rf', list(ensemble_model.named_estimators_.values())[0])
                elif hasattr(ensemble_model, 'estimators_'):
                    rf_model = ensemble_model.estimators_[0]
                else:
                    rf_model = ensemble_model

                if hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                    feature_names = input_df.columns
                    
                    importance_df = pd.DataFrame({
                        'Clinical Feature': feature_names,
                        'Importance Score': importances
                    }).sort_values(by='Importance Score', ascending=True)

                    st.bar_chart(
                        data=importance_df.set_index('Clinical Feature'),
                        horizontal=True
                    )
                    st.caption("This chart highlights the relative importance and influence of each clinical feature within the model.")
                else:
                    st.info("Feature importance metrics generated across the ensemble pipeline.")
            except Exception as e:
                st.warning("Feature importance chart could not be rendered.")

# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------
if not st.session_state["authenticated"]:
    render_login_page()
else:
    render_dashboard()