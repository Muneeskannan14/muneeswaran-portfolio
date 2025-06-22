import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import warnings
import hashlib
import time
import datetime
import re
from io import BytesIO, StringIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Autism Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

# Define models globally
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42, probability=True, kernel='rbf'),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, max_depth=3),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Voting Classifier": VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=50)),
            ('xgb', XGBClassifier(random_state=42, n_estimators=50))
        ],
        voting='soft'
    )
}

# Initialize session state
if 'users' not in st.session_state:
    st.session_state['users'] = {}
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None
if 'remember_me' not in st.session_state:
    st.session_state['remember_me'] = False
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False
if 'activity_log' not in st.session_state:
    st.session_state['activity_log'] = []
if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = {}
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'label_encoders' not in st.session_state:
    st.session_state['label_encoders'] = None

# Enhanced Custom CSS with Dark Mode
def get_css():
    if st.session_state['dark_mode']:
        return """
        <style>
        .main {background-color: #1e1e1e; color: #ffffff;}
        .main-title {font-size: 36px; color: #61dafb; font-weight: bold; text-align: center; margin-bottom: 20px;}
        .subheader {font-size: 24px; color: #61dafb; font-weight: 600;}
        .card {background-color: #2d2d2d; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); margin-bottom: 20px;}
        .sidebar .sidebar-content {background-color: #252525; color: #ffffff;}
        .stButton>button {background-color: #61dafb; color: #1e1e1e; border-radius: 5px;}
        .stButton>button:hover {background-color: #21a1f1;}
        .auth-box {background-color: #2d2d2d; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
        .profile-card {background-color: #333333; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);}
        </style>
        """
    else:
        return """
        <style>
        .main {background-color: #f5f6fa; color: #1e1e1e;}
        .main-title {font-size: 36px; color: #1f2a44; font-weight: bold; text-align: center; margin-bottom: 20px;}
        .subheader {font-size: 24px; color: #34495e; font-weight: 600;}
        .card {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;}
        .sidebar .sidebar-content {background-color: #ecf0f1; color: #1e1e1e;}
        .stButton>button {background-color: #3498db; color: white; border-radius: 5px;}
        .stButton>button:hover {background-color: #2980b9;}
        .auth-box {background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
        .profile-card {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        </style>
        """

st.markdown(get_css(), unsafe_allow_html=True)

# Helper Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password_strength(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"
    return True, "Password strength: Strong"

def signup(username, email, password, confirm_password):
    if username in st.session_state['users']:
        return False, "Username already exists."
    if password != confirm_password:
        return False, "Passwords do not match."
    is_strong, message = check_password_strength(password)
    if not is_strong:
        return False, message
    st.session_state['users'][username] = {
        "password": hash_password(password),
        "email": email,
        "last_login": None
    }
    st.session_state['activity_log'].append(f"{username} signed up at {datetime.datetime.now()}")
    return True, "Signup successful! Please login."

def login(username, password):
    if username in st.session_state['users']:
        user_data = st.session_state['users'][username]
        if user_data["password"] == hash_password(password):
            st.session_state['authenticated'] = True
            st.session_state['current_user'] = username
            user_data["last_login"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state['activity_log'].append(f"{username} logged in at {datetime.datetime.now()}")
            return True, f"Welcome back, {username}!"
    return False, "Invalid username or password."

def delete_account(username):
    if username in st.session_state['users']:
        del st.session_state['users'][username]
        st.session_state['activity_log'].append(f"{username} deleted account at {datetime.datetime.now()}")
        return True, "Account deleted successfully."
    return False, "Account not found."

# Preprocessing Function
@st.cache_data
def preprocess_data(df):
    try:
        le_dict = {}
        categorical_cols = df.select_dtypes(include=['object']).columns
        df_processed = df.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
        
        if 'Class/ASD' not in df_processed.columns:
            raise ValueError("Target column 'Class/ASD' not found in dataset.")
        
        X = df_processed.drop('Class/ASD', axis=1)
        y = df_processed['Class/ASD']
        
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)
        
        # Split data and store in session state
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler
        st.session_state['label_encoders'] = le_dict
        st.session_state['feature_names'] = X.columns.tolist()
        
        return X_scaled, y_res, X.columns
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None, None, None

# Authentication UI
if not st.session_state['authenticated']:
    st.markdown('<div class="main-title">Autism Prediction Dashboard</div>', unsafe_allow_html=True)
    auth_option = st.sidebar.radio("Authentication", ["Login", "Signup", "Forgot Password"])
    
    with st.container():
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        if auth_option == "Signup":
            st.subheader("✨ Create New Account")
            signup_username = st.text_input("Username", help="Choose a unique username")
            signup_email = st.text_input("Email", help="For account recovery")
            signup_password = st.text_input("Password", type="password", help="Min 8 chars, include upper, lower, number")
            signup_confirm = st.text_input("Confirm Password", type="password")
            if st.button("Signup"):
                success, message = signup(signup_username, signup_email, signup_password, signup_confirm)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        elif auth_option == "Login":
            st.subheader("🔑 Login")
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            st.session_state['remember_me'] = st.checkbox("Remember Me")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    success, message = login(login_username, login_password)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
            with col2:
                if st.button("Guest Login"):
                    st.session_state['authenticated'] = True
                    st.session_state['current_user'] = "Guest"
                    st.session_state['activity_log'].append(f"Guest logged in at {datetime.datetime.now()}")
                    st.success("Logged in as Guest")
                    time.sleep(1)
                    st.rerun()
        
        elif auth_option == "Forgot Password":
            st.subheader("🔄 Reset Password")
            reset_username = st.text_input("Username")
            reset_email = st.text_input("Registered Email")
            if st.button("Send Reset Link"):
                if reset_username in st.session_state['users'] and \
                   st.session_state['users'][reset_username]["email"] == reset_email:
                    st.success("Password reset link sent to your email! (Simulation)")
                else:
                    st.error("Username or email not found.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Sidebar
st.sidebar.markdown(f"👤 Logged in as: {st.session_state['current_user']}")
st.sidebar.checkbox("Dark Mode", value=st.session_state['dark_mode'], key="dark_mode_switch",
                    on_change=lambda: st.session_state.update({'dark_mode': not st.session_state['dark_mode']}))

if st.session_state['current_user'] != "Guest":
    with st.sidebar.expander("User Profile"):
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        user_data = st.session_state['users'][st.session_state['current_user']]
        st.write(f"Email: {user_data['email']}")
        st.write(f"Last Login: {user_data['last_login'] or 'Never'}")
        st.write(f"Remember Me: {'Yes' if st.session_state['remember_me'] else 'No'}")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button("Delete Account", key="delete_account"):
            success, message = delete_account(st.session_state['current_user'])
            if success:
                st.success(message)
                st.session_state['authenticated'] = False
                st.session_state['current_user'] = None
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)

if st.sidebar.button("🚪 Logout"):
    st.session_state['activity_log'].append(f"{st.session_state['current_user']} logged out at {datetime.datetime.now()}")
    st.session_state['authenticated'] = False
    st.session_state['current_user'] = None
    st.rerun()

# Navigation
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Select Page", [
    "Dashboard", "Data Upload", "EDA", "Visualization", "Model Training",
    "Model Comparison", "Autism Quiz Game", "Real-Time Prediction"
])

# Dashboard Page
if page == "Dashboard":
    st.markdown('<div class="main-title">🏠 Dashboard Overview</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Users")
        st.write(len(st.session_state['users']))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🕒 Last Login")
        user_data = st.session_state['users'].get(st.session_state['current_user'], {})
        st.write(user_data.get('last_login', 'N/A') if st.session_state['current_user'] != "Guest" else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📜 Activity")
        st.write(len(st.session_state['activity_log']))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recent Activity")
    for log in st.session_state['activity_log'][-5:][::-1]:
        st.write(f"• {log}")
    st.markdown('</div>', unsafe_allow_html=True)

# Data Upload
elif page == "Data Upload":
    st.markdown('<div class="main-title">📤 Data Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="uploader")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.success("File uploaded successfully!")
            st.write("Dataset Preview", df.head())
            # Preprocess data immediately after upload
            preprocess_data(df)
            st.success("Data preprocessed successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to proceed.")
    st.markdown('</div>', unsafe_allow_html=True)

# Ensure data is loaded for subsequent pages
if 'df' not in st.session_state or st.session_state['df'] is None:
    if page not in ["Dashboard", "Data Upload", "Autism Quiz Game"]:
        st.warning("No data uploaded yet. Please go to the 'Data Upload' page first.")
        st.stop()

# EDA Page
elif page == "EDA":
    st.markdown('<div class="main-title">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df = st.session_state['df']
    eda_option = st.selectbox("Select EDA Technique", [
        "Basic Statistics", "Missing Values", "Outlier Detection", 
        "Feature Distribution", "Skewness & Kurtosis", "Categorical Analysis",
        "Duplicate Records"
    ])
    
    if eda_option == "Basic Statistics":
        st.write("Dataset Shape:", df.shape)
        st.write("Data Types:", df.dtypes)
        st.write("Descriptive Statistics:", df.describe())
    
    elif eda_option == "Missing Values":
        missing = df.isnull().sum()
        st.write("Missing Values Count:", missing)
        if missing.any():
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title("Missing Values Heatmap")
            st.pyplot(fig)
        else:
            st.success("No missing values found!")
    
    elif eda_option == "Outlier Detection":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select Column", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_col], ax=ax)
        ax.set_title(f"Outlier Detection - {selected_col}")
        st.pyplot(fig)
        q1, q3 = df[selected_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = df[(df[selected_col] < q1 - 1.5 * iqr) | (df[selected_col] > q3 + 1.5 * iqr)][selected_col]
        st.write(f"Number of Outliers: {len(outliers)}")
    
    elif eda_option == "Feature Distribution":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select Column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)
    
    elif eda_option == "Skewness & Kurtosis":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skew_kurt = pd.DataFrame({
            'Skewness': df[numeric_cols].skew(),
            'Kurtosis': df[numeric_cols].kurtosis()
        })
        st.write("Skewness & Kurtosis:", skew_kurt)
    
    elif eda_option == "Categorical Analysis":
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            selected_col = st.selectbox("Select Column", cat_cols)
            st.write(df[selected_col].value_counts())
            fig, ax = plt.subplots()
            df[selected_col].value_counts().plot.bar(ax=ax)
            ax.set_title(f"Value Counts of {selected_col}")
            st.pyplot(fig)
        else:
            st.info("No categorical columns found.")
    
    elif eda_option == "Duplicate Records":
        duplicates = df.duplicated().sum()
        st.write(f"Number of Duplicate Records: {duplicates}")
        if duplicates > 0:
            st.write("Sample Duplicate Rows:", df[df.duplicated()].head())
    st.markdown('</div>', unsafe_allow_html=True)

# Visualization Page
elif page == "Visualization":
    st.markdown('<div class="main-title">📊 Data Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df = st.session_state['df']
    viz_type = st.selectbox("Select Visualization", [
        "Pairplot", "Box Plot", "Pie Chart", "Distribution of ASD Classes",
        "Age Distribution", "Score Distribution", "Correlation Heatmap"
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if viz_type == "Pairplot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        pair_df = df[numeric_cols.tolist() + ['Class/ASD']]
        sns.pairplot(pair_df, hue='Class/ASD')
        st.pyplot(plt.gcf())
    
    elif viz_type == "Box Plot":
        score_cols = [col for col in df.columns if col.lower().startswith("a") and len(col) <= 3 or col in [
            "Notices Small Sounds", "Focus on Details", "Multitasking Difficulty",
            "Trouble Resuming After Interruption", "Social Situations Are Easy",
            "Notices Details Others Miss", "Prefers Library Over Party",
            "Imagination Challenges", "Fascination With Dates", "Difficulty Reading Intentions"
        ]]
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
        df_filtered = df[score_cols].dropna(axis=1, how='all')
        if not df_filtered.empty:
            df_filtered.boxplot(ax=ax)
            ax.set_title("Box Plot of Autism Screening Scores")
            st.pyplot(fig)
        else:
            st.warning("No numeric score columns available for box plot.")
    
    elif viz_type == "Pie Chart":
        df['Class/ASD'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_title("ASD Class Distribution")
        st.pyplot(fig)
    
    elif viz_type == "Distribution of ASD Classes":
        sns.countplot(data=df, x='Class/ASD', ax=ax)
        ax.set_title("Distribution of ASD Classes")
        st.pyplot(fig)
    
    elif viz_type == "Age Distribution":
        sns.histplot(data=df, x='age', bins=30, ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
    
    elif viz_type == "Score Distribution":
        score_cols = [col for col in df.columns if col in [
            "Notices Small Sounds", "Focus on Details", "Multitasking Difficulty",
            "Trouble Resuming After Interruption", "Social Situations Are Easy",
            "Notices Details Others Miss", "Prefers Library Over Party",
            "Imagination Challenges", "Fascination With Dates", "Difficulty Reading Intentions"]]
        df[score_cols] = df[score_cols].apply(pd.to_numeric, errors='coerce')
        df_filtered = df[score_cols].dropna(axis=1, how='all')
        if not df_filtered.empty:
            df_filtered.sum().plot(kind='bar', ax=ax)
            ax.set_title("Distribution of Screening Scores")
            ax.set_ylabel("Total Score Across Samples")
            st.pyplot(fig)
        else:
            st.warning("No numeric score columns available for distribution plot.")
    
    elif viz_type == "Correlation Heatmap":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Model Training Page
elif page == "Model Training":
    st.markdown('<div class="main-title">🧠 Model Training and Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state['X_train'] is None or len(st.session_state['X_train']) == 0:
        st.warning("Training data not available. Please upload and preprocess data first.")
        st.stop()

    
    model_choice = st.selectbox("Select Model", list(models.keys()))
    selected_model = models[model_choice]
    
    if st.button("Train Model"):
        with st.spinner(f"Training {model_choice}..."):
            try:
                selected_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['trained_models'][model_choice] = selected_model
                y_pred = selected_model.predict(st.session_state['X_test'])
                
                accuracy = accuracy_score(st.session_state['y_test'], y_pred)
                report = classification_report(st.session_state['y_test'], y_pred)
                
                st.write(f"**Accuracy:** {accuracy:.2f}")
                st.text("Classification Report:")
                st.text(report)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(st.session_state['y_test'], y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"Confusion Matrix - {model_choice}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                
                if hasattr(selected_model, 'feature_importances_'):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance = pd.DataFrame({
                        'feature': st.session_state['feature_names'],
                        'importance': selected_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    sns.barplot(x='importance', y='feature', data=importance, ax=ax)
                    ax.set_title(f"Feature Importance - {model_choice}")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Training error: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# Model Comparison Page
elif page == "Model Comparison":
    st.markdown('<div class="main-title">⚖️ Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state['X_train'] is None or st.session_state['y_train'] is None:
        st.warning("Training data not available. Please upload and preprocess data first.")
        st.stop()
    
    if st.button("Compare All Models"):
        with st.spinner("Training all models..."):
            results = {}
            for name, model in models.items():
                try:
                    model.fit(st.session_state['X_train'], st.session_state['y_train'])
                    st.session_state['trained_models'][name] = model
                    y_pred = model.predict(st.session_state['X_test'])
                    results[name] = accuracy_score(st.session_state['y_test'], y_pred)
                except Exception as e:
                    st.warning(f"Error training {name}: {str(e)}")
                    results[name] = 0.0
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=list(results.keys()), y=list(results.values()), ax=ax)
            ax.set_title("Model Accuracy Comparison")
            ax.set_ylabel("Accuracy")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.subheader("Detailed Results")
            results_df = pd.DataFrame({
                "Model": results.keys(),
                "Accuracy": results.values()
            }).sort_values("Accuracy", ascending=False)
            st.write(results_df)
    st.markdown('</div>', unsafe_allow_html=True)

# Autism Quiz Game
elif page == "Autism Quiz Game":
    st.markdown('<div class="main-title">🎲 Autism Quiz Game</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    language = st.selectbox("Select Language / மொழி தேர்வு", ["English", "தமிழ்"])
    player_type = st.radio(
        "Who is playing? / யார் விளையாடுகிறார்கள்?",
        ["Adult", "Child"] if language == "English" else ["வயது வந்தவர்", "குழந்தை"]
    )
    
    if language == "English":
        st.subheader("Instructions")
        st.write("Answer honestly. Based on your responses, this game gives a fun, informal assessment of autism traits.")
    else:
        st.subheader("வழிமுறைகள்")
        st.write("உங்கள் பதில்களை நேர்மையாக அளியுங்கள். இது ஒரு சுலபமான ஆட்டிசம் பண்புகள் மதிப்பீட்டைக் கொடுக்கும்.")

    adult_questions = [
        {"question_en": "Do you find it hard to start a conversation with someone new?", "question_ta": "புதிய நபர்களுடன் உரையாடலைத் தொடங்குவது உங்களுக்கு கடினமாக இருக்கிறதா?", "options": ["Yes", "Sometimes", "No"], "score": {"Yes": 2, "Sometimes": 1, "No": 0}},
        {"question_en": "Do you get uncomfortable with unexpected changes in plans?", "question_ta": "திடீர் திட்ட மாற்றங்களில் நீங்கள் சிரமப்படுகிறீர்களா?", "options": ["Yes", "A little", "No"], "score": {"Yes": 2, "A little": 1, "No": 0}},
        {"question_en": "Do loud sounds or bright lights bother you?", "question_ta": "மிகுந்த சத்தங்கள் அல்லது ஒளிகள் உங்களைத் தொந்தரவு செய்கிறதா?", "options": ["Yes", "Sometimes", "No"], "score": {"Yes": 2, "Sometimes": 1, "No": 0}},
        {"question_en": "Do you prefer routines and structure over flexibility?", "question_ta": "நீங்கள் கட்டுப்பாடு மற்றும் திட்டமிடலுடன் செயல்பட விரும்புகிறீர்களா?", "options": ["Strongly Agree", "Somewhat Agree", "Disagree"], "score": {"Strongly Agree": 2, "Somewhat Agree": 1, "Disagree": 0}},
        {"question_en": "Do you struggle to understand jokes or sarcasm?", "question_ta": "நகைச்சுவை அல்லது கிண்டல்களை புரிந்து கொள்வதில் சிரமம் உள்ளதா?", "options": ["Often", "Sometimes", "Rarely"], "score": {"Often": 2, "Sometimes": 1, "Rarely": 0}},
        {"question_en": "Do you feel more comfortable being alone than with people?", "question_ta": "மக்களுடன் இருப்பதைவிட தனியாக இருப்பது உங்கள் விருப்பமா?", "options": ["Always", "Sometimes", "Never"], "score": {"Always": 2, "Sometimes": 1, "Never": 0}}
    ]

    child_questions = [
        {"question_en": "Do you like playing by yourself more than with others?", "question_ta": "நண்பர்களுடன் விளையாடுவதற்குப் பதிலாக தனியாக விளையாட விரும்புகிறீர்களா?", "options": ["Yes", "Sometimes", "No"], "score": {"Yes": 2, "Sometimes": 1, "No": 0}},
        {"question_en": "Do loud sounds scare you?", "question_ta": "உயர்ந்த சத்தங்கள் உங்களுக்கு பயமளிக்கிறதா?", "options": ["Yes", "A little", "No"], "score": {"Yes": 2, "A little": 1, "No": 0}},
        {"question_en": "Do you like to play the same game again and again?", "question_ta": "அதே விளையாட்டை மீண்டும் மீண்டும் விளையாட விரும்புகிறீர்களா?", "options": ["Yes", "Sometimes", "No"], "score": {"Yes": 2, "Sometimes": 1, "No": 0}},
        {"question_en": "Do you look at people’s eyes when they talk to you?", "question_ta": "மற்றவர்கள் பேசும்போது அவர்களின் கண்களைப் பார்ப்பதா?", "options": ["No", "Sometimes", "Yes"], "score": {"No": 2, "Sometimes": 1, "Yes": 0}},
        {"question_en": "Do you flap your hands or spin around when excited?", "question_ta": "உற்சாகமாக இருக்கும்போது உங்கள் கைகளை அசைப்பதா அல்லது சுழற்சி செய்வதா?", "options": ["Yes", "Sometimes", "No"], "score": {"Yes": 2, "Sometimes": 1, "No": 0}},
        {"question_en": "Do you get very upset if your routine changes?", "question_ta": "உங்கள் வழக்கமான நடவடிக்கைகள் மாற்றப்படும் போது நீங்கள் மிகவும் கவலையடைவீர்களா?", "options": ["Yes", "Sometimes", "No"], "score": {"Yes": 2, "Sometimes": 1, "No": 0}}
    ]

    questions = adult_questions if player_type in ["Adult", "வயது வந்தவர்"] else child_questions

    score = 0
    with st.form("quiz_form"):
        for i, q in enumerate(questions):
            q_text = q["question_en"] if language == "English" else q["question_ta"]
            answer = st.radio(q_text, q["options"], key=f"q{i}")
            if answer:
                score += q["score"][answer]
            st.progress((i + 1) / len(questions))
            time.sleep(0.1)
        
        submit = st.form_submit_button("Submit" if language == "English" else "சமர்ப்பிக்கவும்")

    if submit:
        st.markdown("---")
        if language == "English":
            st.subheader("📝 Result")
        else:
            st.subheader("📝 முடிவு")
        
        if score <= 4:
            result = "Low likelihood of autism traits." if language == "English" else "ஆட்டிசம் பண்புகள் குறைவாக இருக்கலாம்."
            st.success("🎉 " + result)
        elif score <= 8:
            result = "Some traits associated with autism may be present." if language == "English" else "சில ஆட்டிசம் தொடர்பான பண்புகள் இருக்கலாம்."
            st.info("😊 " + result)
        else:
            result = "High likelihood of autism traits. Please consult a specialist." if language == "English" else "உயர் சாத்தியமுள்ள ஆட்டிசம் பண்புகள். தயவுசெய்து மருத்துவரை அணுகவும்."
            st.warning("🚨 " + result)
        
        st.write(f"**{('Total Score' if language == 'English' else 'மொத்த மதிப்பெண்')}: {score}**")
        st.caption("This game is for educational purposes only and does not replace medical diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)

# Real-Time Prediction Page
elif page == "Real-Time Prediction":
    st.markdown('<div class="main-title">🔮 Real-Time Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Check if required training data exists
    missing_training_data = (
        'X_train' not in st.session_state or
        'y_train' not in st.session_state or
        st.session_state['X_train'] is None or
        st.session_state['y_train'] is None or
        len(st.session_state['X_train']) == 0
    )
    if missing_training_data:
        st.warning("Training data not available. Please upload and preprocess data first.")
        st.stop()

    df = st.session_state['df']

    if not st.session_state['trained_models']:
        st.warning("No models trained yet. Please train a model first on the 'Model Training' page.")
        st.stop()

    if st.session_state['feature_names'] is None:
        st.error("Feature names not found. Please upload and preprocess data first.")
        st.stop()

    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []

    model_choice = st.selectbox("Select Trained Model", list(st.session_state['trained_models'].keys()))

    input_data = {}
    a_score_explanations = {
        "A1_Score": "Social interaction difficulties",
        "A2_Score": "Resistance to change",
        "A3_Score": "Lack of empathy",
        "A4_Score": "Communication issues",
        "A5_Score": "Fixation on routines",
        "A6_Score": "Repetitive behavior",
        "A7_Score": "Sensory sensitivity",
        "A8_Score": "Trouble understanding sarcasm",
        "A9_Score": "Preference for solitude",
        "A10_Score": "Challenges with imagination"
    }

    with st.form(key="prediction_form"):
        st.write("### Input Features")
        for feature in st.session_state['feature_names']:
            if feature in a_score_explanations:
                tooltip = a_score_explanations[feature]
                input_data[feature] = int(st.radio(f"{feature} ❔", [0, 1], horizontal=True, help=tooltip))
            elif feature in df.select_dtypes(include=['object']).columns:
                options = df[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(f"{feature}", options)
            else:
                try:
                    cleaned_col = pd.to_numeric(df[feature], errors='coerce')
                    min_val = float(cleaned_col.min(skipna=True))
                    max_val = float(cleaned_col.max(skipna=True))
                    mean_val = float(cleaned_col.mean(skipna=True))
                    input_data[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
                except:
                    st.warning(f"Skipping {feature} due to invalid data.")

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        try:
            selected_model = st.session_state['trained_models'][model_choice]
            input_df = pd.DataFrame([input_data])[st.session_state['feature_names']]
            for col in st.session_state['label_encoders']:
                le = st.session_state['label_encoders'][col]
                input_df[col] = le.transform(input_df[col].astype(str))
            input_scaled = st.session_state['scaler'].transform(input_df)
            prediction = selected_model.predict(input_scaled)[0]
            prediction_prob = selected_model.predict_proba(input_scaled)[0]

            result_label = "ASD Positive" if prediction == 1 else "ASD Negative"
            advice = "Please consult a medical specialist for further evaluation." if prediction == 1 else "No indication of ASD based on the input. Keep monitoring."

            st.subheader("Prediction Results")
            st.write(f"**Prediction Label:** {result_label}")
            st.write(f"**Prediction Score (0 or 1):** {int(prediction)}")
            st.write(f"**Probability (Negative):** {prediction_prob[0]:.2f}")
            st.write(f"**Probability (Positive):** {prediction_prob[1]:.2f}")

            st.markdown("---")
            st.subheader("Summary")
            st.markdown(f"""
                - 🧠 **Result:** `{result_label}`
                - 📊 **Score:** `{int(prediction)}`
                - 🔍 **Confidence:** {prediction_prob[int(prediction)]:.2f}
                - 💡 **Advice:** {advice}
            """)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=['Negative', 'Positive'], y=prediction_prob, ax=ax)
            ax.set_title("Prediction Probabilities")
            ax.set_ylabel("Probability")
            st.pyplot(fig)

            # Explanation for A1–A10
            st.markdown("### 🔍 Input Reasoning (A1–A10)")
            a_scores_used = {k: v for k, v in input_data.items() if k in a_score_explanations}
            for feature, value in a_scores_used.items():
                reason = a_score_explanations.get(feature, "No explanation available.")
                emoji = "✅" if value == 0 else "⚠️"
                st.markdown(f"- **{feature}**: {value} {emoji} — _{reason}_")

            # Radar Chart
            if len(a_scores_used) > 2:
                st.markdown("### 🕸️ Radar Chart of A1–A10 Scores")
                features = list(a_scores_used.keys())
                values = list(a_scores_used.values()) + [list(a_scores_used.values())[0]]
                angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist() + [0]
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.plot(angles, values, color='blue', linewidth=2)
                ax.fill(angles, values, color='skyblue', alpha=0.4)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(features, size=10)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["0", "1"])
                ax.set_title("A1–A10 Score Profile", size=14)
                st.pyplot(fig)

            # Save result
            result_dict = {
                "Prediction Label": result_label,
                "Prediction Score": int(prediction),
                "Probability (Negative)": round(prediction_prob[0], 2),
                "Probability (Positive)": round(prediction_prob[1], 2),
                "Advice": advice
            }

            # Append history
            st.session_state['prediction_history'].append({
                "User": st.session_state['current_user'],
                "Model": model_choice,
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **result_dict,
                **a_scores_used
            })

            # CSV
            csv_buffer = StringIO()
            pd.DataFrame([result_dict]).to_csv(csv_buffer, index=False)
            st.download_button("📊 Download Result as CSV", csv_buffer.getvalue(), "autism_prediction_result.csv", mime="text/csv")

            # PDF
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            c.setFont("Helvetica", 12)
            text_obj = c.beginText(40, 750)
            text_obj.textLine("Autism Prediction Report")
            text_obj.textLine("========================")
            for key, value in result_dict.items():
                text_obj.textLine(f"{key}: {value}")
            c.drawText(text_obj)
            c.showPage()
            c.save()
            st.download_button("📄 Download Result as PDF", data=pdf_buffer.getvalue(), file_name="autism_prediction_result.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

    # Show history table
    if st.session_state['prediction_history']:
        st.markdown("## 📚 Prediction History (This Session)")
        hist_df = pd.DataFrame(st.session_state['prediction_history'])
        st.dataframe(hist_df, use_container_width=True)

# Footer
st.markdown(f"<hr><small>Developed by MUNEESWARAN | {datetime.datetime.now().year}</small>", unsafe_allow_html=True)