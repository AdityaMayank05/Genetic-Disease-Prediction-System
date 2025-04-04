import streamlit as st
import pandas as pd
import numpy as np
from app import preprocess_data, prepare_features_and_target, train_random_forest, train_xgboost, create_neural_network
import joblib
import os
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Genetic Disorder Prediction",
    page_icon="🧬",
    layout="wide"
)

# Title and description
st.title("🧬 Genetic Disorder Prediction System")
st.markdown("""
This application helps predict genetic disorders based on patient data. 
Please fill in the patient information below and select a model for prediction.
""")

def load_or_train_models():
    """Load pre-trained models or train new ones if they don't exist"""
    models = {}
    
    # Load training data
    train_df = pd.read_csv("train.csv")
    processed_df = preprocess_data(train_df)
    X, y, scaler = prepare_features_and_target(processed_df)
    
    # Store feature names and order
    feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
    
    # Check if models exist, if not train them
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Random Forest
    if os.path.exists('models/random_forest.joblib'):
        models['Random Forest'] = joblib.load('models/random_forest.joblib')
    else:
        models['Random Forest'] = train_random_forest(X, y)
        joblib.dump(models['Random Forest'], 'models/random_forest.joblib')
    
    # XGBoost
    if os.path.exists('models/xgboost.joblib'):
        models['XGBoost'] = joblib.load('models/xgboost.joblib')
    else:
        models['XGBoost'] = train_xgboost(X, y)
        joblib.dump(models['XGBoost'], 'models/xgboost.joblib')
    
    # Neural Network
    if os.path.exists('models/neural_network.keras'):
        models['Neural Network'] = tf.keras.models.load_model('models/neural_network.keras')
    else:
        num_classes = len(np.unique(y))
        models['Neural Network'] = create_neural_network((X.shape[1],), num_classes)
        models['Neural Network'].fit(X, y, epochs=50, batch_size=32, verbose=0)
        models['Neural Network'].save('models/neural_network.keras')
    
    # Save feature names
    if feature_names is not None:
        joblib.dump(feature_names, 'models/feature_names.joblib')
    
    return models, scaler, feature_names

# Load or train models
@st.cache_resource
def get_models_and_scaler():
    return load_or_train_models()

models, scaler, feature_names = get_models_and_scaler()

# Create the input form
st.sidebar.header("Select Model")
selected_model = st.sidebar.selectbox(
    "Choose a model for prediction:",
    ["Random Forest", "XGBoost", "Neural Network"]
)

# Create columns for input fields
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Patient Information")
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    father_age = st.number_input("Father's Age", min_value=0, max_value=120, value=35)
    mother_age = st.number_input("Mother's Age", min_value=0, max_value=120, value=30)
    heart_rate = st.number_input("Heart Rate (rates/min", min_value=40, max_value=200, value=75)  # Note: no closing parenthesis as per training data
    respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=10, max_value=60, value=16)
    genes_mother = st.selectbox("Genes in mother's side", ["Yes", "No"])
    inherited_father = st.selectbox("Inherited from father", ["Yes", "No"])
    maternal_gene = st.selectbox("Maternal gene", ["Yes", "No"])
    paternal_gene = st.selectbox("Paternal gene", ["Yes", "No"])
    place_birth = st.selectbox("Place of birth", ["Hospital", "Home", "Other"])
    status = st.selectbox("Status", ["Active", "Inactive", "Under Observation"])

with col2:
    st.subheader("Medical History")
    blood_cell_count = st.number_input("Blood cell count (mcL)", min_value=0.0, max_value=10.0, value=5.0)
    white_blood_cell_count = st.number_input("White Blood cell count", min_value=0.0, max_value=20.0, value=7.0)
    radiation_exposure = st.selectbox("H/O radiation exposure (x-ray)", ["Yes", "No"])
    maternal_illness = st.selectbox("H/O serious maternal illness", ["Yes", "No"])
    substance_abuse = st.selectbox("H/O substance abuse", ["Yes", "No"])
    blood_test = st.selectbox("Blood test result", ["normal", "abnormal", "inconclusive"])
    birth_defects = st.selectbox("Birth defects", ["None", "Single", "Multiple"])
    birth_asphyxia = st.selectbox("Birth asphyxia", ["Yes", "No"])
    assisted_conception = st.selectbox("Assisted conception IVF/ART", ["Yes", "No"])
    parental_consent = st.selectbox("Parental consent", ["Yes", "No"])
    
    st.subheader("Test Results")
    test1 = st.number_input("Test 1", min_value=0.0, max_value=100.0, value=50.0)
    test2 = st.number_input("Test 2", min_value=0.0, max_value=100.0, value=50.0)
    test3 = st.number_input("Test 3", min_value=0.0, max_value=100.0, value=50.0)
    test4 = st.number_input("Test 4", min_value=0.0, max_value=100.0, value=50.0)
    test5 = st.number_input("Test 5", min_value=0.0, max_value=100.0, value=50.0)

with col3:
    st.subheader("Pregnancy History")
    previous_abortions = st.number_input("No. of previous abortion", min_value=0, max_value=10, value=0)
    history_anomalies = st.selectbox("History of anomalies in previous pregnancies", ["Yes", "No"])
    folic_acid = st.selectbox("Folic acid details (peri-conceptional)", ["Yes", "No"])
    follow_up = st.selectbox("Follow-up", ["Yes", "No"])
    autopsy_defect = st.selectbox("Autopsy shows birth defect", ["Yes", "No", "Not Applicable"])

    st.subheader("Institute Information")
    institute = st.text_input("Institute Name", value="General Hospital")
    location = st.text_input("Location of Institute", value="City Center")
    
    st.subheader("Symptoms")
    symptom1 = st.checkbox("Vision Problems", help="Issues with sight, including blurry vision or vision loss")
    symptom2 = st.checkbox("Muscle Weakness", help="Reduced strength in muscles")
    symptom3 = st.checkbox("Developmental Delay", help="Slower than usual development of physical or mental skills")
    symptom4 = st.checkbox("Breathing Difficulties", help="Problems with normal breathing patterns")
    symptom5 = st.checkbox("Neurological Issues", help="Seizures, coordination problems, or cognitive impairment")

# Create prediction button
if st.button("Predict Genetic Disorder"):
    try:
        # Prepare input data with all required features
        input_data = {
            'Patient Age': age,
            'Gender': gender,
            'Father\'s age': father_age,
            'Mother\'s age': mother_age,
            'Heart Rate (rates/min': heart_rate,  # Note: no closing parenthesis as per training data
            'Respiratory Rate (breaths/min)': respiratory_rate,
            'Genes in mother\'s side': genes_mother,
            'Inherited from father': inherited_father,
            'Maternal gene': maternal_gene,
            'Paternal gene': paternal_gene,
            'Blood cell count (mcL)': blood_cell_count,
            'White Blood cell count (thousand per microliter)': white_blood_cell_count,
            'Blood test result': blood_test,
            'Birth defects': birth_defects,
            'Birth asphyxia': birth_asphyxia,
            'Assisted conception IVF/ART': assisted_conception,
            'Folic acid details (peri-conceptional)': folic_acid,
            'Autopsy shows birth defect (if applicable)': autopsy_defect,
            'H/O radiation exposure (x-ray)': radiation_exposure,
            'H/O serious maternal illness': maternal_illness,
            'H/O substance abuse': substance_abuse,
            'Follow-up': follow_up,
            'No. of previous abortion': previous_abortions,
            'History of anomalies in previous pregnancies': history_anomalies,
            'Institute Name': institute,
            'Location of Institute': location,
            'Place of birth': place_birth,
            'Status': status,
            'Parental consent': parental_consent,
            'Test 1': test1,
            'Test 2': test2,
            'Test 3': test3,
            'Test 4': test4,
            'Test 5': test5,
            'Symptom 1': float(symptom1),  # Vision Problems
            'Symptom 2': float(symptom2),  # Muscle Weakness
            'Symptom 3': float(symptom3),  # Developmental Delay
            'Symptom 4': float(symptom4),  # Breathing Difficulties
            'Symptom 5': float(symptom5)   # Neurological Issues
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # For new models, we need to rebuild the training data and process again to get correct feature order
        if feature_names is None:
            st.warning("Feature order information not found. Rebuilding from training data...")
            train_df = pd.read_csv("train.csv")
            processed_train = preprocess_data(train_df)
            X_train, _, _ = prepare_features_and_target(processed_train)
            feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
            # Save for future use
            if feature_names is not None:
                if not os.path.exists('models'):
                    os.makedirs('models')
                joblib.dump(feature_names, 'models/feature_names.joblib')
        
        # Add any missing columns from training data with default values
        if feature_names is not None:
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Use appropriate default value
            
            # Ensure columns are in the same order as during training
            input_df = input_df[feature_names]
        
        # Preprocess input
        processed_input = preprocess_data(input_df)
        
        # Get features in same way as training
        X_input, _, _ = prepare_features_and_target(processed_input, target_column=None)
        
        # Scale input
        X_input_scaled = scaler.transform(X_input)
        
        # Make prediction
        model = models[selected_model]
        if selected_model == "Neural Network":
            prediction = np.argmax(model.predict(X_input_scaled), axis=1)[0]
        else:
            prediction = model.predict(X_input_scaled)[0]
        
        # Display prediction
        st.success("### Prediction Results")
        
        # Load training data to get disorder categories
        train_df = pd.read_csv("train.csv")
        disorders = train_df['Genetic Disorder'].unique()
        disorders = [d for d in disorders if isinstance(d, str)]  # Remove any NaN values
        
        if prediction < len(disorders):
            predicted_disorder = disorders[prediction]
            st.write(f"The predicted genetic disorder category is: **{predicted_disorder}**")
            
            # Display confidence scores if available
            if selected_model == "Neural Network":
                probabilities = model.predict(X_input_scaled)[0]
                st.write("\nConfidence scores:")
                for disorder, prob in zip(disorders, probabilities):
                    if isinstance(disorder, str):  # Skip if disorder is NaN
                        st.write(f"{disorder}: {prob:.2%}")
            elif hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_input_scaled)[0]
                st.write("\nConfidence scores:")
                for disorder, prob in zip(disorders, probabilities):
                    if isinstance(disorder, str):  # Skip if disorder is NaN
                        st.write(f"{disorder}: {prob:.2%}")
        else:
            st.error("Unable to determine the specific disorder category. Please consult with a medical professional.")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please ensure all required fields are filled correctly.") 