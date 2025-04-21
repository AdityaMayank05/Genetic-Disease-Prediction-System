import streamlit as st
import pandas as pd
import numpy as np
from app import preprocess_data
import pickle
import os
import sys
from tensorflow import keras

# --- TEMPORARY: Clear cache on startup ---
st.cache_resource.clear()
print("Streamlit cache cleared.")
# --- END TEMPORARY ---

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(sys.argv[0] if __name__ == "__main__" else __file__))
# Define alternative models directory (sibling to script's dir)
alt_output_dir_name = 'model_files_alt'
models_dir = os.path.join(os.path.dirname(script_dir), alt_output_dir_name) 

# --- Define paths for saved components ---
SCALER_PATH = os.path.join(models_dir, 'fitted_scaler.pkl')
TARGET_ENCODER_PATH = os.path.join(models_dir, 'target_label_encoder.pkl')
RF_MODEL_PATH = os.path.join(models_dir, 'random_forest_model.pkl')
XGB_MODEL_PATH = os.path.join(models_dir, 'xgboost_model.pkl')
# NN_MODEL_PATH = os.path.join(models_dir, 'neural_network_model.keras') # Removed NN
FEATURE_NAMES_PATH = os.path.join(models_dir, 'feature_names.pkl')
SUBCLASS_MAP_PATH = os.path.join(models_dir, 'disorder_to_subclass.pkl')

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

def load_prediction_components():
    """Load all necessary components for prediction."""
    print(f"Attempting to load components from: {models_dir}") # Debug print
    if not os.path.exists(models_dir):
        st.error(f"Models directory not found at {models_dir}. Please run the training script (app.py) first.")
        st.stop()

    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(TARGET_ENCODER_PATH, 'rb') as f:
            target_encoder = pickle.load(f)
        with open(RF_MODEL_PATH, 'rb') as f:
            rf_model = pickle.load(f)
        with open(XGB_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        # nn_model = keras.models.load_model(NN_MODEL_PATH) # Removed NN

        # Load feature names (important for column order)
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            # Attempt to get feature names from the scaler if available
            if hasattr(scaler, 'feature_names_in_'):
                feature_names = scaler.feature_names_in_
            else:
                st.warning("Feature names file not found. Column order might be incorrect.")
                feature_names = None

        # Load subclass mapping
        print(f"Checking for subclass map at: {SUBCLASS_MAP_PATH}") # DEBUG
        subclass_exists = os.path.exists(SUBCLASS_MAP_PATH) # DEBUG
        print(f"Subclass map exists? {subclass_exists}") # DEBUG
        
        # --- Add more detailed check ---
        disorder_to_subclass = {}
        if subclass_exists:
            try:
                print(f"Attempting to open: {SUBCLASS_MAP_PATH}")
                with open(SUBCLASS_MAP_PATH, 'rb') as f:
                     # Try reading a byte to check readability
                     # test_read = f.read(1) 
                     # print(f"Successfully read a byte from subclass map.")
                     # f.seek(0) # Reset file pointer
                     disorder_to_subclass = pickle.load(f)
                print("Subclass map loaded successfully.")
            except PermissionError as pe:
                 print(f"[ERROR] Permission denied when trying to read {SUBCLASS_MAP_PATH}: {pe}")
                 st.warning(f"Permission denied for subclass map file. Displaying warning.")
            except Exception as e:
                 print(f"[ERROR] Failed to load subclass map from {SUBCLASS_MAP_PATH}: {e}")
                 st.warning(f"Error loading subclass map file. Displaying warning.")
        else:
            print("Subclass map file reported as not existing by os.path.exists.")
            st.warning("Subclass mapping file not found.")
        # --- End detailed check ---

    except FileNotFoundError as e:
        st.error(f"Error loading required file: {e}. Did you run the training script (app.py) successfully?")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during loading: {e}")
        st.stop()

    models_dict = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        # "Neural Network": nn_model # Removed NN from dict
    }

    return models_dict, scaler, target_encoder, feature_names, disorder_to_subclass

# Load components using caching
@st.cache_resource
def get_prediction_components():
    print("Loading prediction components...")
    return load_prediction_components()

models, scaler, target_encoder, feature_names, disorder_to_subclass = get_prediction_components()

# Create the input form
st.sidebar.header("Select Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a model for prediction:",
    list(models.keys())
)

# Create columns for input fields
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Patient Information")
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Ambiguous"])
    father_age = st.number_input("Father's Age", min_value=0, max_value=120, value=35)
    mother_age = st.number_input("Mother's Age", min_value=0, max_value=120, value=30)
    heart_rate = st.selectbox("Heart Rate", ["Normal", "Tachycardia"])
    respiratory_rate = st.selectbox("Respiratory Rate", ["Normal", "Tachypnea"])
    genes_mother = st.selectbox("Genes in mother's side", ["Yes", "No"])
    inherited_father = st.selectbox("Inherited from father", ["Yes", "No"])
    maternal_gene = st.selectbox("Maternal gene", ["Yes", "No"])
    paternal_gene = st.selectbox("Paternal gene", ["Yes", "No"])
    place_birth = st.selectbox("Place of birth", ["Hospital", "Home", "Other"])
    status = st.selectbox("Status", ["Alive", "Deceased"])

with col2:
    st.subheader("Medical History")
    blood_cell_count = st.number_input("Blood cell count (mcL)", min_value=0.0, max_value=10.0, value=5.0)
    white_blood_cell_count = st.number_input("White Blood cell count", min_value=0.0, max_value=20.0, value=7.0)
    radiation_exposure = st.selectbox("H/O radiation exposure (x-ray)", ["Yes", "No"])
    maternal_illness = st.selectbox("H/O serious maternal illness", ["Yes", "No"])
    substance_abuse = st.selectbox("H/O substance abuse", ["Yes", "No"])
    blood_test = st.selectbox("Blood test result", ["normal", "abnormal", "inconclusive"])
    birth_defects = st.selectbox("Autopsy shows birth defect", ["No", "Yes", "Not Applicable"])
    birth_defect_type = st.selectbox("Birth defect", ["Multiple", "Singular"])
    birth_asphyxia = st.selectbox("Birth asphyxia", ["Yes", "No"])
    assisted_conception = st.selectbox("Assisted conception IVF/ART", ["Yes", "No"])
    parental_consent = st.selectbox("Parental consent", ["Yes", "No"])
    
    st.subheader("Test Results")
    test1 = st.checkbox("Test 1")
    test2 = st.checkbox("Test 2")
    test3 = st.checkbox("Test 3")
    test4 = st.checkbox("Test 4")
    test5 = st.checkbox("Test 5")

with col3:
    st.subheader("Pregnancy History")
    previous_abortions = st.number_input("No. of previous abortion", min_value=0, max_value=10, value=0)
    history_anomalies = st.selectbox("History of anomalies in previous pregnancies", ["Yes", "No"])
    folic_acid = st.selectbox("Folic acid details (peri-conceptional)", ["Yes", "No"])
    follow_up = st.selectbox("Follow-up", ["Low", "High"])

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
    if not institute or not location:
        st.warning("Please fill in Institute Name and Location.")
    else:
        try:
            input_data = {
                'Patient Age': age,
                'Gender': gender,
                'Father\'s age': father_age,
                'Mother\'s age': mother_age,
                'Heart Rate (rates/min)': heart_rate,
                'Respiratory Rate (breaths/min)': respiratory_rate,
                'Genes in mother\'s side': genes_mother,
                'Inherited from father': inherited_father,
                'Maternal gene': maternal_gene,
                'Paternal gene': paternal_gene,
                'Blood cell count (mcL)': blood_cell_count,
                'White Blood cell count (thousand per microliter)': white_blood_cell_count,
                'Blood test result': blood_test,
                'Autopsy shows birth defect (if applicable)': birth_defects,
                'Birth defects': birth_defect_type,
                'Birth asphyxia': birth_asphyxia,
                'Assisted conception IVF/ART': assisted_conception,
                'Folic acid details (peri-conceptional)': folic_acid,
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
                'Test 1': float(test1),
                'Test 2': float(test2),
                'Test 3': float(test3),
                'Test 4': float(test4),
                'Test 5': float(test5),
                'Symptom 1': float(symptom1),
                'Symptom 2': float(symptom2),
                'Symptom 3': float(symptom3),
                'Symptom 4': float(symptom4),
                'Symptom 5': float(symptom5)
            }

            input_df = pd.DataFrame([input_data])
            st.write("Raw Input DataFrame:", input_df)

            processed_input_df = preprocess_data(input_df.copy())
            st.write("Processed Input DataFrame (before scaling/reordering):", processed_input_df)

            if feature_names is not None:
                missing_cols = set(feature_names) - set(processed_input_df.columns)
                for c in missing_cols:
                    processed_input_df[c] = 0
                processed_input_df = processed_input_df[feature_names]
                st.write("Processed Input DataFrame (After Reordering/Adding Missing):", processed_input_df)
            else:
                st.warning("Feature names not loaded. Scaling might fail or be incorrect if column order differs from training.")

            X_input_scaled = scaler.transform(processed_input_df)
            st.write("Scaled Input Data (Shape):", X_input_scaled.shape)

            model = models[selected_model_name]

            # if selected_model_name == "Neural Network": # Removed NN prediction block
            #     prediction_proba = model.predict(X_input_scaled)
            #     prediction_encoded = np.argmax(prediction_proba, axis=1)[0]
            #     probabilities = prediction_proba[0]
            # else: 
            # Use standard sklearn predict/predict_proba
            prediction_encoded = model.predict(X_input_scaled)[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_input_scaled)[0]

            st.write(f"Raw prediction value (encoded): {prediction_encoded}")

            try:
                predicted_disorder = target_encoder.inverse_transform([prediction_encoded])[0]
            except ValueError as ve:
                st.error(f"Error decoding prediction: {ve}. The model might be predicting a class index not seen during encoder fitting.")
                st.write("Available classes in encoder:", target_encoder.classes_)
                predicted_disorder = f"Unknown (Index: {prediction_encoded})"
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction decoding: {e}")
                predicted_disorder = f"Error decoding (Index: {prediction_encoded})"

            st.success("### Prediction Results")
            st.write(f"Selected Model: **{selected_model_name}**")
            st.write(f"The predicted genetic disorder category is: **{predicted_disorder}**")

            if predicted_disorder in disorder_to_subclass:
                subclass = disorder_to_subclass[predicted_disorder]
                st.write(f"Disorder subclass: **{subclass}**")
            else:
                st.write("Disorder subclass: *Not available for this disorder*")

            if probabilities is not None:
                st.write("\nConfidence scores:")
                try:
                    class_names = target_encoder.classes_
                    prob_df = pd.DataFrame({'Disorder': class_names, 'Confidence': probabilities})
                    prob_df = prob_df.sort_values(by='Confidence', ascending=False)
                    st.dataframe(prob_df.style.format({'Confidence': '{:.2%}'}))
                except Exception as e:
                    st.error(f"Error displaying probability scores: {str(e)}")
            else:
                st.write("Confidence scores not available for this model.")

        except FileNotFoundError as e:
            st.error(f"Prediction failed. A required file is missing: {e}. Please ensure training was successful.")
        except ValueError as e:
            st.error(f"An error occurred during data preparation or prediction: {e}")
            st.error("This might be due to unexpected input values or inconsistencies between training and prediction data.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {str(e)}")
            st.exception(e)
            st.error("Please ensure all required fields are filled correctly and consistently with training data.")

st.sidebar.markdown("---")
st.sidebar.subheader("Loaded Components")
if scaler:
    st.sidebar.write(f"Scaler: Loaded ({SCALER_PATH.split(os.sep)[-1]})")
if target_encoder:
    st.sidebar.write(f"Target Encoder: Loaded ({TARGET_ENCODER_PATH.split(os.sep)[-1]})")
    st.sidebar.write(f" - Classes: {len(target_encoder.classes_)}")
if models:
    st.sidebar.write(f"Models Loaded: {', '.join(models.keys())}")
if feature_names:
    st.sidebar.write(f"Feature Names: Loaded ({len(feature_names)} features)")
else:
    st.sidebar.warning("Feature Names: Not Loaded")
if disorder_to_subclass:
    st.sidebar.write("Subclass Map: Loaded")
