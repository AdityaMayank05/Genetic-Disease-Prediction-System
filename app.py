import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt
import pickle # Add pickle import
import os # Add os import for path joining
import sys # Add sys import for script path

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(sys.argv[0] if __name__ == "__main__" else __file__))
# Save to an alternative directory directly under the workspace root
# alt_output_dir_name = 'model_files_alt' 
# models_dir = os.path.join(os.path.dirname(script_dir), alt_output_dir_name) # Go up one level from script dir

def load_data(train_path="train.csv", test_path="test.csv"):
    """
    Load and prepare the genetic disorder dataset
    Looks for files relative to the script's directory.
    """
    # Construct full paths based on script directory
    full_train_path = os.path.join(script_dir, train_path)
    full_test_path = os.path.join(script_dir, test_path)
    
    try:
        train_df = pd.read_csv(full_train_path)
    except FileNotFoundError:
        print(f"Error: Training file not found at {full_train_path}")
        print(f"Please ensure '{train_path}' is in the same directory as the script ('{script_dir}').")
        sys.exit(1) # Exit if train file is missing
        
    try:
        test_df = pd.read_csv(full_test_path)
    except FileNotFoundError:
        print(f"Warning: Test file not found at {full_test_path}. Proceeding without it.")
        test_df = None # Allow proceeding without test data if needed
        
    return train_df, test_df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical variables.
    Uses column names consistent with the original train.csv file.
    Note: Scaling is now handled in prepare_features_and_target.
    """
    df = df.copy()

    # Define column types explicitly using original train.csv names
    binary_yes_no_columns = [
        'Genes in mother\'s side', 'Inherited from father', 'Maternal gene', 'Paternal gene',
        'H/O radiation exposure (x-ray)', 'H/O serious maternal illness',
        'H/O substance abuse', 'Birth asphyxia', 'Assisted conception IVF/ART',
        'Folic acid details (peri-conceptional)', 
        'History of anomalies in previous pregnancies', 'Parental consent'
        # Note: 'Follow-up' from train.csv was Yes/No, but UI changed to Low/High.
        # It will be handled in other_categorical_columns now.
        # Note: 'Autopsy shows birth defect (if applicable)' from train.csv was Yes/No/Not applicable
        # It will be handled in other_categorical_columns now.
    ]
    
    other_categorical_columns = [
        'Gender', 'Blood test result', 
        'Birth defects', # Original name for Singular/Multiple type
        'Follow-up', # Original name, but with new Low/High values
        'Autopsy shows birth defect (if applicable)', # Original name for Yes/No/Not applicable
        'Institute Name', 'Location of Institute', 'Place of birth', 'Status',
        'Heart Rate (rates/min)', # Original name, now categorical
        'Respiratory Rate (breaths/min)' # Original name, now categorical
    ]

    numerical_columns = [
        'Patient Age', 'Father\'s age', 'Mother\'s age', 
        'Blood cell count (mcL)', 'White Blood cell count (thousand per microliter)',
        'No. of previous abortion',
        'Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5', # Floats 0.0/1.0
        'Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4', 'Symptom 5' # Floats 0.0/1.0
    ]
    
    # Rename columns coming from Streamlit UI to match train.csv expected names
    # This is crucial because the input_df from streamlit might have UI-based names
    rename_map = {
        'Heart Rate': 'Heart Rate (rates/min)',
        'Respiratory Rate': 'Respiratory Rate (breaths/min)',
        # The following depends on how streamlit_app.py names the keys:
        # If streamlit uses 'Autopsy shows birth defect' for the Yes/No/NA field:
        'Autopsy shows birth defect': 'Autopsy shows birth defect (if applicable)', 
        # If streamlit uses 'Birth defect' for the Multiple/Singular field:
        'Birth defect': 'Birth defects'
    }
    df.rename(columns=rename_map, inplace=True, errors='ignore')

    # Ensure all expected columns (using original names) exist
    all_expected_features = binary_yes_no_columns + other_categorical_columns + numerical_columns
    for col in all_expected_features:
        if col not in df.columns:
            # Add missing columns, initialize based on type guess or default
            if col in numerical_columns:
                 df[col] = 0.0
            else:
                 df[col] = 'Unknown' # Default for categorical
            # df[col] = np.nan # Alternative: add as NaN then fill later

    # Fill numerical missing values with mean
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean() if not df[col].isnull().all() else 0)

    # Fill categorical missing values with mode
    all_categorical = binary_yes_no_columns + other_categorical_columns
    for col in all_categorical:
        if col in df.columns:
            df[col] = df[col].astype(str)
            mode_val = df[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else 'Unknown'
            df[col] = df[col].replace(['nan', 'NaN', 'None'], np.nan) # More robust NaN replacement
            df[col] = df[col].fillna(fill_value)

    # Map Yes/No binary columns (using original names list)
    for col in binary_yes_no_columns:
        if col in df.columns:
             # Ensure mapping happens correctly even after filling NaNs with 'Unknown'
            if df[col].dtype == 'object':
                 df[col] = df[col].str.lower().map({'yes': 1, 'no': 0}).fillna(0) 
            else: # If already numeric (e.g. came in as 0/1), ensure it's int
                 df[col] = df[col].fillna(0).astype(int)

    # Encode all other categorical variables (using original names list)
    le = LabelEncoder()
    for col in other_categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            
    # Drop non-relevant administrative columns 
    columns_to_drop_admin = [
        'Patient Id', 'Patient First Name', 'Family Name', 'Father\'s name', 
        'Mother\'s name', 'Sister\'s name', 'Brother\'s name', 
        'Hospital', 'Doctor', 'Lab'
    ]
    df = df.drop([col for col in columns_to_drop_admin if col in df.columns], axis=1, errors='ignore')
    
    # Ensure all feature columns are numeric
    feature_cols = [col for col in df.columns if col not in ['Genetic Disorder', 'Disorder Subclass']]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Ensure integer type where appropriate (e.g., binary, encoded categorical)
        if df[col].apply(lambda x: x.is_integer()).all():
             df[col] = df[col].astype(int)

    return df

def prepare_features_and_target(df, target_column='Genetic Disorder', scaler=None):
    """
    Separate features and target, scale features.
    If scaler is provided, use it to transform. 
    If target_column is provided and scaler is None, fit a new scaler.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    is_training = target_column is not None and target_column in df.columns
    
    # For prediction mode, no target column is needed
    if not is_training:
        # Ensure all the columns needed for prediction are available
        # Dropping these columns if they exist as they're not features
        columns_to_drop = ['Genetic Disorder', 'Disorder Subclass']
        X = df.drop(columns_to_drop, axis=1, errors='ignore')
        y = None
        if scaler is None:
             raise ValueError("Scaler must be provided for prediction mode.")
    else:
        # Separate features and target for training
        X = df.drop([target_column, 'Disorder Subclass'], axis=1, errors='ignore')
        y = df[target_column] if target_column in df.columns else None
    
    # Ensure all data is numeric before scaling
    for col in X.columns:
        if X[col].dtype == 'object':
            # This encoding here might still be problematic if called during prediction
            # Ideally, encoding happens consistently in preprocess_data
            le = LabelEncoder() 
            X[col] = le.fit_transform(X[col].astype(str)) 
            
    # Handle the scaler
    if is_training:
        if scaler is None:
            print("Fitting new scaler...")
            scaler = StandardScaler()
            X_scaled_array = scaler.fit_transform(X)
        else:
            print("Using provided scaler for training data transformation...")
            X_scaled_array = scaler.transform(X)
    else: # Prediction mode
        print("Using provided scaler for prediction data transformation...")
        X_scaled_array = scaler.transform(X)

    # Create DataFrame with original column names
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

    # Return the scaler only if it was fitted here (during training)
    if is_training:
        return X_scaled, y, scaler
    else:
        return X_scaled, y # Scaler is not returned as it was provided

def train_random_forest(X_train, y_train, random_state=42, n_estimators=100, max_depth=None):
    """
    Train a Random Forest classifier
    """
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=max_depth
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train, random_state=42, n_estimators=100, max_depth=3):
    """
    Train an XGBoost classifier
    """
    xgb_model = XGBClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=max_depth
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def create_neural_network(input_shape, num_classes):
    """
    Create a neural network model
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def evaluate_model(model, X_test, y_test, model_name="Model", label_encoder=None):
    """
    Evaluate model performance
    If label_encoder is provided, use its classes for reporting.
    """
    if isinstance(model, keras.Model):
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)
    
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    target_names = label_encoder.classes_ if label_encoder is not None else None
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=target_names if target_names is not None else "auto", 
                yticklabels=target_names if target_names is not None else "auto")
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load data
    print("Loading data...")
    train_df, test_df = load_data()
    
    # --- Handle potential NaN in target variable ---
    target_col = 'Genetic Disorder'
    initial_rows = len(train_df)
    train_df.dropna(subset=[target_col], inplace=True)
    rows_dropped = initial_rows - len(train_df)
    if rows_dropped > 0:
        print(f"[INFO] Dropped {rows_dropped} rows with missing target ('{target_col}') values.")
    # --- End NaN handling for target ---

    # --- Create Disorder -> Subclass Mapping --- 
    print("Creating disorder to subclass mapping...")
    disorder_to_subclass = {}
    if 'Disorder Subclass' in train_df.columns:
        # Ensure we use the train_df *after* dropping NaNs in the target column
        mapping_df = train_df.dropna(subset=['Disorder Subclass'])
        for _, row in mapping_df.iterrows():
            disorder = str(row[target_col])
            subclass = str(row['Disorder Subclass'])
            if disorder not in disorder_to_subclass: # Store the first subclass found for a disorder
                disorder_to_subclass[disorder] = subclass
    print(f"Created mapping for {len(disorder_to_subclass)} disorders.")
    # --- End Mapping Creation ---
    
    # Preprocess data
    print("Preprocessing data...")
    # It's crucial that preprocessing is consistent. 
    # Consider fitting encoders here and passing them if needed.
    train_df_processed = preprocess_data(train_df)
    test_df_processed = preprocess_data(test_df) # Use same preprocessing
    
    # Prepare features and target for the training set (fit scaler)
    # Ensure test_df_processed has the target column temporarily for consistent processing IF needed
    # For now, assuming prepare_features_and_target can handle df without target if needed
    print("Preparing features and target for training data...")
    X_scaled_train_full, y_train_full, fitted_scaler = prepare_features_and_target(
        train_df_processed, target_column='Genetic Disorder', scaler=None
    )
    
    # Prepare features for the test set (use fitted scaler)
    # We need X from test_df_processed, scaled with fitted_scaler
    # Temporarily add dummy target to test_df if needed by prepare_features_and_target structure
    # or better, modify prepare_features_and_target to handle df without target gracefully for scaling
    
    # Let's prepare test features separately using the fitted scaler
    print("Preparing features for test data...")
    # Extract features from test_df_processed, dropping potential target/subclass if present
    X_test_unscaled = test_df_processed.drop(['Genetic Disorder', 'Disorder Subclass'], axis=1, errors='ignore')
    
    # Ensure columns match training data *before* scaling
    # Get columns from the scaled training data (order matters for transform)
    train_cols = X_scaled_train_full.columns
    X_test_unscaled = X_test_unscaled.reindex(columns=train_cols, fill_value=0) # Align columns, fill missing with 0 (check if appropriate)

    # Ensure numeric types (might be redundant if preprocess handles it)
    for col in X_test_unscaled.columns:
         X_test_unscaled[col] = pd.to_numeric(X_test_unscaled[col], errors='coerce').fillna(0)
         if X_test_unscaled[col].apply(lambda x: x.is_integer()).all():
              X_test_unscaled[col] = X_test_unscaled[col].astype(int)

    # --- Fit LabelEncoder on the FULL training target BEFORE split ---
    print("Fitting target label encoder on full training data...")
    le_target = LabelEncoder()
    # Ensure y_train_full is string type before fitting if necessary
    y_train_full_str = y_train_full.astype(str) 
    le_target.fit(y_train_full_str) 
    print(f"Target encoder classes: {le_target.classes_}")
    # --- End fitting encoder ---

    # Split the *processed and scaled* training data
    print("Splitting training data for validation...")
    X_train, X_val, y_train_str, y_val_str = train_test_split(
        X_scaled_train_full, y_train_full_str, test_size=0.2, random_state=42 # Use string targets for split
    )
    
    # --- Transform y_train and y_val using the already fitted encoder ---
    y_train = le_target.transform(y_train_str)
    y_val = le_target.transform(y_val_str)
    # --- End transforming split targets ---

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train) # Use encoded y_train
    # Evaluate requires encoded labels too
    evaluate_model(rf_model, X_val, y_val, "Random Forest", le_target)
    
    # Train XGBoost
    print("Training XGBoost...")
    # No longer need to encode here, as y_train is already encoded
    # le_target = LabelEncoder()
    # y_train_encoded = le_target.fit_transform(y_train)
    # y_val_encoded = le_target.transform(y_val) # Use the same encoder
    
    xgb_model = train_xgboost(X_train, y_train) # Use encoded y_train 
    # Evaluate XGBoost - requires encoded labels
    print("\nEvaluating XGBoost...")
    y_pred_xgb_encoded = xgb_model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred_xgb_encoded))
    print("\nClassification Report:")
    # Use target names from the encoder for report labels
    print(classification_report(y_val, y_pred_xgb_encoded, target_names=le_target.classes_)) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_val, y_pred_xgb_encoded), annot=True, fmt='d', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.title('XGBoost Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Train Neural Network
    print("Training Neural Network...")
    # NN needs numerically encoded targets
    num_classes = len(le_target.classes_)
    nn_model = create_neural_network((X_train.shape[1],), num_classes)
    # Use encoded labels for training and validation
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val)) 
    # Evaluate NN - requires encoded labels for comparison
    print("\nEvaluating Neural Network...")
    y_pred_nn_proba = nn_model.predict(X_val)
    y_pred_nn_encoded = np.argmax(y_pred_nn_proba, axis=1)
    print("Accuracy:", accuracy_score(y_val, y_pred_nn_encoded))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_nn_encoded, target_names=le_target.classes_))
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_val, y_pred_nn_encoded), annot=True, fmt='d', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.title('Neural Network Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # --- Save the necessary components --- 
    print("Saving scaler, encoders, and models...")
    # Define output directory relative to script's PARENT directory
    alt_output_dir_name = 'model_files_alt'
    output_dir = os.path.join(os.path.dirname(script_dir), alt_output_dir_name) 
    print(f"Attempting to save components to: {output_dir}") # Debug print
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to create directory {output_dir}: {e}")
            # Decide how to handle this - maybe exit or default to local?
            print("Defaulting to saving in script directory.")
            output_dir = script_dir # Fallback to script directory
            if not os.path.exists(output_dir):
                 os.makedirs(output_dir) # Should exist but double-check
                 
    # Use .pkl extension for pickle files
    scaler_path = os.path.join(output_dir, 'fitted_scaler.pkl')
    encoder_path = os.path.join(output_dir, 'target_label_encoder.pkl')
    rf_path = os.path.join(output_dir, 'random_forest_model.pkl')
    xgb_path = os.path.join(output_dir, 'xgboost_model.pkl')
    nn_path = os.path.join(output_dir, 'neural_network_model.keras') 
    feature_names_path = os.path.join(output_dir, 'feature_names.pkl') 
    # Add path for subclass map
    subclass_map_path = os.path.join(output_dir, 'disorder_to_subclass.pkl') 
    
    save_success = True # Flag to track success

    # Save scaler using pickle
    try:
        with open(scaler_path, 'wb') as f:
             pickle.dump(fitted_scaler, f)
        if not os.path.exists(scaler_path):
            print(f"[ERROR] Failed to save scaler to {scaler_path} - File not found after pickle dump.")
            save_success = False
        else:
            print(f"[INFO] Scaler saved to {scaler_path}")
    except Exception as e:
        print(f"[ERROR] Exception saving scaler: {e}")
        save_success = False
        
    # Save target encoder using pickle
    try:
        with open(encoder_path, 'wb') as f:
            pickle.dump(le_target, f)
        if not os.path.exists(encoder_path):
            print(f"[ERROR] Failed to save target encoder to {encoder_path} - File not found after pickle dump.")
            save_success = False
        else:
            print(f"[INFO] Target encoder saved to {encoder_path}")
    except Exception as e:
        print(f"[ERROR] Exception saving target encoder: {e}")
        save_success = False

    # Save the feature names using pickle
    try:
        feature_names = list(X_scaled_train_full.columns) 
        with open(feature_names_path, 'wb') as f:
            pickle.dump(feature_names, f)
        if not os.path.exists(feature_names_path):
            print(f"[ERROR] Failed to save feature names to {feature_names_path} - File not found after pickle dump.")
            save_success = False
        else:
            print(f"[INFO] Feature names saved to {feature_names_path}")
    except Exception as e:
        print(f"[ERROR] Exception saving feature names: {e}")
        save_success = False
        
    # Save Random Forest model using pickle
    try:
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_model, f)
        if not os.path.exists(rf_path):
            print(f"[ERROR] Failed to save Random Forest model to {rf_path} - File not found after pickle dump.")
            save_success = False
        else:
            print(f"[INFO] Random Forest model saved to {rf_path}")
    except Exception as e:
        print(f"[ERROR] Exception saving Random Forest model: {e}")
        save_success = False

    # Save XGBoost model using pickle
    try:
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        if not os.path.exists(xgb_path):
            print(f"[ERROR] Failed to save XGBoost model to {xgb_path} - File not found after pickle dump.")
            save_success = False
        else:
            print(f"[INFO] XGBoost model saved to {xgb_path}")
    except Exception as e:
        print(f"[ERROR] Exception saving XGBoost model: {e}")
        save_success = False

    # Save Disorder -> Subclass Map using pickle
    try:
        with open(subclass_map_path, 'wb') as f:
             pickle.dump(disorder_to_subclass, f)
        if not os.path.exists(subclass_map_path):
            print(f"[ERROR] Failed to save subclass map to {subclass_map_path} - File not found after pickle dump.")
            save_success = False
        else:
            print(f"[INFO] Subclass map saved to {subclass_map_path}")
    except Exception as e:
        print(f"[ERROR] Exception saving subclass map: {e}")
        save_success = False

    # Save Neural Network model (Keras format)
    try:
        nn_model.save(nn_path) # Use recommended Keras format
        if not os.path.exists(nn_path):
             print(f"[ERROR] Failed to save Neural Network model to {nn_path} - File not found after save.")
             save_success = False
        else:
            print(f"[INFO] Neural Network model saved to {nn_path}")
    except Exception as e:
        print(f"[ERROR] Exception saving Neural Network model: {e}")
        save_success = False

    if save_success:
        print(f"All components saved successfully to '{output_dir}'.")
    else:
        print(f"[WARNING] One or more components failed to save correctly to '{output_dir}'.")

if __name__ == "__main__":
    main()
