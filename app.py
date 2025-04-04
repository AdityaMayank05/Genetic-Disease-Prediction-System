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

def load_data(train_path="train.csv", test_path="test.csv"):
    """
    Load and prepare the genetic disorder dataset
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical variables,
    and scaling numerical features
    """
    # Create copy to avoid modifying original dataframe
    df = df.copy()
    
    # Identify column types
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Fill numerical missing values with mean
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert binary columns to numeric
    binary_columns = ['Genes in mother\'s side', 'Inherited from father', 
                     'Maternal gene', 'Paternal gene']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Encode all remaining categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns and col not in binary_columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Drop non-relevant columns
    columns_to_drop = ['Patient Id', 'Patient First Name', 'Family Name', 
                      'Father\'s name', 'Mother\'s name', 'Sister\'s name', 
                      'Brother\'s name', 'Hospital', 'Doctor', 'Lab']
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    return df

def prepare_features_and_target(df, target_column='Genetic Disorder'):
    """
    Separate features and target, scale features
    """
    # Separate features and target
    X = df.drop([target_column, 'Disorder Subclass'], axis=1, errors='ignore')
    y = df[target_column] if target_column in df.columns else None
    
    # Ensure all data is numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier
    """
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def create_neural_network(input_shape, num_classes):
    """
    Create a neural network model
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance
    """
    if isinstance(model, keras.Model):
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)
    
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Load data
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    train_df_processed = preprocess_data(train_df)
    test_df_processed = preprocess_data(test_df)
    
    # Prepare features and target
    X_scaled, y, scaler = prepare_features_and_target(train_df_processed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # Train Neural Network
    print("Training Neural Network...")
    num_classes = len(np.unique(y))
    nn_model = create_neural_network((X_train.shape[1],), num_classes)
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    evaluate_model(nn_model, X_test, y_test, "Neural Network")

if __name__ == "__main__":
    main()
