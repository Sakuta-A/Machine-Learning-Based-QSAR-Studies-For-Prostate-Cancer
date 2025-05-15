import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFECV
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from xgboost import XGBRegressor
import logging
from scipy.stats import randint
import joblib
import optuna
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

activity_property_name = "Ki (nM)"  # Correct property name from SDF file

# Load SDF files
folder_path = "/Users/sakuta/Desktop/Desktop/Major_Project/Project/Molecules"
sdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".sdf")]

sdf_supplier = Chem.SDMolSupplier(sdf_files[0])
for mol in sdf_supplier:
    if mol is not None:
        print("Available properties in SDF:", list(mol.GetPropNames()))
        break

valid_mols = []
for sdf_file in sdf_files:
    sdf_supplier = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
    valid_mols.extend([mol for mol in sdf_supplier if mol is not None])

print(f"Total valid molecules: {len(valid_mols)}")

mols = []
activities = []

for sdf_file in tqdm(sdf_files, desc="Processing SDF Files"):
    sdf_supplier = Chem.SDMolSupplier(sdf_file)
    for mol in tqdm(sdf_supplier, desc=f"Processing molecules in {os.path.basename(sdf_file)}", leave=False):
        if mol is None:
            print(f"Warning: Skipping an invalid molecule in {sdf_file}.")
            continue
        if mol.HasProp(activity_property_name):
            try:
                activity = float(mol.GetProp(activity_property_name))  # Convert to float
                if activity <= 9000:  # Filter molecules with Ki ≤ 9000 nM
                    activities.append(activity)
                    mols.append(mol)
            except ValueError:
                print(f"Warning: Skipping molecule with invalid activity value in {sdf_file}.")

# Validate extracted data
if not mols:
    raise ValueError("No valid molecules found. Check the SDF file and property names.")

# Generate binary Morgan fingerprints
morgan_gen = GetMorganGenerator(radius=2, fpSize=4096)
fingerprints = [morgan_gen.GetFingerprint(m) for m in mols]

# Convert fingerprints to NumPy arrays
def fingerprints_to_numpy(fps):
    array = np.zeros((len(fps), 4096))
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, array[i])
    return array

X = fingerprints_to_numpy(fingerprints)
y = np.log1p(np.array(activities))  # Apply log transformation

# Add Molecular Descriptors to Features
descriptors = [Descriptors.MolWt, Descriptors.TPSA, Descriptors.MolLogP, Descriptors.NumHDonors, Descriptors.NumHAcceptors]

def get_descriptors(mol):
    return [desc(mol) for desc in descriptors]

extra_features = np.array([get_descriptors(mol) for mol in mols])
X = np.hstack((X, extra_features))  # Append descriptors to fingerprint data

# Set random seed for reproducibility
seed = 42

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Saving & Loading
model_path = "/Users/sakuta/Desktop/Desktop/Major_Project/Project/trained_model.pkl"

if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = joblib.load(model_path)
    retrain = False
else:
    retrain = True

# Initial Feature Reduction using SelectFromModel
rf_initial = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_initial.fit(X_train, y_train)

selector_initial = SelectFromModel(rf_initial, threshold="median", prefit=True)
X_train_reduced = selector_initial.transform(X_train)
X_test_reduced = selector_initial.transform(X_test)

# Print initial feature counts before variance thresholding
print(f"Initial feature count: {X_train.shape[1]}, Reduced feature count: {X_train_reduced.shape[1]}")
from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(threshold=0.01)  # Remove low-variance features
X_train_reduced = var_thresh.fit_transform(X_train_reduced)
X_test_reduced = var_thresh.transform(X_test_reduced)

print(f"Features after variance thresholding: {X_train_reduced.shape[1]}")

# Now apply RFECV on reduced features on a small subset of features for testing
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
selector = RFECV(rf, step=20, cv=3, scoring="r2", n_jobs=-1)  # Increase step size, reduce CV folds

print("Starting RFECV on the full dataset...")
selector.fit(X_train_reduced, y_train)  # Run RFECV on the full dataset

print(f"RFECV completed. Optimal features: {selector.n_features_}")

# Apply final feature selection
X_train_selected = selector.transform(X_train_reduced)
X_test_selected = selector.transform(X_test_reduced)

print(f"Optimal number of features: {selector.n_features_}")

# Hyperparameter tuning using Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),  # L1 Regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)  # L2 Regularization
    }
    model = XGBRegressor(**params, random_state=seed)
    model.fit(X_train_selected, y_train)
    preds = model.predict(X_test_selected)
    return mean_absolute_error(y_test, preds)

if retrain:
    print("Optimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction="minimize")
    with tqdm(total=50, desc="Optimizing Hyperparameters with Optuna", leave=True) as pbar:
        for _ in range(50):
            study.optimize(objective, n_trials=1)
            pbar.update(1)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # Train final model with best parameters
    best_model = XGBRegressor(**best_params, random_state=seed)
    with tqdm(total=100, desc="Training Stacking Model", leave=True) as pbar:
        best_model.fit(X_train_selected, y_train)
        pbar.update(100)
    joblib.dump(best_model, model_path)
    print(f"Optimized model saved to {model_path}")
else:
    print("Using pre-trained model.")
    best_model = model

# Implement Stacking Instead of VotingRegressor
stack_model = StackingRegressor(
    estimators=[
        ("RandomForest", RandomForestRegressor(n_estimators=200)),
        ("XGBoost", XGBRegressor(n_estimators=500, learning_rate=0.05))
    ],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.01)
)

# Initialize KFold for cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=seed)

# Train and evaluate model using cross-validation
for train_idx, val_idx in tqdm(cv.split(X_train_selected, y_train), total=cv.get_n_splits(), desc="Cross-validation Progress", dynamic_ncols=True, leave=True):
    with tqdm(total=100, desc=f"Training Fold {train_idx}", leave=False) as pbar:
        X_tr, X_val = X_train_selected[train_idx], X_train_selected[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        logging.info(f"Training on fold {train_idx} started.")
        
        # Train the model and update progress
        stack_model.fit(X_tr, y_tr)
        pbar.update(100)  # Mark as complete

        logging.info(f"Training on fold {train_idx} completed.")

        # Predict on validation set
        y_pred = stack_model.predict(X_val)

        # Evaluate
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"Validation - MAE: {mae:.2f}, R²: {r2:.2f}")

# Evaluate model on test data
y_pred_test = stack_model.predict(X_test_selected)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\nTest Data Evaluation:")
print(f"MAE: {mae_test:.2f}, R²: {r2_test:.2f}")
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Compute and print RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# Residuals histogram
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color="purple")
plt.title("Residual Distribution (log Ki)")
plt.xlabel("Residual (log Ki)")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Cross-validated R² scoring
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(stack_model, X_train_selected, y_train, cv=5, scoring='r2')
print(f"Cross-validated R² scores: {cv_scores}")
print(f"Mean CV R²: {np.mean(cv_scores):.2f}")

# Cross-validated R² boxplot
plt.figure(figsize=(6, 5))
sns.boxplot(data=cv_scores, color='orchid')
plt.title("Cross-Validated R² Scores")
plt.ylabel("R²")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Feature importance visualization
importances_rf = stack_model.named_estimators_['RandomForest'].feature_importances_  # RandomForest
importances_xgb = stack_model.named_estimators_['XGBoost'].feature_importances_  # XGBoost

# Average feature importances
importances = (importances_rf + importances_xgb) / 2
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

def predict_ki_for_unknowns(sdf_file, model, scaler):
    print("\nProcessing unknown molecules...")

    # Load unknown molecules
    sdf_supplier = Chem.SDMolSupplier(sdf_file)
    unknown_mols = [mol for mol in tqdm(sdf_supplier, desc="Processing Unknown Molecules", leave=True) if mol is not None]

    if not unknown_mols:
        raise ValueError("No valid molecules found in the unknown SDF file.")

    # Generate fingerprints
    morgan_gen = GetMorganGenerator(radius=2, fpSize=4096)
    unknown_fingerprints = [morgan_gen.GetFingerprint(mol) for mol in unknown_mols]

    # Convert fingerprints to NumPy array
    unknown_X = fingerprints_to_numpy(unknown_fingerprints)

    # Generate molecular descriptors
    unknown_extra_features = np.array([get_descriptors(mol) for mol in unknown_mols])

    # Append descriptors to fingerprint data
    unknown_X = np.hstack((unknown_X, unknown_extra_features))

    # Normalize features using the same scaler
    unknown_X = scaler.transform(unknown_X)
    
    # Apply the same feature selection process as training
    unknown_X = selector_initial.transform(unknown_X)
    unknown_X = var_thresh.transform(unknown_X)
    unknown_X = selector.transform(unknown_X)  # Final feature selection

    # Predict log(Ki) values
    preds = np.array([tree.predict(unknown_X) for tree in model.estimators_])
    log_ki_predictions = preds.mean(axis=0)  # Mean prediction
    uncertainty = preds.std(axis=0)  # Standard deviation (uncertainty)
    confidence_interval = 1.96 * uncertainty  # 95% Confidence Interval

    # Define a confidence threshold (e.g., 100 nM)
    confidence_threshold = 100  # You can adjust this

    # Determine prediction confidence
    confidence_labels = ["Confident" if ci < confidence_threshold else "Uncertain" for ci in confidence_interval]

    # Convert back to original Ki values
    ki_predictions = np.expm1(log_ki_predictions)  # Reverse log transformation

    # Print results
    print("\nPredicted Ki values for unknown molecules:")
    for i, (pred, unc, ci) in enumerate(zip(ki_predictions, uncertainty, confidence_interval)):
        print(f"Molecule {i+1}: Predicted Ki = {pred:.2f} nM, Uncertainty ±{unc:.2f}, 95% CI ±{ci:.2f}")

    return ki_predictions, confidence_labels

unknown_sdf_path = "/Users/sakuta/Desktop/Desktop/Major_Project/Project/MeFSAT_3D_Structures.sdf"
predictions, confidences = predict_ki_for_unknowns(unknown_sdf_path, stack_model, scaler)

# Define a function to determine activity scale
def get_activity_scale(ki):
    if ki <= 50:
        return "+"
    elif 50 < ki <= 500:
        return "++"
    else:
        return "+++"

# Prepare predictions DataFrame for activity table
predictions_df = pd.DataFrame({
    "Molecule": range(1, len(predictions) + 1),
    "Pred Ki": predictions,
    "Confidence": confidences
})

# Add activity scales
predictions_df["Pred Scale"] = predictions_df["Pred Ki"].apply(get_activity_scale)

# Save to CSV
csv_filename = "molecule_activity_table.csv"
predictions_df.to_csv(csv_filename, index=False)

# -----------------------------------------------
# Evaluate Prediction Reliability
# -----------------------------------------------

# 1. Tanimoto Similarity to Training Molecules
print("\nCalculating Tanimoto similarity to training data...")
unknown_sdf_supplier = Chem.SDMolSupplier(unknown_sdf_path)
unknown_mols = [mol for mol in unknown_sdf_supplier if mol is not None]
morgan_gen = GetMorganGenerator(radius=2, fpSize=4096)
unknown_fingerprints = [morgan_gen.GetFingerprint(mol) for mol in unknown_mols]
max_similarities = []
for unk_fp in tqdm(unknown_fingerprints, desc="Tanimoto Similarity"):
    similarities = DataStructs.BulkTanimotoSimilarity(unk_fp, fingerprints)
    max_similarities.append(max(similarities))

predictions_df["Max Tanimoto to Train"] = max_similarities

# Visualize similarity distribution
plt.figure(figsize=(8, 6))
sns.histplot(max_similarities, bins=30, kde=True, color="blue")
plt.title("Max Tanimoto Similarity to Training Molecules")
plt.xlabel("Tanimoto Similarity")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 2. Compare Predicted vs Training Ki Distribution
plt.figure(figsize=(10, 6))
plt.hist(np.expm1(y), bins=30, alpha=0.5, label="Training Ki", color="skyblue")
plt.hist(predictions, bins=30, alpha=0.5, label="Predicted Ki", color="orange")
plt.title("Distribution of Ki Values: Training vs Predicted")
plt.xlabel("Ki (nM)")
plt.ylabel("Count")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

import seaborn as sns

# Enhanced Confidence Bar Chart
plt.figure(figsize=(8, 6))
sns.countplot(data=predictions_df, x="Confidence", hue="Confidence", palette={"Confident": "green", "Uncertain": "orange"}, legend=False)
plt.title("Prediction Confidence Distribution", fontsize=14)
plt.xlabel("Confidence Level", fontsize=12)
plt.ylabel("Number of Molecules", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Scatter Plot: Predicted Ki vs Molecule Index
plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(1, len(predictions) + 1), y=predictions, hue=confidences,
                palette={"Confident": "green", "Uncertain": "orange"}, s=60)
plt.title("Predicted Ki Values with Confidence Labels", fontsize=14)
plt.xlabel("Molecule Index", fontsize=12)
plt.ylabel("Predicted Ki (nM)", fontsize=12)
plt.legend(title="Confidence")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Histogram of Predicted Ki Values by Confidence
plt.figure(figsize=(10, 6))
sns.histplot(data=predictions_df, x="Pred Ki", hue="Confidence", bins=30,
             palette={"Confident": "green", "Uncertain": "orange"}, element="step", stat="count", common_norm=False)
plt.title("Distribution of Predicted Ki Values by Confidence", fontsize=14)
plt.xlabel("Predicted Ki (nM)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print(f"Table saved as {csv_filename}")