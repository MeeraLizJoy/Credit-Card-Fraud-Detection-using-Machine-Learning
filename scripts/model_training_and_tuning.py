import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score

def train_tune_evaluate():
    """
    Trains, tunes, and evaluates a Random Forest model.
    """

    # Load the balanced and selected data
    X_train_selected, X_test_selected, y_train_resampled, y_test = joblib.load('balanced_selected_data.joblib')

    # Train an initial Random Forest model
    rf_initial = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_initial.fit(X_train_selected, y_train_resampled)

    # Save the initial model
    joblib.dump(rf_initial, 'rf_initial_model.joblib')
    print("\nInitial Random Forest model saved as 'rf_initial_model.joblib'")

    # Define the hyperparameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Perform hyperparameter tuning using RandomizedSearchCV on a subset of the training data
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_selected, y_train_resampled, train_size=0.2, random_state=42)

    rf_random = RandomizedSearchCV(estimator=rf_initial, param_distributions=param_grid,
                                   n_iter=50, cv=3, verbose=2, random_state=42,
                                   n_jobs=-1)
    rf_random.fit(X_train_subset, y_train_subset)

    # Get the best hyperparameters
    best_params = rf_random.best_params_

    # Train the final Random Forest model with the best hyperparameters on the full training data
    rf_final = RandomForestClassifier(**best_params, random_state=42)
    rf_final.fit(X_train_selected, y_train_resampled)

    # Save the final model
    joblib.dump(rf_final, 'rf_final_model.joblib')
    print("\nFinal Random Forest model saved as 'rf_final_model.joblib'")

    # Make predictions on the test set
    y_pred_final = rf_final.predict(X_test_selected)

    # Evaluate the final model
    report_final = classification_report(y_test, y_pred_final)
    auc_final = roc_auc_score(y_test, rf_final.predict_proba(X_test_selected)[:, 1])

    # Print evaluation results
    print("\nFinal Tuned Random Forest Results:")
    print(report_final)
    print(f"AUC: {auc_final}")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split #Importing train_test split inside the main method so it doesnt give an error when running the script.
    train_tune_evaluate()
