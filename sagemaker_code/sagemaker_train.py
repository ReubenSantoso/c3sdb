import argparse
import os
import time
import pickle
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from c3sdb.ml.kmcm import kmcm_p_grid, KMCMulti
from c3sdb.ml.metrics import compute_metrics_train_test, train_test_summary_figure

def print_debug(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def main():    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments. Defaults are set by SageMaker.
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-x', type=str, default=os.environ.get('SM_CHANNEL_TRAINX'))
    parser.add_argument('--train-y', type=str, default=os.environ.get('SM_CHANNEL_TRAINY'))
    parser.add_argument('--test-x', type=str, default=os.environ.get('SM_CHANNEL_TESTX'))
    parser.add_argument('--test-y', type=str, default=os.environ.get('SM_CHANNEL_TESTY'))
    
    args, _ = parser.parse_known_args()
    
    print_debug("Loading the dataset.")
    # Load data from the input files
    X_train = pd.read_csv(os.path.join(args.train_x, 'x-train.csv'))
    y_train = pd.read_csv(os.path.join(args.train_y, 'y-train.csv'))
    X_test = pd.read_csv(os.path.join(args.test_x, 'x-test.csv'))
    y_test = pd.read_csv(os.path.join(args.test_y, 'y-test.csv'))

    # Print shapes for debugging
    print_debug(f"X_train shape: {X_train.shape}")
    print_debug(f"y_train shape before processing: {y_train.shape}")
    print_debug(f"X_test shape: {X_test.shape}")
    print_debug(f"y_test shape before processing: {y_test.shape}")

    # Check for consistency
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in number of training samples between X and y"
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in number of testing samples between X and y"

    # Load data from the input files
    # Debugging: Print shapes and columns
    print_debug(f"y_train shape after loading: {y_train.shape}")
    print_debug(f"y_train columns: {y_train.columns.tolist()}")

    # Ensure y_train and y_test are 1D arrays
    # If your target column is named 'target', replace accordingly
    y_train = y_train.iloc[:, -1]  # or use y_train['target']
    y_test = y_test.iloc[:, -1]    # or use y_test['target']

    # Convert to NumPy arrays if necessary
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # Print shapes after processing
    print_debug(f"y_train shape after processing: {y_train.shape}")
    print_debug(f"y_test shape after processing: {y_test.shape}")

    # Center and Scaling
    print_debug("Centering and scaling the data.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    print_debug("Saving the scaler.")
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    #K means & SVR Cross Folding Hyperparameterization
    kmcm_svr_p_grid = kmcm_p_grid([4, 5], {"C": [1000, 10000], "gamma": [0.001, 0.1]}) 
    kmcm_svr_gs = GridSearchCV(KMCMulti(n_clusters=4, seed=2345, use_estimator=SVR(cache_size=1024, tol=1e-3)),
                            param_grid=kmcm_svr_p_grid, n_jobs=-1, cv=3, scoring="neg_mean_squared_error",
                            verbose=3)

    #Fit X_train and Y_train
    kmcm_svr_gs.fit(X_train_scaled, y_train)
    kmcm_svr_best = kmcm_svr_gs.best_estimator_

    # Save the trained model
    print_debug("Saving the trained model.")
    model_path = os.path.join(args.model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(kmcm_svr_best, f)
    
    # Compute predictions
    print_debug("Computing predictions for the training data.")
    y_pred_train = kmcm_svr_best.predict(X_train_scaled)

    print_debug("Computing predictions for the testing data.")
    y_pred_test = kmcm_svr_best.predict(X_test_scaled)
    
    # Calculate performance metrics using your custom functions
    print_debug("Calculating performance metrics using custom functions.")
    summary = compute_metrics_train_test(y_train, y_test, y_pred_train, y_pred_test)
    
    
    # Save metrics to a file
    print_debug("Generating summary figure and saving it as 'metrics_clean.png'.")
    train_test_summary_figure(summary, "metrics.png")

if __name__ == '__main__':
    main()
