import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('hansen_dwi.csv')

# Define the thresholds and outcome variable
thresholds = [0.08, 0.15]
outcome = 'recidivism'

# Function to compute MSE for a given bandwidth and threshold
def compute_mse(data, bandwidth, threshold, outcome):
    # Data within bandwidth around the threshold
    data_band = data[(data['bac1'] >= threshold - bandwidth) & (data['bac1'] <= threshold + bandwidth)].copy()
    data_band['treatment'] = (data_band['bac1'] > threshold).astype(int)
    # Local linear regression
    X = data_band[['bac1', 'treatment']]
    y = data_band[outcome]
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return mse

# Function to perform cross-validation over a range of bandwidths for multiple thresholds
def bandwidth_cv(data, thresholds, outcome, bandwidths, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best_mse = np.inf
    best_bandwidth = None
    for bandwidth in bandwidths:
        mse_list = []
        for threshold in thresholds:
            threshold_mse = []
            for train_index, test_index in kf.split(data):
                # Split data
                train_data, test_data = data.iloc[train_index], data.iloc[test_index]
                # Compute MSE on test set
                mse = compute_mse(test_data, bandwidth, threshold, outcome)
                threshold_mse.append(mse)
            # Average MSE across folds for the current threshold
            mse_list.append(np.mean(threshold_mse))
        # Average MSE across all thresholds for the current bandwidth
        avg_mse = np.mean(mse_list)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_bandwidth = bandwidth
        # print(f'Bandwidth: {bandwidth}, Avg MSE: {avg_mse}')
    return best_bandwidth

# Range of potential bandwidths to evaluate
bandwidths = np.linspace(0.01, 0.15, 100)

# Perform cross-validation to find the optimal bandwidth for all thresholds
optimal_bandwidth = bandwidth_cv(data, thresholds, outcome, bandwidths)
print(f'The optimal bandwidth for the thresholds 0.08 and 0.15 BAC: {optimal_bandwidth:.3f}')
