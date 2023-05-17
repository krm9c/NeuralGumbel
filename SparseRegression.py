import numpy as np
from scipy.stats import genextreme

def fit_gev_moving_window(data, window_length):
    num_timesteps = data.shape[0]
    num_windows = num_timesteps - window_length + 1

    fit_results = []
    for i in range(num_windows):
        window_data = data[i:i+window_length]
        fit_result = genextreme.fit(window_data)
        fit_results.append(np.asarray(fit_result))

    return np.asarray(fit_results)

# Example usage
data = np.random.normal(size=(100, 1))  # Replace with your actual data
window_length = 10
result = fit_gev_moving_window(data, window_length)

# pip install pysindy
from pysindy import SINDy
from sklearn.preprocessing import PolynomialFeatures

# Define the SINDy-based dynamical system identification function
def sindy_identification(data, library_degree):
    # Split the data into time and state variables
    time = np.linspace(0, 1, data.shape[0])
    state = data[:, :]

    # Create polynomial features
    poly = PolynomialFeatures(degree=library_degree)
    state_poly = poly.fit_transform(state)

    # Perform SINDy identification
    sindy = SINDy(feature_names=poly.get_feature_names_out()[1:])
    sindy.fit(state_poly, t=time)

    # Retrieve the identified model coefficients
    model_coefficients = sindy.coefficients()

    return model_coefficients

# Example usage
library_degree = 2
model_coefficients = sindy_identification(result, library_degree)

print(model_coefficients)
