'''
curve fitting for Cottrell exp
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths

def exponential_fit(x, a, b, c):
    return a * np.exp(b * x) + c

def linear_fit(x, m, c):
    return m * x + c

def fit_curves_to_data(file_path, row_range, column_index):
    data = pd.read_csv(file_path, delimiter='\t', header=None)

    y_data = data.iloc[row_range[0]-1:row_range[1], column_index-1].to_numpy()
    x_data = np.arange(1, len(y_data) + 1)

    # Fit the exponential curve
    params, covariance = curve_fit(exponential_fit, x_data, y_data, p0=(1, 0.01, np.mean(y_data)), maxfev=10000)
    fitted_y_data = exponential_fit(x_data, *params)
    equation = f"y = {params[0]:.2f} * exp({params[1]:.6f} * x) + {params[2]:.2f}"
    output_folder = os.path.join(os.path.dirname(file_path), 'Diffusion coefficient plots')
    os.makedirs(output_folder, exist_ok=True)

    # Plot exponential fit to original data
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_data, y_data, label='Datapoints', color='#000004', marker='o')
    ax.plot(x_data, fitted_y_data, label='Exponential Fit', color='#ff621e')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Scores (a.u.)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    ax.legend(loc='upper left', fontsize=10)
    fig.savefig(os.path.join(output_folder, 'exponential_fit.png'), format='png')
    fig.savefig(os.path.join(output_folder, 'exponential_fit.svg'), format='svg')

    # Plot original y_data vs 1/sqrt values
    reciprocal_sqrt_values = 1 / np.sqrt([100, 101.1, 102.2, 103.3, 104.4, 105.5, 106.6, 107.7, 108.8, 109.9])
    params_sqrt, covariance_sqrt = curve_fit(linear_fit, reciprocal_sqrt_values, y_data[:len(reciprocal_sqrt_values)])  # Fit only using available y_data
    fitted_y_sqrt = linear_fit(reciprocal_sqrt_values, *params_sqrt)
    
    fig, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(reciprocal_sqrt_values, y_data[:len(reciprocal_sqrt_values)], label='Data vs 1/sqrt(values)', color='#000004', marker='o')  # Use y_data
    ax2.plot(reciprocal_sqrt_values, fitted_y_sqrt, label='Linear Fit (1/sqrt)', color='#ff7f0e')
    
    ax2.set_xlabel('1/Sqrt of specified values', fontsize=12)
    ax2.set_ylabel('Scores (a.u.)', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    ax2.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    
    fig.savefig(os.path.join(output_folder, 'linear_fit_1_sqrt.png'), format='png')
    fig.savefig(os.path.join(output_folder, 'linear_fit_1_sqrt.svg'), format='svg')
    
    plt.show()

    equation_sqrt = f"y = {params_sqrt[0]:.2f} * 1/sqrt(x) + {params_sqrt[1]:.2f}"
    print("Exponential Fit Equation (Original):", equation)
    print("Linear Fit Equation (Sqrt):", equation_sqrt)

    return equation, params, y_data

file_path = '...'
row_range = (91, 100)
column_index = 2  # PC1 (notice column_index - 1)

equation, params, y_data = fit_curves_to_data(file_path, row_range, column_index)
