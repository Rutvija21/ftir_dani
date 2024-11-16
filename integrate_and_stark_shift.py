import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import curve_fit
import os

#%% Function to create a Gaussian curve
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

#%% Combined function for multiple Gaussian peaks with fixed sigma values
def combined_function(x, *params):
    num_gaussians = len(params) // 2
    y = np.zeros_like(x)
    for i in range(num_gaussians):
        amp = params[i*2]
        mean = params[i*2+1]
        sigma = sigma_values[i]  # Use the fixed sigma values
        y += gaussian(x, amp, mean, sigma)
    return y

#%% Load the data and choose the bubbled gas

file_paths = {
    'df': r"..."
}

folder_path = os.path.dirname(file_paths['df'])
folder_name = os.path.basename(os.path.dirname(file_paths['df']))
files_in_folder = os.listdir(folder_path)
csv_files = [file for file in files_in_folder if (file.startswith("DS_") or file.startswith("CZ_")) and file.endswith(".csv")]
if csv_files:
    csv_file = csv_files[0]  # First CSV file that matches the criteria
    experiment_classification = os.path.splitext(csv_file)[0][-3:]
else:
    print("No CSV file starting with 'DS_' found in the specified folder.")

dfs = {}
for key, file_path in file_paths.items():
    dfs[key] = pd.read_csv(file_path, header=None, skiprows=1)
    dfs[key] = dfs[key].iloc[:, :911]
#    dfs[key] = dfs[key][::-1]  # Reverse if needed

#%% Define the range of wavenumbers, initial parameters, and fixed sigmas

start_reciprocal_cm = 1900
end_reciprocal_cm = 2200
spectrum_to_plot_as_example = 600

reciprocal_cm = dfs[key].iloc[:, 0]

initial_params = [1, 2078, 1, 2050] # Example: Initial parameters for 2 Gaussians [amp1, mean1, amp2, mean2, ...]
sigma_values = [10, 30] # Example: Fixed sigma values for 2 Gaussians

filename = os.path.basename(folder_path)

#%% Plot single FTIR deconvoluted spectrum

for key in dfs.keys():
    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]

    trimmed_wavenumbers = dfs[key].iloc[start_index:end_index+1, 0].values
    spectrum = dfs[key].iloc[start_index:end_index+1, spectrum_to_plot_as_example].values

    # Background correction
    start_reciprocal_cm_bkg = 2000  
    end_reciprocal_cm_bkg = 2200
    index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
    index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
    slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
    intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
    background_array = slope * trimmed_wavenumbers + intercept
    spectrum_corrected = spectrum - background_array

    popt, _ = curve_fit(combined_function, reciprocal_cm[start_index:end_index+1], spectrum_corrected, p0=initial_params, maxfev=1000000)

    fitted_curve = combined_function(reciprocal_cm[start_index:end_index+1], *popt)
    gaussians = []
    for i in range(len(popt) // 2):
        amp = popt[i*2]
        mean = popt[i*2+1]
        sigma = sigma_values[i]  # Use the fixed sigma values
        gaussians.append(gaussian(reciprocal_cm[start_index:end_index+1], amp, mean, sigma))

    fig, ax = plt.subplots(figsize=(18, 8))
    for i, gauss in enumerate(gaussians):
        plt.plot(reciprocal_cm[start_index:end_index+1], gauss, label=f'Gaussian {i+1}')
        plt.fill_between(reciprocal_cm[start_index:end_index+1], gauss, alpha=0.6)
    plt.plot(reciprocal_cm[start_index:end_index+1], spectrum_corrected, label='Original Data')
    plt.plot(reciprocal_cm[start_index:end_index+1], fitted_curve, label='Fitted Curve')
    plt.xlabel('Wavenumbers (cm$^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.title(f'Spectrum {spectrum_to_plot_as_example}')
    plt.show()

#%%Evolution of the Integrated Areas and Peak Positions

# Create time array for experiments
experiment_numbers = dfs[key].columns[1:]
experiment_time = np.arange(len(experiment_numbers)) * 1.1

integrated_areas = {f'Area{i+1}': [] for i in range(len(initial_params)//2)}
peak_positions = {f'Peak{i+1}': [] for i in range(len(initial_params)//2)}

for key in dfs.keys():
    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]

    for experiment_number in experiment_numbers:
        trimmed_wavenumbers = dfs[key].iloc[start_index:end_index+1, 0].values
        spectrum = dfs[key].iloc[start_index:end_index+1, experiment_number].values

        index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
        index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
        slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
        intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
        background_array = slope * trimmed_wavenumbers + intercept
        spectrum_corrected = spectrum - background_array

        popt, _ = curve_fit(combined_function, reciprocal_cm[start_index:end_index+1], spectrum_corrected, p0=initial_params, maxfev=1000000)

        for i in range(len(popt) // 2):
            amp = popt[i*2]
            mean = popt[i*2+1]
            sigma = sigma_values[i]  # Use the fixed sigma values
            gauss = gaussian(reciprocal_cm[start_index:end_index+1], amp, mean, sigma)
            area = trapz(gauss, reciprocal_cm[start_index:end_index+1])

            integrated_areas[f'Area{i+1}'].append(area)
            peak_positions[f'Peak{i+1}'].append(mean)

    integrated_areas_df = pd.DataFrame(integrated_areas)
    integrated_areas_df['Time (s)'] = experiment_time
    integrated_areas_df = integrated_areas_df[['Time (s)'] + [col for col in integrated_areas_df.columns if col != 'Time (s)']]
    csv_filename = os.path.join(folder_path, f"{filename}_integrated_areas.csv")
    integrated_areas_df.to_csv(csv_filename, index=False)

    fig, ax = plt.subplots(figsize=(18, 8))
    for i in range(len(initial_params) // 2):
        plt.plot(experiment_time, integrated_areas[f'Area{i+1}'], label=f'Area {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Integrated Area (a.u.)')
    plt.xlim(0, experiment_time[-1])
    plt.legend()
    plt.tight_layout()
    plt.title('Evolution of Integrated Areas')
    plt.show()

    peak_positions_df = pd.DataFrame(peak_positions)
    peak_positions_df['Time (s)'] = experiment_time
    peak_positions_df = peak_positions_df[['Time (s)'] + [col for col in peak_positions_df.columns if col != 'Time (s)']]
    peak_positions_csv_filename = os.path.join(folder_path, f"{filename}_peak_shift.csv")
    peak_positions_df.to_csv(peak_positions_csv_filename, index=False)

    fig, ax = plt.subplots(figsize=(18, 8))
    for i in range(len(initial_params) // 2):
        plt.plot(experiment_time, peak_positions[f'Peak{i+1}'], label=f'Peak {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Wavenumbers (cm$^{-1}$)')
    plt.xlim(0, experiment_time[-1])
    plt.legend()
    plt.tight_layout()
    plt.title('Evolution of Peak Positions')
    plt.show()
