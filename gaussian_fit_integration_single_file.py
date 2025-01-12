# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:12:33 2025

@author: Rutvija
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.integrate import trapz
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import os
from sklearn.metrics import r2_score
from matplotlib.cm import magma
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths


file_paths = {
    'df': r"E:\07012025\changing the discharge gap\21kV\1.0 mm_pellet\Exp._3_rep\CSV file\1 mm pellet_3.csv"
}

start_reciprocal_cm = 349
end_reciprocal_cm = 1141
start_reciprocal_cm_bkg = 350  
end_reciprocal_cm_bkg = 1140 
spectrum_to_plot_as_example = 600
experiment_classification = '_08'
liquid = 'H2O'


folder_name = os.path.basename(os.path.dirname(file_paths['df']))
#experiment_classification = folder_name[-3:]


dfs = {}

for key, file_path in file_paths.items():
    dfs[key] = pd.read_csv(file_path, header=None, skiprows=0) #original 955, others 0
#    dfs[key] = dfs[key].T
    dfs[key] = dfs[key].iloc[:, :] #Activate for original

#    dfs[key]=dfs[key][::-1] #activate for original
#    dfs[key]=dfs[key].dropna(axis=1, how='any') # drop columns with NaN, relevant for the CVs, where the start of a new cycle may not coincide with end of previous



    


if liquid == 'H2O':
    
    initial_amplitudes = [1, 1, 1, 1, 1
                                                            
                                               ] 
    
    mean_values = [760, 751, 744, 736, 725  
                                      ]
    
    
    sigma_values = [2, 2, 2, 3, 2
                                      ]
    
    initial_params = initial_amplitudes
    


def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def combined_function(x, *amps):
    return sum(gaussian(x, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values))

num_peaks = len(mean_values)
labels = [mean_values[i] for i in range(num_peaks)]
colors = magma(np.linspace(0, 1, num_peaks))


reciprocal_cm = dfs[key].iloc[:, 0]


folder_name = f"{start_reciprocal_cm}_to_{end_reciprocal_cm}"
os.makedirs(folder_name, exist_ok=True)

for key in dfs.keys():
    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]

    experiment_number = spectrum_to_plot_as_example
    trimmed_wavenumbers = dfs[key].iloc[start_index:end_index+1, 0].values
    spectrum = dfs[key].iloc[start_index:end_index+1, experiment_number].values
    
    #bkg
    index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
    index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
    
    slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
    
    intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
    background_array = slope * trimmed_wavenumbers + intercept
    spectrum_corrected = spectrum - background_array
        
    # Perform the curve fitting
    popt, _ = curve_fit(combined_function, trimmed_wavenumbers, spectrum_corrected, p0=initial_params, maxfev=10000)
    amps = popt
    gaussians = [gaussian(trimmed_wavenumbers, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values)]
    fitted_curve = combined_function(trimmed_wavenumbers, *popt)
    correlation_coefficient, _ = pearsonr(spectrum_corrected, fitted_curve)
    r_squared = correlation_coefficient ** 2
    print(f"R-squared: {r_squared}")    

        
    fig, ax = plt.subplots(figsize=(18, 8))
    for i, (gaussian_component, label, color) in enumerate(zip(gaussians, labels, colors)):
        plt.plot(reciprocal_cm[start_index:end_index+1], gaussian_component, label=label, color=color)
        plt.fill_between(reciprocal_cm[start_index:end_index+1], gaussian_component, color=color, alpha=0.6)
    plt.plot(reciprocal_cm[start_index:end_index+1], spectrum_corrected, label='Original Data')
    plt.plot(reciprocal_cm[start_index:end_index+1], fitted_curve, label='Fitted Curve')
    plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize=18)
    plt.ylabel('Intensity (a.u.)', fontsize=18)
#    plt.xlim(start_reciprocal_cm, end_reciprocal_cm)
    plt.gca().invert_xaxis()
    plt.legend(fontsize=18, ncol=7)
    plt.title('Raw data', fontsize=18)

    folder = os.path.dirname(file_paths[key])
    folder_path = os.path.join(folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_withoutbackground_correction"

    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")

    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)

    plt.show()

    # Save the fitted curve and the Gaussians to a CSV file
    fitted_values = combined_function(reciprocal_cm[start_index:end_index+1], *popt)
    fit_df = pd.DataFrame({'Reciprocal_cm': reciprocal_cm[start_index:end_index+1], 'Fitted_Values': fitted_values})
    for i, (gaussian_component, label) in enumerate(zip(gaussians, labels)):
        fit_df[f'Gaussian_{label}'] = gaussian_component
    fit_csv_path = os.path.join(folder_path, f"{filename}_fitted_values.csv")
    fit_df.to_csv(fit_csv_path, index=False)





experiment_numbers = dfs[key].columns[1:]
experiment_time = np.arange(len(experiment_numbers)) * 0.1

integrated_areas = {f'Peak {i}': [] for i in range(num_peaks)}

for key in dfs.keys():
    reciprocal_cm = dfs[key].iloc[:, 0]    
    for experiment_number in experiment_numbers:        
        start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
        end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]
        trimmed_wavenumbers = dfs[key].iloc[start_index:end_index+1, 0].values
        spectrum = dfs[key].iloc[start_index:end_index+1, experiment_number].values
        index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
        index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
        slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
        intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
        background_array = slope * trimmed_wavenumbers + intercept
        spectrum_corrected = spectrum - background_array
        popt, _ = curve_fit(combined_function, trimmed_wavenumbers, spectrum_corrected, p0=initial_params, maxfev=10000)
        amps = popt
        gaussians = [gaussian(trimmed_wavenumbers, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values)]

        # Calculate and store the integrated areas for each peak
        for i in range(num_peaks):
            area = trapz(gaussians[i], trimmed_wavenumbers)
            integrated_areas[f'Peak {i}'].append(area)
            
#%%  Saving instegrated areas as csv

integrated_areas_df = pd.DataFrame(integrated_areas)
integrated_areas_df['Time (s)'] = experiment_time
header_mapping = {f'Peak {i}': f'Mean {mean_values[i]}' for i in range(num_peaks)}
integrated_areas_df.rename(columns=header_mapping, inplace=True)
integrated_areas_df = integrated_areas_df[['Time (s)'] + [col for col in integrated_areas_df.columns if col != 'Time (s)']]
csv_filename = os.path.join(folder_path, f"{filename}_integrated_areas.csv")
integrated_areas_df.to_csv(csv_filename, index=False)

fig, ax = plt.subplots(figsize=(18, 8))

for i, peak_label in enumerate(integrated_areas.keys()):
    plt.plot(experiment_time, integrated_areas[peak_label], label=mean_values[i], color=colors[i])

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18, ncol=3)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('Integrated Area (a.u.)', fontsize=18)
#plt.ylim(-6,5)
#plt.xlim(0,1000)

# Add vertical lines at the specified intersections
intersections = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for intersection in intersections:
    plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)



if experiment_classification == '_07':
    text_labels = ["-0.05 V", "-0.4 V"]
elif experiment_classification == '_08':
    text_labels = ["-0.4 V", "-0.8 V"]
elif experiment_classification == '_09':
    text_labels = ["-0.8 V", "-1.1 V"]


for i in range(len(intersections) - 1):
    x_start = intersections[i]
    x_end = intersections[i + 1]
    text_label = text_labels[i % 2]


    text_x = (x_start + x_end) / 2  # Calculate the x-coordinate for the text label
    plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center', fontsize=16)


folder = os.path.dirname(file_paths[key])
folder_path = os.path.join(folder, folder_name)
os.makedirs(folder_path, exist_ok=True)
filename = os.path.basename(folder) + "_integration of deconvoluted peak"

png_path = os.path.join(folder_path, f"{filename}.png")
eps_path = os.path.join(folder_path, f"{filename}.eps")
svg_path = os.path.join(folder_path, f"{filename}.svg")

plt.subplots_adjust(top=0.9) 
   
plt.savefig(png_path)
plt.savefig(eps_path)
plt.savefig(svg_path, format='svg', transparent=True)

plt.show()

folder = os.path.dirname(file_paths[key])
folder_path = os.path.join(folder, folder_name)
os.makedirs(folder_path, exist_ok=True)
filename = os.path.basename(folder) + "_integrated areas"

png_path = os.path.join(folder_path, f"{filename}.png")
svg_path = os.path.join(folder_path, f"{filename}.svg")

plt.savefig(png_path)
plt.savefig(svg_path, format='svg', transparent=True)
plt.show()
