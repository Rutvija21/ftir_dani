'''
two parts. First section does fitting and integration. Second one extracts integrated values of 
the peaks that you choose at a certain time to use later in the bar charts
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.integrate import trapz
from scipy.ndimage import rotate
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import os
from matplotlib.cm import magma
import matplotlib as mpl
mpl.use('SVG')

mpl.rcParams['svg.fonttype'] = 'none'


def process_spectrum_file(file_path, start_reciprocal_cm=1101, 
                          end_reciprocal_cm=3999,
                          start_reciprocal_cm_bkg=2500, 
                          end_reciprocal_cm_bkg=3997,
                          spectrum_to_plot_as_example=600, 
                          experiment_classification='_08',
                          #experiment_classification = folder_name[-3:]
                          liquid='H2O'):

    folder = os.path.dirname(file_path)
    folder_name = os.path.basename(folder)
    
    if liquid == 'H2O':
        
        initial_amplitudes = [1,
                              1,
                                  1,
                                  1,
                                  1,
                                      1,
                                      1,
                                          1,
                                          1,
                                              1,
                                                    1,
                                                    1,
                                                        1,
                                                        1,
                                                        1,
                                                            1,
                                                            1,
                                                            1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1
                                                                
                                                   ] 
        initial_params = initial_amplitudes
        
        mean_values = [3680, 
                       3520,
                            3360, 
                            3210,
                            3100,
                                2870,
                                2800,
                                    2367,
                                    2350, 
                                    2320, 
                                        2127, 
                                            2085, 
                                            2078, 
                                            2050,
                                                1800,
                                                1700, 
                                                1639, 
                                                    1610, 
                                                    1541, 
                                                    1508, 
                                                        1430, 
                                                        1368, 
                                                        1227, 
                                                        1170 
                                          ]
        
        
        sigma_values = [70, 
                        100,
                            100,
                            80,
                            100,
                                80,
                                80,
                                    10,
                                    10,
                                    10,
                                        100,
                                            30,
                                            10,
                                            30,
                                                100,
                                                30,
                                                40, 
                                                    40,
                                                    40,
                                                    40,
                                                        30,
                                                        40,
                                                        40,
                                                        40,
                                                        40
                                          ]
        
    

    def gaussian(x, amp, mean, sigma):
        return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))
    
    def combined_function(x, *amps):
        return sum(gaussian(x, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values))


    

    
    file_name = os.path.basename(file_path)
    
    if file_name.startswith("DS"):
        df = pd.read_csv(file_path, header=None, skiprows=1)
        df = df.iloc[:, :911]  # Activate for original
        df = df[::-1]
        reciprocal_cm = df.iloc[:, 0]
    elif file_name.startswith("Reconstructed"):
        df = pd.read_csv(file_path, header=None, skiprows=0)
        reciprocal_cm = df.iloc[:, 0]
    



    
    num_peaks = len(mean_values)
    labels = [mean_values[i] for i in range(num_peaks)]
    colors = magma(np.linspace(0, 1, num_peaks))
    
    
    
    folder_name = f"{start_reciprocal_cm}_to_{end_reciprocal_cm}"
    os.makedirs(folder_name, exist_ok=True)
    

    start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
    end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]

    experiment_number = spectrum_to_plot_as_example
    trimmed_wavenumbers = df.iloc[start_index:end_index+1, 0].values
    spectrum = df.iloc[start_index:end_index+1, experiment_number].values
    
    index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
    index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
    slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
    intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
    background_array = slope * trimmed_wavenumbers + intercept
    spectrum_corrected = spectrum - background_array
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
    plt.xlim(start_reciprocal_cm, end_reciprocal_cm)
    plt.gca().invert_xaxis()
    plt.legend(fontsize=18, ncol=7)
    plt.title('Raw data', fontsize=18)

    folder = os.path.dirname(file_path)
    folder_path = os.path.join(folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_withoutbackground_correction"

    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")

    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)

    plt.show()

    fitted_values = combined_function(reciprocal_cm[start_index:end_index+1], *popt)
    fit_df = pd.DataFrame({'Reciprocal_cm': reciprocal_cm[start_index:end_index+1], 'Fitted_Values': fitted_values})
    for i, (gaussian_component, label) in enumerate(zip(gaussians, labels)):
        fit_df[f'Gaussian_{label}'] = gaussian_component
    fit_csv_path = os.path.join(folder_path, f"{filename}_fitted_values.csv")
    fit_df.to_csv(fit_csv_path, index=False)
    
    
    
    
    
    experiment_numbers = df.columns[1:]
    experiment_time = np.arange(len(experiment_numbers)) * 1.1
    integrated_areas = {f'Peak {i}': [] for i in range(num_peaks)}
    reciprocal_cm = df.iloc[:, 0]
    
    
    for experiment_number in experiment_numbers:
        
        start_index = np.where(reciprocal_cm >= start_reciprocal_cm)[0][0]
        end_index = np.where(reciprocal_cm <= end_reciprocal_cm)[0][-1]
        trimmed_wavenumbers = df.iloc[start_index:end_index+1, 0].values
        spectrum = df.iloc[start_index:end_index+1, experiment_number].values
        index_start_bkg = np.where(trimmed_wavenumbers >= start_reciprocal_cm_bkg)[0][0]
        index_end_bkg = np.where(trimmed_wavenumbers <= end_reciprocal_cm_bkg)[0][-1]
        slope = (spectrum[index_end_bkg] - spectrum[index_start_bkg]) / (trimmed_wavenumbers[index_end_bkg] - trimmed_wavenumbers[index_start_bkg])
        intercept = spectrum[index_start_bkg] - slope * trimmed_wavenumbers[index_start_bkg]
        background_array = slope * trimmed_wavenumbers + intercept
        spectrum_corrected = spectrum - background_array
        popt, _ = curve_fit(combined_function, trimmed_wavenumbers, spectrum_corrected, p0=initial_params, maxfev=10000)
        amps = popt
        gaussians = [gaussian(trimmed_wavenumbers, amp, mean, sigma) for amp, mean, sigma in zip(amps, mean_values, sigma_values)]
        for i in range(num_peaks):
            area = trapz(gaussians[i], trimmed_wavenumbers)
            integrated_areas[f'Peak {i}'].append(area)
                
    #%%  Saving areas as csv
    
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
    plt.xlim(0,1000)
    
    intersections = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='black', alpha=0.1)
    

    
    if experiment_classification == '_07':
        text_labels = ["-0.05 V", "-0.4 V"]
    elif experiment_classification == '_08':
        text_labels = ["-0.4 V", "-0.8 V"]
    elif experiment_classification == '_09':
        text_labels = ["-0.8 V", "-1.1 V"]
    else: # if error
        text_labels = ["-0.4 V", "-0.8 V"]
    
    
    for i in range(len(intersections) - 1):
        x_start = intersections[i]
        x_end = intersections[i + 1]
        text_label = text_labels[i % 2]
    
    
        text_x = (x_start + x_end) / 2
        plt.text(text_x, plt.ylim()[1], text_label, rotation=45, va='bottom', ha='center', fontsize=16)
    
    
    folder = os.path.dirname(file_path)
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
    
    folder = os.path.dirname(file_path)
    folder_path = os.path.join(folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder) + "_integrated areas"
    
    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")
    
    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)
    plt.show()
    



if __name__ == "__main__":
    file_patterns = [
        "mean-center contribution/ReconstructedData_PCs1.csv",
        "PC1/ReconstructedData_PCs1.csv",
        "PC2-15/ReconstructedData_PCs2&3&4&5&6&7&8&9&10&11&12&13&14&15.csv"
    ]

    # Subfolders to include
    include_folders = {"DS_00132", "DS_00133", "DS_00134","DS_00127","DS_00163", "DS_00131",
                       "DS_00138","DS_00135", "DS_00139", "DS_00136", "DS_00140", "DS_00137",
                       "DS_00141", "DS_00144", "DS_00142", "DS_00145", "DS_00143", "DS_00146",
                       "DS_00181", "DS_00180", "DS_00148", "DS_00152", "DS_00149", "DS_00153",
                       }

    
    suffixes = ["_07", "_08", "_09"]

    base_dir = "..."


    for folder_name in include_folders:
        folder_path = os.path.join(base_dir, folder_name)

        for suffix in suffixes:
            alternating_file = os.path.join(folder_path, f"{folder_name}{suffix}.csv")

            if os.path.exists(alternating_file):
                print(f"Processing file: {alternating_file}")
                process_spectrum_file(
                    file_path=alternating_file,
                    start_reciprocal_cm=1101,
                    end_reciprocal_cm=3999,
                    start_reciprocal_cm_bkg=2500,
                    end_reciprocal_cm_bkg=3997,
                    spectrum_to_plot_as_example=600,
                    experiment_classification=suffix,
                    liquid='H2O'
                )
            else:
                print(f"Alternating file not found: {alternating_file}")

        for pattern in file_patterns:
            file_path = os.path.join(folder_path, pattern)

            if os.path.exists(file_path):
                experiment_classification = None
                for suffix in suffixes:
                    test_file = os.path.join(folder_path, f"{folder_name}{suffix}.csv")
                    if os.path.exists(test_file):
                        experiment_classification = suffix
                        break

                if experiment_classification is None:
                    experiment_classification = '_unknown'

                print(f"Processing file: {file_path} with classification {experiment_classification}")
                process_spectrum_file(
                    file_path=file_path,
                    start_reciprocal_cm=1101,
                    end_reciprocal_cm=3999,
                    start_reciprocal_cm_bkg=2500,
                    end_reciprocal_cm_bkg=3997,
                    spectrum_to_plot_as_example=600,
                    experiment_classification=experiment_classification,
                    liquid='H2O'
                )
            else:
                print(f"File not found: {file_path}")
                
#%% extract the integrated areas at time 195.8

import os
import pandas as pd
import numpy as np

results = []
processed_files = set()
target_peaks = [3680, 3520, 3360, 3210, 3100, 2870]
target_time = 195.8

def find_integrated_area_files(base_path, ignore_folders):
    integrated_area_files = []
    for root, dirs, files in os.walk(base_path): 
        dirs[:] = [d for d in dirs if d.lower() not in ignore_folders]
        for file in files:
            if file.endswith("_integrated_areas.csv"):
                integrated_area_files.append(os.path.join(root, file))
    return integrated_area_files

ignore_folders = {"raw data", "1635_peak"}

for folder_name in include_folders:
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    integrated_files = find_integrated_area_files(folder_path, ignore_folders)
    if not integrated_files:
        print(f"No integrated areas files found in {folder_path}")
        continue

    for full_path in integrated_files:
        file_key = (os.path.basename(full_path), os.path.dirname(full_path))  # Unique identifier: (File, Folder)
        if file_key in processed_files:  # Skip if already processed
            print(f"Skipping duplicate file: {full_path}")
            continue
        processed_files.add(file_key)  # Mark as processed

        print(f"Processing file: {full_path}")
        df_integrated = pd.read_csv(full_path)

        if 'Time (s)' not in df_integrated.columns:
            print(f"'Time (s)' column not found in {full_path}")
            continue

        row = df_integrated[np.isclose(df_integrated['Time (s)'], target_time, atol=0.1)]
        if not row.empty:
            experiment_group = os.path.relpath(full_path, base_dir).split(os.sep)[0]
            row_data = {
                'Experiment': experiment_group,
                'File': os.path.basename(full_path),
                'Folder': os.path.dirname(full_path),
            }
            for peak in target_peaks:
                peak_col = f'Mean {peak}'
                row_data[peak_col] = row[peak_col].values[0] if peak_col in row else None
            results.append(row_data)
        else:
            print(f"Target time {target_time} not found in {full_path}")

if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.drop_duplicates()
    consolidated_csv_path = os.path.join(base_dir, "consolidated_integrated_areas.csv")
    results_df.to_csv(consolidated_csv_path, index=False)
else:
    print("Error")
