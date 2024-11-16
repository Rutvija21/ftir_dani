
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import os
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths



csv_file_path = r"..."
df = pd.read_csv(csv_file_path)
wavenumbers = df.iloc[:, 0]
spectra = df.iloc[:, 1:910]

folder_name = os.path.basename(os.path.dirname(csv_file_path))

time = np.linspace(1.1 * spectra.shape[1], 0, spectra.shape[1]) 


def vertical_lines(real_experiment_time, experiment_classification):
    time_ticks = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    time_positions = np.interp(time_ticks, real_experiment_time, range(len(real_experiment_time)))
    
    for position in time_positions:
        plt.axhline(position, color='black', alpha=0.4, linestyle='--')


    if experiment_classification == '_07':
        yticks_positions = np.interp(time_ticks, real_experiment_time, range(len(real_experiment_time)))
        yticks_labels = ["-0.4 V", "-0.05 V"] * ((len(yticks_positions) + 1) // 2)
        for i, position in enumerate(yticks_positions[:-1]):
            plt.text(40, position + (yticks_positions[i + 1] - position) / 2, yticks_labels[i], ha='right', va='center')
        
        plt.text(40, yticks_positions[0] - (yticks_positions[1] - yticks_positions[0]) / 2, yticks_labels[-1], ha='right', va='center')
        plt.text(40, yticks_positions[-1] + (yticks_positions[-1] - yticks_positions[-2]) / 2, yticks_labels[0], ha='right', va='center')

    if experiment_classification == '_08':
        yticks_positions = np.interp(time_ticks, real_experiment_time, range(len(real_experiment_time)))
        yticks_labels = ["-0.8 V", "-0.4 V"] * ((len(yticks_positions) + 1) // 2)
        for i, position in enumerate(yticks_positions[:-1]):
            plt.text(40, position + (yticks_positions[i + 1] - position) / 2, yticks_labels[i], ha='right', va='center')
        
        plt.text(40, yticks_positions[0] - (yticks_positions[1] - yticks_positions[0]) / 2, yticks_labels[-1], ha='right', va='center')
        plt.text(40, yticks_positions[-1] + (yticks_positions[-1] - yticks_positions[-2]) / 2, yticks_labels[0], ha='right', va='center')

    if experiment_classification == '_09':
        yticks_positions = np.interp(time_ticks, real_experiment_time, range(len(real_experiment_time)))
        yticks_labels = ["-1.1 V", "-0.8 V"] * ((len(yticks_positions) + 1) // 2)
        for i, position in enumerate(yticks_positions[:-1]):
            plt.text(40, position + (yticks_positions[i + 1] - position) / 2, yticks_labels[i], ha='right', va='center')
        
        plt.text(40, yticks_positions[0] - (yticks_positions[1] - yticks_positions[0]) / 2, yticks_labels[-1], ha='right', va='center')
        plt.text(40, yticks_positions[-1] + (yticks_positions[-1] - yticks_positions[-2]) / 2, yticks_labels[0], ha='right', va='center')

def save_figures(folder_path, filename):
    png_path = os.path.join(folder_path, f"{filename}.png")
    svg_path = os.path.join(folder_path, f"{filename}.svg")

    plt.savefig(png_path)
    plt.savefig(svg_path, format='svg', transparent=True)

def plot_spectra(spectra, start_wavenumber, end_wavenumber):
    start_index = np.where(wavenumbers >= start_wavenumber)[0][0]
    end_index = np.where(wavenumbers <= end_wavenumber)[0][-1]
    
    spectra_subset = spectra.iloc[start_index:end_index + 1, :]     
    
    fig, ax = plt.subplots(figsize=(3.5, 12))  # Adjust the figure size as needed. (3.5, 12) for zoomed in regions, (10, 12) for full spectrum
    img = ax.imshow(spectra_subset.T, cmap='magma', aspect='auto', extent=[start_wavenumber, end_wavenumber, time[0], time[-1]])
    plt.xlabel('Wavenumbers (cm$^{-1}$)', fontsize=18, rotation=180)
    plt.ylabel('Time (s)', fontsize=18)
    plt.title('Time series evolution FTIR spectra', fontsize=18)
    plt.xticks(fontsize=16, rotation='vertical')
    plt.yticks(fontsize=16, rotation='vertical')
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.xaxis.set_major_locator(MaxNLocator(3))
    cbar = fig.colorbar(img)
    cbar.set_label('Intensity', fontsize=16)
 

    
    real_experiment_time = np.arange(len(spectra)) * 1.1
    folder_path = os.path.dirname(csv_file_path)
    folder_name = os.path.basename(os.path.dirname(csv_file_path))
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.basename(folder_path) + "_withoutbackground_correction"
    
    
#    vertical_lines(real_experiment_time, '_08')
    save_figures(folder_path, filename)

    plt.show()
    
    
plot_spectra(spectra, 1100, 2200)
