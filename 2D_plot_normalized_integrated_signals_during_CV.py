import os
'''
after gaussian integration signals time series

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths




input_file = '...'
input_folder = os.path.dirname(input_file)
output_folder = os.path.join(input_folder, 'Integrations Different Cycles')  # output folder inside the same directory

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.read_csv(input_file).iloc[:-1]

peaks = ['Mean 3680', 'Mean 3520', 'Mean 3360', 'Mean 3210', 'Mean 3100', 'Mean 2870']
columns_of_interest = ['Time (s)'] + peaks
df_filtered = df[columns_of_interest]

#%% normalization steps following C´s protocol

# 1
min_per_peak = df_filtered[peaks].min()
max_per_peak = df_filtered[peaks].max()

# 2
absolute_min = min_per_peak.min()
absolute_max = max_per_peak.max()

# 3
magnitude_per_peak = max_per_peak - min_per_peak

# 4
absolute_magnitude_per_peak = magnitude_per_peak.abs()

# 5
absolute_magnitude = absolute_magnitude_per_peak.sum()

# 6
fraction_per_peak = absolute_magnitude_per_peak / absolute_magnitude

# 7
adjusted_values = df_filtered[peaks] + abs(absolute_min)

# 8
scaled_values = adjusted_values.multiply(fraction_per_peak, axis=1)

# 9
percentage_values = scaled_values.div(scaled_values.sum(axis=1), axis=0) * 100

#%%

df_percentage = pd.concat([df_filtered['Time (s)'], percentage_values], axis=1)

# Divide the new data into cycles
cycle_duration = 141 # seconds per cycle, since 0.01 V/s
time_col = df_percentage['Time (s)']
start_index = 0
cycle_count = 1

all_cycles_data = []

cmap = get_cmap('magma')
colors = [cmap(i) for i in np.linspace(0.15, 1, len(peaks))]  # not using the black tone of magma

while start_index < len(time_col):
    end_index = np.where(time_col >= time_col[start_index] + cycle_duration)[0]     # Find the indices where the time values fall within the current cycle
    
    if len(end_index) > 0:
        end_index = end_index[0]  # Take the first instance where the time exceeds the cycle duration
    else:
        end_index = len(time_col)  # Last cycle may be incomplete, take until the end

    cycle_df = df_percentage.iloc[start_index:end_index]     # Slice the DataFrame for this cycle
    
    all_cycles_data.append(cycle_df) # Collect the data for combined plot

    if cycle_df.empty:
        print(f"Cycle {cycle_count} has no data, skipping.")
        start_index = end_index
        cycle_count += 1
        continue
    time = cycle_df['Time (s)']
    percentages = [cycle_df[peak] for peak in peaks]

    fig, ax = plt.subplots(figsize=(10, 6))  # Control figure size
    ax.stackplot(time, percentages, labels=peaks, colors=colors)
    ax.legend(loc='upper right', fontsize='small', title='Peaks')  # Adjust legend
    ax.set_title(f'Percentage Values Over Time - Cycle {cycle_count}', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_ylim(0, 100) # Because of plotting percentages
    ax.set_xlim(time.min(), time.max())
    #ax.grid(True, linestyle='--', alpha=0.7)  # Grid option

    plot_filename_png = os.path.join(output_folder, f'cycle_{cycle_count}_percentage_plot.png')
    plot_filename_svg = os.path.join(output_folder, f'cycle_{cycle_count}_percentage_plot.svg')
    fig.tight_layout() 
    fig.savefig(plot_filename_png)
    fig.savefig(plot_filename_svg)
    plt.close(fig)

    # Move to the next cycle
    start_index = end_index
    cycle_count += 1

# Plot all cycles together
fig, ax = plt.subplots(figsize=(12, 8))
for cycle_data in all_cycles_data:
    time = cycle_data['Time (s)']
    percentages = [cycle_data[peak] for peak in peaks]
    ax.stackplot(time, percentages, colors=colors)

ax.legend(peaks, loc='upper right', fontsize='small', title='Peaks')
ax.set_title('Percentage Values Over Time - All Cycles Combined', fontsize=14)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_ylim(0, 100)

# Fixing because of blank space
combined_time_min = min([cycle['Time (s)'].min() for cycle in all_cycles_data])
combined_time_max = max([cycle['Time (s)'].max() for cycle in all_cycles_data])
ax.set_xlim(combined_time_min, combined_time_max)

#ax.grid(True, linestyle='--', alpha=0.7)

combined_plot_filename_png = os.path.join(output_folder, 'all_cycles_percentage_plot.png')
combined_plot_filename_svg = os.path.join(output_folder, 'all_cycles_percentage_plot.svg')
fig.tight_layout()
fig.savefig(combined_plot_filename_png)
fig.savefig(combined_plot_filename_svg)
plt.close(fig)
