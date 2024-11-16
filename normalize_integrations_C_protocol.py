import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths




input_file = '...'
input_folder = os.path.dirname(input_file)  # Get the folder where the input CSV is located
output_folder = os.path.join(input_folder, 'Integrations Water Normalized Method')  # Create output folder inside the same directory

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.read_csv(input_file).iloc[:-1]

peaks = ['Mean 3680.00 cm⁻¹', 'Mean 3520.00 cm⁻¹', 'Mean 3360.00 cm⁻¹', 
         'Mean 3210.00 cm⁻¹', 'Mean 3100.00 cm⁻¹', 'Mean 2870.00 cm⁻¹']
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

cycle_duration = 100 # seconds per cycle, since 0.01 V/s
time_col = df_percentage['Time (s)']
start_index = 0
cycle_count = 1

all_cycles_data = []

cmap = get_cmap('magma')
colors = [cmap(i) for i in np.linspace(0.15, 1, len(peaks))]  # not using the black tone of magma

while start_index < len(time_col):
    # Find the indices where the time values fall within the current cycle
    end_index = np.where(time_col >= time_col[start_index] + cycle_duration)[0]
    
    if len(end_index) > 0:
        end_index = end_index[0]  # Take the first instance where the time exceeds the cycle duration
    else:
        end_index = len(time_col)  # Last cycle may be incomplete, take until the end

    # Slice the DataFrame for this cycle
    cycle_df = df_percentage.iloc[start_index:end_index]
    
    all_cycles_data.append(cycle_df) # Collect the data for combined plot

    # Check if the cycle_df is empty
    if cycle_df.empty:
        print(f"Cycle {cycle_count} has no data, skipping.")
        start_index = end_index
        cycle_count += 1
        continue
    # Plot the percentage values for each cycle
    time = cycle_df['Time (s)']
    percentages = [cycle_df[peak] for peak in peaks]

    fig, ax = plt.subplots(figsize=(10, 6))  # Control figure size
    ax.stackplot(time, percentages, labels=peaks, colors=colors)
    ax.legend(loc='upper right', fontsize='small', title='Peaks')  # Adjust legend
    ax.set_title(f'Percentage Values Over Time - Cycle {cycle_count}', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_ylim(0, 100)  # Set y-limit to 100% since we are plotting percentages
    ax.set_xlim(time.min(), time.max())
    #ax.grid(True, linestyle='--', alpha=0.7)  # Add a grid for better readability

    # Save the plot for the current cycle
    plot_filename_png = os.path.join(output_folder, f'cycle_{cycle_count}_percentage_plot.png')
    plot_filename_svg = os.path.join(output_folder, f'cycle_{cycle_count}_percentage_plot.svg')
    fig.tight_layout() 
    fig.savefig(plot_filename_png)
    fig.savefig(plot_filename_svg)
    plt.close(fig)

    # Move to the next cycle
    start_index = end_index
    cycle_count += 1

# Plot all cycles together using the same consistent colors
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

# x-limit based on the combined data to avoid blank space
combined_time_min = min([cycle['Time (s)'].min() for cycle in all_cycles_data])
combined_time_max = max([cycle['Time (s)'].max() for cycle in all_cycles_data])
ax.set_xlim(combined_time_min, combined_time_max)

#ax.grid(True, linestyle='--', alpha=0.7)

# Save the combined plot
combined_plot_filename_png = os.path.join(output_folder, 'all_cycles_percentage_plot.png')
combined_plot_filename_svg = os.path.join(output_folder, 'all_cycles_percentage_plot.svg')
fig.tight_layout()
fig.savefig(combined_plot_filename_png)  # Save as PNG
fig.savefig(combined_plot_filename_svg)  # Save as SVG
plt.close(fig)


# Combine all cycle data into a single DataFrame for export to CSV
combined_cycles_df = pd.concat(all_cycles_data, ignore_index=True)

#%% Save the combined data into a CSV file
csv_filename = os.path.join(output_folder, 'combined_cycles_percentage_data.csv')
combined_cycles_df.to_csv(csv_filename, index=False)


combined_cycles_file = os.path.join(output_folder, 'combined_cycles_percentage_data.csv')
df_combined = pd.read_csv(combined_cycles_file)
# Function to calculate stats for custom time ranges
def calculate_stats_for_ranges(df, time_col, ranges):
    filtered_df = df[df[time_col].apply(lambda x: any(lower <= x <= upper for lower, upper in ranges))]
    mean_values = filtered_df.mean()
    std_values = filtered_df.std()
    median_values = filtered_df.median()
    return mean_values, std_values, median_values

# Time ranges for custom calculations
ranges_101_199 = [(101, 199), (301, 399), (501, 599), (701, 799), (901, 999)]
ranges_201_299 = [(1, 99), (201, 299), (401, 499), (601, 699), (801, 899)]

# Calculate stats
mean_all = df_combined[peaks].mean()
std_all = df_combined[peaks].std()
median_all = df_combined[peaks].median()

mean_101_199, std_101_199, median_101_199 = calculate_stats_for_ranges(df_combined, 'Time (s)', ranges_101_199) # Calculate the stats for time ranges 101-199, 301-399, etc.
mean_201_299, std_201_299, median_201_299 = calculate_stats_for_ranges(df_combined, 'Time (s)', ranges_201_299) # Calculate the stats for time ranges 201-299, 401-499, etc.

stats_df = pd.DataFrame({
    'Time (s)': ['All Data', 'All Data', 'All Data', 'High overpotentials', 'High overpotentials', 'High overpotentials',
                 'Low overpotentials', 'Low overpotentials', 'Low overpotentials'],
    'Statistic': ['Mean', 'Std', 'Median', 'Mean', 'Std', 'Median', 'Mean', 'Std', 'Median']
})

# Add peak statistics
for peak in peaks:
    stats_df[peak] = [
        mean_all[peak], std_all[peak], median_all[peak],
        mean_101_199[peak], std_101_199[peak], median_101_199[peak],
        mean_201_299[peak], std_201_299[peak], median_201_299[peak]
    ]

# Concatenate the original data with the new statistics
df_with_stats = pd.concat([df_combined, stats_df], ignore_index=True)

# Save the updated combined CSV file with the statistics appended
df_with_stats.to_csv(combined_cycles_file, index=False)
