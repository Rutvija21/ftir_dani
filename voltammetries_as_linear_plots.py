
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths

def load_data(file_path):
    data = pd.read_csv(file_path, header=1)  # First row in files has the units
    return data

def plot_first_five_cycles(data, sweep_rate, voltage_increment, pH, save_path, diameter):
    conversion_factor_RHE = 0.197 + 0.059 * pH
    
    area = np.pi * ((diameter / 1000) / 2) ** 2  # area electrode in m^2
    
    fig, ax1 = plt.subplots(figsize=(15, 3))
    
    cmap = get_cmap('magma')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current Density (A/m^2)')
    ax1.tick_params(axis='y')
    
    
    total_time = 0  # Initialize total time for adjusting the x-axis
    all_times = []
    all_potentials = []
    tick_positions = []
    tick_labels = []

    for cycle in range(1, 11): # (1,6) for first five cycles, (1,11) for first 10 cycles, etc
        # Extract the relevant columns for the chosen cycle
        potential_col = 2 * (cycle - 1)
        current_col = 2 * (cycle - 1) + 1

        # Check if columns are within the data range
        if potential_col >= len(data.columns) or current_col >= len(data.columns):
            print(f"Cycle {cycle} is out of range. Skipping this cycle.")
            continue

        potentials = data.iloc[:, potential_col] + conversion_factor_RHE
        currents = data.iloc[:, current_col]
        current_density = currents / area  # A/m^2

        time = [total_time + i * voltage_increment / sweep_rate for i in range(len(potentials))]
        total_time = time[-1] + voltage_increment / sweep_rate  # Update total time
        print(f"Cycle {cycle} ends at time: {time[-1]:.2f} seconds")
        all_times.extend(time)
        all_potentials.extend(potentials)

        color = cmap((cycle - 1) / 10)  # Extracting magma cmap colors
        ax1.plot(time, current_density, label=f'Cycle {cycle}', color=color)

        # Add tick positions for the beginning of the cycle
        tick_positions.append(time[0])
        tick_labels.append(f'{potentials.iloc[0]:.2f}')

        # Find the points where the potential stops decreasing and starts increasing
        for i in range(1, len(potentials) - 1):
            if potentials.iloc[i-1] > potentials.iloc[i] < potentials.iloc[i+1]:
                tick_positions.append(time[i])
                tick_labels.append(f'{potentials.iloc[i]:.2f}')
                break


    ax1.set_xlim(0, total_time)
    
    # Create a secondary x-axis for potentials
    ax2 = ax1.twiny()
    ax2.set_xlabel('Potential (V vs. RHE)')
    ax2.set_xlim(ax1.get_xlim())

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    # Draw vertical lines at the boundaries of each cycle and where potential changes
    for tick in tick_positions:
        ax1.axvline(x=tick, color='gray', linestyle='--')

    plt.title(f'Current Density vs. Time for First Five Cycles (vs. RHE)')
    fig.tight_layout()
    plt.legend()
    plt.grid(True)

    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save the figure
    base_filename = os.path.splitext(os.path.basename(save_path))[0]
    png_filename = os.path.join(save_dir, f'{base_filename}_first_five_cycles.png')
    svg_filename = os.path.join(save_dir, f'{base_filename}_first_five_cycles.svg')

    plt.savefig(png_filename)
    plt.savefig(svg_filename)

    plt.show()

def plot_second_cycle(data, sweep_rate, voltage_increment, pH, save_path, diameter):
    conversion_factor_RHE = 0.197 + 0.059 * pH
    area = np.pi * ((diameter / 1000) / 2) ** 2  # Area of the electrode in m^2

    fig, ax1 = plt.subplots(figsize=(15, 3))
    
    cmap = get_cmap('magma')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Current Density (A/m^2)')
    ax1.tick_params(axis='y')
    
    total_time = 0  # Initialize total time for adjusting the x-axis
    tick_positions = []
    tick_labels = []
    
    # Extract the relevant columns for the second cycle (cycle = 2)
    cycle = 2
    potential_col = 2 * (cycle - 1)
    current_col = 2 * (cycle - 1) + 1

    if potential_col >= len(data.columns) or current_col >= len(data.columns):
        print(f"Cycle {cycle} is out of range. Skipping this cycle.")
        return

    potentials = data.iloc[:, potential_col] + conversion_factor_RHE
    currents = data.iloc[:, current_col]
    current_density = currents / area  # A/m^2

    time = [total_time + i * voltage_increment / sweep_rate for i in range(len(potentials))]
    total_time = time[-1] + voltage_increment / sweep_rate  # Update total time
    print(f"Cycle {cycle} ends at time: {time[-1]:.2f} seconds")
    
    color = cmap(1 / 10)  # Use a distinct color from 'magma' colormap for the second cycle
    ax1.plot(time, current_density, label=f'Cycle {cycle}', color=color)

    # Add tick positions for the beginning of the cycle
    tick_positions.append(time[0])
    tick_labels.append(f'{potentials.iloc[0]:.2f}')

    # Find the point where the potential stops decreasing and starts increasing
    for i in range(1, len(potentials) - 1):
        if potentials.iloc[i-1] > potentials.iloc[i] < potentials.iloc[i+1]:
            tick_positions.append(time[i])
            tick_labels.append(f'{potentials.iloc[i]:.2f}')
            break

    ax1.set_xlim(time[0], time[-1])
    
    # Create a secondary x-axis for potentials
    ax2 = ax1.twiny()
    ax2.set_xlabel('Potential (V vs. RHE)')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    plt.title(f'Current Density vs. Time for Second Cycle (vs. RHE)')
    fig.tight_layout()
    plt.legend()
    plt.grid(True)

    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save the figure
    base_filename = os.path.splitext(os.path.basename(save_path))[0]
    png_filename = os.path.join(save_dir, f'{base_filename}_second_cycle.png')
    svg_filename = os.path.join(save_dir, f'{base_filename}_second_cycle.svg')

    plt.savefig(png_filename)
    plt.savefig(svg_filename)

    plt.show()

# Main script
if __name__ == "__main__":
    # Dictionary mapping CSV file names to pH values
    file_to_pH = {
        'DS_00184_13.csv': 7.45,
        'DS_00185_13.csv': 6.7,
        'DS_00186_13.csv': 7.18,
        'DS_00187_13.csv': 7.16,
        'DS_00188_13.csv': 7.16,
        'DS_00189_13.csv': 6.56,
        'DS_00190_13.csv': 5.86,
        'DS_00191_13.csv': 7.8,
        'DS_00192_13.csv': 7.8,
        'DS_00197_13.csv': 7.97
    }
    
    # Directory containing the CSV files
    directory_path = '...'

    # Parameters
    sweep_rate = 0.01  # V/s
    voltage_increment = 0.01  # V
    diameter = 7  # mm

    # Loop through all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            data = load_data(file_path)

            pH = file_to_pH.get(file_name, 7.0)  # Default to 7.0 if file not in dictionary

            plot_second_cycle(data, sweep_rate, voltage_increment, pH, file_path, diameter)
