
'''
This script works with a specific csv file (Deconvolution water peak (integrated areas at 999 seconds).csv)
It plots rows 9-11, 22-24, etc

This plot has two sections, one for different concentrations and another for different potentials
'''



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.use('SVG')
mpl.rcParams['svg.fonttype'] = 'none'  # Do not convert fonts to paths

file_path = "..."


df = pd.read_csv(file_path, header=None)

main_title = df.iloc[0, 0]
x_labels = df.iloc[2, 3:9].values # extract x labels from D3 to I3 (index 2, columns 3 to 8)
subplot_titles = [df.iloc[0, 0], df.iloc[13, 0], df.iloc[26, 0], df.iloc[39, 0]] # extract subplot titles from A1, A14, A27, and A40

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(main_title, fontsize=16)

for i, ax in enumerate(axes.flat):
    start_row = 3 + 13 * i
    end_row = start_row + 4
    
    hues = df.iloc[start_row:end_row, 2].values
    y_values = df.iloc[start_row:end_row, 14:20].values.astype(float)
    
    data = pd.DataFrame(y_values, columns=x_labels, index=hues)
    data = data.reset_index().melt(id_vars='index', var_name='Water structure', value_name='Integrated area (a.u.)')
    data.rename(columns={'index': 'Hues'}, inplace=True)
    
    palette = sns.color_palette("magma", n_colors=len(hues))
    sns.barplot(x='Water structure', y='Integrated area (a.u.)', hue='Hues', data=data, palette=palette, ax=ax)
    
    ax.set_title(subplot_titles[i], fontsize=14)
    ax.set_xlabel('Water structure', fontsize=12)
    ax.set_ylabel('Integrated area (a.u.)', fontsize=12)
    ax.legend(title='Concentration (M)', title_fontsize='10', fontsize='9')

plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust to fit the main title

output_dir = os.path.dirname(file_path)
fig.savefig(os.path.join(output_dir, "diff_conc_Dani.png"))
fig.savefig(os.path.join(output_dir, "diff_conc_Dani.svg"))

plt.show()



main_title = df.iloc[0, 0]

x_labels = df.iloc[2, 3:9].values
subplot_titles = [df.iloc[0, 0], df.iloc[13, 0], df.iloc[26, 0], df.iloc[39, 0]]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(main_title, fontsize=16)

for i, ax in enumerate(axes.flat):
    start_row = 8 + 13 * i
    end_row = start_row + 3

    hues = df.iloc[start_row:end_row, 1].astype(str).values
    y_values = df.iloc[start_row:end_row, 14:20].values.astype(float)

    data = pd.DataFrame(y_values, columns=x_labels, index=hues)
    data = data.reset_index().melt(id_vars='index', var_name='Water structure', value_name='Integrated area (a.u.)')
    data.rename(columns={'index': 'Hues'}, inplace=True)
    
    palette = sns.color_palette("magma", n_colors=len(hues))
    sns.barplot(x='Water structure', y='Integrated area (a.u.)', hue='Hues', data=data, palette=palette, ax=ax)
    
    ax.set_title(subplot_titles[i], fontsize=14)
    ax.set_xlabel('Water structure', fontsize=12)
    ax.set_ylabel('Integrated area (a.u.)', fontsize=12)
    ax.legend(title='Classification', title_fontsize='10', fontsize='9')

plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust to fit the main title

fig.savefig(os.path.join(output_dir, "diff_pot.png"))
fig.savefig(os.path.join(output_dir, "diff_pot.svg"))

plt.show()
