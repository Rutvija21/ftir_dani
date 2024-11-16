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

x_labels_1 = df.iloc[3:8, 2].values.astype(str)  # C4 to C8
hues_1 = df.iloc[2, 3:9].values.astype(str)  # D3 to I3
subplot_titles_1 = [df.iloc[0, 0], df.iloc[13, 0], df.iloc[26, 0], df.iloc[39, 0]]  # A1, A14, A27, A40

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(main_title, fontsize=16)

for i, ax in enumerate(axes.flat):
    start_row = 3 + 13 * i
    end_row = start_row + 5
    
    y_values = df.iloc[start_row:end_row, 32:38].values.astype(float)
    
    data = pd.DataFrame(y_values, columns=hues_1, index=x_labels_1)
    data = data.reset_index().melt(id_vars='index', var_name='Hues', value_name='Integrated area (a.u.)')
    data.rename(columns={'index': 'Water structure'}, inplace=True)
    
    palette = sns.color_palette("magma", n_colors=len(hues_1))
    sns.histplot(data=data, x='Water structure', weights='Integrated area (a.u.)', hue='Hues', multiple='stack', palette=palette, ax=ax, shrink=0.8)
    
    ax.set_title(subplot_titles_1[i], fontsize=14)
    ax.set_xlabel('Concentration (M)', fontsize=12)
    ax.set_ylabel('Integrated area (a.u.)', fontsize=12)
    ax.legend(labels=hues_1, title_fontsize='10', fontsize='9')

plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust to fit the main title

output_dir = os.path.dirname(file_path)
fig.savefig(os.path.join(output_dir, "diff_conc_Dani.png"))
fig.savefig(os.path.join(output_dir, "diff_conc_Dani.svg"))

plt.show()

# Second plot configuration
x_labels_2 = df.iloc[8:11, 1].values  # B9 to B11
hues_2 = df.iloc[2, 3:9].values  # D3 to I3
subplot_titles_2 = [df.iloc[0, 0], df.iloc[13, 0], df.iloc[26, 0], df.iloc[39, 0]]  # A1, A14, A27, A40

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(main_title, fontsize=16)

for i, ax in enumerate(axes.flat):
    start_row = 8 + 13 * i
    end_row = start_row + 3

    y_values = df.iloc[start_row:end_row, 32:38].values.astype(float)

    data = pd.DataFrame(y_values, columns=hues_2, index=x_labels_2)
    data = data.reset_index().melt(id_vars='index', var_name='Hues', value_name='Integrated area (a.u.)')
    data.rename(columns={'index': 'Water structure'}, inplace=True)
    
    palette = sns.color_palette("magma", n_colors=len(hues_2))
    sns.histplot(data=data, x='Water structure', weights='Integrated area (a.u.)', hue='Hues', multiple='stack', palette=palette, ax=ax, shrink=0.8)
    
    ax.set_title(subplot_titles_2[i], fontsize=14)
    ax.set_xlabel('Potential (V)', fontsize=12)
    ax.set_ylabel('Integrated area (a.u.)', fontsize=12)
    ax.legend(labels=hues_2, title_fontsize='10', fontsize='9')

plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Adjust to fit the main title

fig.savefig(os.path.join(output_dir, "diff_pot.png"))
fig.savefig(os.path.join(output_dir, "diff_pot.svg"))

plt.show()
