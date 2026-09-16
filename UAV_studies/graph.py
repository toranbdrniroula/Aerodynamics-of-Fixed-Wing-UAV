import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

def setup_professional_plot_style():
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'Palatino', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['figure.titlesize'] = 20
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['grid.linewidth'] = 0.8
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['figure.figsize'] = (16, 12)
    mpl.rcParams['figure.dpi'] = 120
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.linewidth'] = 1.5
    return sns.color_palette("husl", 5)

base_dir = os.getcwd()
directories = ['baseline', 'cant90', 'cant45']
labels = ['Baseline', 'Winglet at 90°', 'Winglet at 45°']
markers = ['o', 's', '^']
all_data = {}
max_cl_cd_ratios = {}

for idx, directory in enumerate(directories):
    aoa_values = []
    cl_values = []
    cd_values = []
    dir_path = os.path.join(base_dir, directory)
    if not os.path.exists(dir_path):
        print(f"Warning: Directory '{dir_path}' not found. Skipping.")
        continue
    try:
        aoa_dirs = [d for d in os.listdir(dir_path)
                    if d.startswith('AoA_') and os.path.isdir(os.path.join(dir_path, d))]
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        continue

    for aoa_dir in aoa_dirs:
        try:
            aoa = float(aoa_dir.split('_')[1])
            data_file = None
            possible_paths = [
                os.path.join(dir_path, aoa_dir, 'postProcessing/forceCoeffs1/0/coefficient.dat'),
                os.path.join(dir_path, aoa_dir, 'postProcessing/forceCoeffs/0/coefficient.dat')
            ]
            for path in possible_paths:
                 if os.path.exists(path):
                     data_file = path
                     break
            if data_file is None:
                print(f"Warning: coefficient.dat not found in {os.path.join(directory, aoa_dir)}, skipping.")
                continue

            with open(data_file, "r") as file:
                lines = file.readlines()
            data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip() != '']

            if not data_lines:
                print(f"No valid data lines found in {data_file}, skipping.")
                continue

            last_line_content = data_lines[-1]
            cl_val, cd_val = None, None
            try:
                values = last_line_content.strip().split()
                if len(values) > 4:
                    cl_val = float(values[4])
                    cd_val = float(values[1])
                else:
                     print(f"Warning: Skipping last line in {data_file} due to insufficient columns: {last_line_content.strip()}")
                     continue
            except (ValueError, IndexError) as e:
                print(f"Error parsing last line in {data_file}: {e} - Line: {last_line_content.strip()}")
                continue
            
            if cl_val is not None and cd_val is not None:
                aoa_values.append(aoa)
                cl_values.append(cl_val)
                cd_values.append(cd_val)
            else:
                print(f"  No valid coefficient values extracted from last line of {data_file}")

        except Exception as e:
            print(f"Error processing {aoa_dir} in {directory}: {e}")

    if aoa_values:
        sorted_indices = np.argsort(aoa_values)
        aoa_values_sorted = [aoa_values[i] for i in sorted_indices]
        cl_values_sorted = [cl_values[i] for i in sorted_indices]
        cd_values_sorted = [cd_values[i] for i in sorted_indices]
        all_data[directory] = {
            'aoa': aoa_values_sorted,
            'cl': cl_values_sorted,
            'cd': cd_values_sorted
        }
        cl_cd_ratio = []
        max_ratio = 0
        for cl_val, cd_val in zip(cl_values_sorted, cd_values_sorted):
            if cd_val != 0:
                 ratio = cl_val / cd_val
                 cl_cd_ratio.append(ratio)
                 if ratio > max_ratio:
                     max_ratio = ratio
            else:
                cl_cd_ratio.append(np.nan)
                print(f"Warning: CD is zero for {directory} at AoA={aoa_values_sorted[len(cl_cd_ratio)-1]}°, cannot calculate CL/CD ratio.")
        all_data[directory]['cl_cd_ratio'] = cl_cd_ratio
        if max_ratio > 0:
            max_cl_cd_ratios[directory] = max_ratio
        else:
             print(f"  Could not determine valid max $C_L/C_D$ for {directory}")
    else:
        print(f"No valid data found in {directory}")

def customize_plot(ax, title, xlabel, ylabel, show_legend=True):
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=10)
    ax.grid(True, which="major", linestyle='--', linewidth=0.8, alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="minor", linestyle=':', linewidth=0.5, alpha=0.15)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', length=3, width=1.0)
    for spine_pos in ['top', 'right']:
        ax.spines[spine_pos].set_visible(False)
    for spine_pos in ['left', 'bottom']:
         ax.spines[spine_pos].set_linewidth(1.5)
         ax.spines[spine_pos].set_color('#444444')
    ax.set_facecolor('#f8f9fa')
    if show_legend:
        legend = ax.legend(loc='best', frameon=True)
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('lightgray')

def create_2x2_dashboard(all_data, directories, labels, markers, custom_palette, save_path='aerodynamic_plots.png'):
     fig, axs = plt.subplots(2, 2, figsize=(16, 12))
     fig.suptitle("Aerodynamic Performance Comparison of Winglet Configurations",
                  fontsize=mpl.rcParams['figure.titlesize'], fontweight='bold', y=0.98)
     plot_configs = [
         (axs[0, 0], 'aoa', 'cl', 'Lift Coefficient ($C_L$) vs Angle of Attack', 'Angle of Attack ($\\alpha$) [degrees]', 'Lift Coefficient ($C_L$)'),
         (axs[0, 1], 'aoa', 'cd', 'Drag Coefficient ($C_D$) vs Angle of Attack', 'Angle of Attack ($\\alpha$) [degrees]', 'Drag Coefficient ($C_D$)'),
         (axs[1, 0], 'cd', 'cl', 'Lift Coefficient ($C_L$) vs Drag Coefficient ($C_D$)', 'Drag Coefficient ($C_D$)', 'Lift Coefficient ($C_L$)'),
         (axs[1, 1], 'aoa', 'cl_cd_ratio', 'Lift-to-Drag Ratio ($C_L/C_D$) vs Angle of Attack', 'Angle of Attack ($\\alpha$) [degrees]', 'Lift-to-Drag Ratio ($C_L/C_D$)')
     ]
     for ax, x_key, y_key, title, xlabel, ylabel in plot_configs:
         for idx, directory in enumerate(directories):
             if directory in all_data and all_data[directory][x_key] and all_data[directory][y_key]:
                  if y_key == 'cl_cd_ratio':
                      x_vals = [all_data[directory][x_key][i] for i, y_val in enumerate(all_data[directory][y_key]) if not np.isnan(y_val)]
                      y_vals = [y_val for y_val in all_data[directory][y_key] if not np.isnan(y_val)]
                  else:
                      x_vals = all_data[directory][x_key]
                      y_vals = all_data[directory][y_key]
                  if x_vals and y_vals:
                      ax.plot(x_vals, y_vals,
                              marker=markers[idx], linestyle="--",
                              color=custom_palette[idx], label=labels[idx],
                              markersize=mpl.rcParams['lines.markersize'], linewidth=mpl.rcParams['lines.linewidth'])
                      if y_vals: # Ensure y_vals is not empty before calling max
                        ax.set_ylim(bottom=0, top=max(y_vals)*1.1 if y_vals else 1)


         customize_plot(ax, title, xlabel, ylabel)
     plt.tight_layout(rect=[0, 0, 1, 0.96])
     plt.savefig(save_path, dpi=mpl.rcParams['savefig.dpi'])
     plt.close(fig)

def create_percentage_improvement_plot(max_cl_cd_ratios, baseline_dir, labels_map, custom_palette_full, save_path='max_ld_improvement_bar_chart.png'):
    if baseline_dir not in max_cl_cd_ratios:
        print(f"Error: Baseline directory '{baseline_dir}' not found in max_cl_cd_ratios. Cannot create improvement plot.")
        return
    baseline_max_ratio = max_cl_cd_ratios[baseline_dir]
    improvement_data = {}
    print("\nCalculating percentage improvements for Max $C_L/C_D$:")
    for directory, max_ratio in max_cl_cd_ratios.items():
        if directory != baseline_dir:
            if baseline_max_ratio != 0:
                percentage_improvement = ((max_ratio - baseline_max_ratio) / baseline_max_ratio) * 100
                dir_idx = directories.index(directory) # Make sure 'directories' is accessible or passed
                improvement_data[labels[dir_idx]] = percentage_improvement # Make sure 'labels' is accessible or passed
                print(f"  {labels[dir_idx]}: {percentage_improvement:.2f}% improvement")
            else:
                 print(f"  Baseline max $C_L/C_D$ is zero, cannot calculate percentage improvement for {labels_map.get(directory, directory)}")
    if not improvement_data:
        print("No improvement data calculated. Cannot create bar chart.")
        return
    
    sorted_labels_list = sorted(improvement_data, key=improvement_data.get, reverse=False)
    sorted_improvements = [improvement_data[label] for label in sorted_labels_list]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    num_bars = len(sorted_labels_list)
    bar_palette = custom_palette_full[:num_bars] if num_bars <= len(custom_palette_full) else sns.color_palette("husl", n_colors=num_bars)

    sns.barplot(x=sorted_labels_list, y=sorted_improvements, hue=sorted_labels_list, palette=bar_palette, ax=ax, legend=False, dodge=False)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3)
    
    customize_plot(ax, "Percentage Improvement in Maximum $C_L/C_D$ Ratio",
                   "Winglet Configuration", "Percentage Improvement (%)", show_legend=False)
    plt.xticks(rotation=0, ha='center')
    plt.subplots_adjust(bottom=0.15)
    ax.set_ylim(0, max(100, max(sorted_improvements) * 1.1 if sorted_improvements else 100) ) # Adjust y_lim based on data or 100
    plt.savefig(save_path, dpi=mpl.rcParams['savefig.dpi'])
    plt.close(fig)

if __name__ == "__main__":
    custom_palette = setup_professional_plot_style()
    baseline_directory_name = 'baseline'
    
    # Main processing loop is above and populates all_data and max_cl_cd_ratios

    if all_data:
        create_2x2_dashboard(all_data, directories, labels, markers, custom_palette)
        labels_map = dict(zip(directories, labels))
        create_percentage_improvement_plot(max_cl_cd_ratios, baseline_directory_name, labels_map, custom_palette)
    else:
        print("\nNo data was successfully processed. No plots generated.")
