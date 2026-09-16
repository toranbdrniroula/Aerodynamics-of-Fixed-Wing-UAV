import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from matplotlib.ticker import AutoMinorLocator

def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    error_color = '#d62728'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'Palatino', 'DejaVu Serif']
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.titlesize'] = 18
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams['lines.markersize'] = 8
    mpl.rcParams['grid.linewidth'] = 0.8
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['figure.figsize'] = (12, 9)
    mpl.rcParams['figure.dpi'] = 120
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.linewidth'] = 1.5
    return colors, markers, error_color

def process_openfoam_data(base_dir, directories, labels, colors, markers):
    all_data = {}
    for idx, directory in enumerate(directories):
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist. Skipping.")
            continue
        aoa_values = []
        cl_values = []
        cd_values = []
        try:
            aoa_dirs = [d for d in os.listdir(dir_path)
                      if d.startswith('AoA_') and os.path.isdir(os.path.join(dir_path, d))]
            aoa_dirs.sort(key=lambda x: float(x.split('_')[1]))
        except Exception as e:
            print(f"Error accessing directory {directory}: {e}")
            continue
        
        for aoa_dir in aoa_dirs:
            try:
                aoa = float(aoa_dir.split('_')[1])
                data_file = os.path.join(dir_path, aoa_dir,
                                      'postProcessing/forceCoeffs1/0/coefficient.dat')
                if not os.path.exists(data_file):
                    data_file_alt = os.path.join(dir_path, aoa_dir,
                                         'postProcessing/forceCoeffs/0/coefficient.dat')
                    if os.path.exists(data_file_alt):
                        data_file = data_file_alt
                    else:
                         print(f"Warning: Neither {data_file} nor {data_file_alt} found, skipping {aoa_dir} in {directory}")
                         continue
                cl, cd = calculate_average_coefficients(data_file)
                if cl is not None and cd is not None:
                    aoa_values.append(aoa)
                    cl_values.append(cl)
                    cd_values.append(cd)
            except Exception as e:
                print(f"Error processing {aoa_dir} in {directory}: {e}")
        if aoa_values:
            all_data[directory] = {
                'aoa': np.array(aoa_values),
                'cl': np.array(cl_values),
                'cd': np.array(cd_values),
                'label': labels[idx],
                'color': colors[idx % len(colors)],
                'marker': markers[idx % len(markers)]
            }
        else:
            print(f"No valid data found in {directory}")
    return all_data

def calculate_average_coefficients(data_file, num_samples=100):
    try:
        with open(data_file, "r") as file:
            lines = file.readlines()
            data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip() != '']
            if not data_lines:
                print(f"No data lines found in {data_file}")
                return None, None
            num_lines_to_avg = min(num_samples, len(data_lines))
            last_lines = data_lines[-num_lines_to_avg:]
            cl_values_to_avg = []
            cd_values_to_avg = []
            for line in last_lines:
                try:
                    values = line.strip().split()
                    cd_iter = float(values[1])
                    cl_iter = float(values[4])
                    cl_values_to_avg.append(cl_iter)
                    cd_values_to_avg.append(cd_iter)
                except (ValueError, IndexError) as e:
                    continue
            if cl_values_to_avg and cd_values_to_avg:
                return float(values[4]), float(values[1])
            else:
                print(f"Could not extract valid Cl/Cd pairs from the last {num_lines_to_avg} lines of {data_file}")
                return None, None
    except FileNotFoundError:
        print(f"Error: File not found {data_file}")
        return None, None
    except Exception as e:
        print(f"Error reading file {data_file}: {e}")
        return None, None

def apply_professional_styling(ax, title, xlabel, ylabel):
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=10)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=10)
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.3, which='major')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.15, which='minor')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', length=3, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#444444')

def calculate_absolute_error(sim_data, ref_data, data_key='cl'):
    sim_aoa = sim_data['aoa']
    sim_values = sim_data[data_key]
    ref_aoa = np.array(ref_data['aoa'])
    ref_values = np.array(ref_data[data_key])
    common_angles = []
    abs_errors = []
    ref_interp = np.interp(sim_aoa, ref_aoa, ref_values, left=np.nan, right=np.nan)
    for i, aoa in enumerate(sim_aoa):
        if not np.isnan(ref_interp[i]):
            common_angles.append(aoa)
            abs_error = np.abs(sim_values[i] - ref_interp[i])
            abs_errors.append(abs_error)
    return np.array(common_angles), np.array(abs_errors)

def create_cl_alpha_plot(all_data, reference_data=None, error_color='red', save_path='cl_vs_alpha_with_error.png', show_plot=True):
    fig, ax = plt.subplots(figsize=(12, 9))
    data_lines = []
    legend_labels = []
    for directory, data in all_data.items():
        line, = ax.plot(data['aoa'], data['cl'],
                        marker=data['marker'],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor='white',
                        linestyle=":",
                        linewidth=2.5,
                        color=data['color'],
                        alpha=0.9)
        data_lines.append(line)
        legend_labels.append(data['label'])
    if reference_data:
        line, = ax.plot(reference_data['aoa'], reference_data['cl'],
                        marker="*",
                        markersize=14,
                        markeredgewidth=1.5,
                        markeredgecolor='white',
                        linestyle="--",
                        linewidth=3,
                        color='#222222',
                        alpha=0.9)
        data_lines.append(line)
        legend_labels.append(reference_data['label'])
        if all_data:
            first_sim_key = list(all_data.keys())[0]
            sim_data = all_data[first_sim_key]
            error_aoa, cl_abs_error = calculate_absolute_error(sim_data, reference_data, 'cl')
            if error_aoa.size > 0:
                line, = ax.plot(error_aoa, cl_abs_error,
                                marker='x',
                                markersize=8,
                                linestyle="-.",
                                linewidth=2.0,
                                color=error_color,
                                alpha=0.8)
                data_lines.append(line)
                legend_labels.append(f"{sim_data['label']} CL Abs Error (Sim - Ref)")
    apply_professional_styling(
        ax,
        "Lift Coefficient vs Angle of Attack (with Absolute Error)",
        "Angle of Attack (α) [degrees]",
        "Lift Coefficient (CL) / Absolute Error"
    )
    ax.set_facecolor('#f8f9fa')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_ylim(0, 1.2)
    legend = ax.legend(data_lines, legend_labels,
        loc='best',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        edgecolor='#cccccc'
    )
    legend.get_frame().set_facecolor('#ffffff')
    fig.text(0.99, 0.01, 'OpenFOAM Airfoil Analysis',
             fontsize=9, color='gray', ha='right', fontstyle='italic')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig

def create_cd_alpha_plot(all_data, reference_data=None, error_color='red', save_path='cd_vs_alpha_with_error.png', show_plot=True):
    fig, ax = plt.subplots(figsize=(12, 9))
    data_lines = []
    legend_labels = []
    for directory, data in all_data.items():
        line, = ax.plot(data['aoa'], data['cd'],
                        marker=data['marker'],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor='white',
                        linestyle=":",
                        linewidth=2.5,
                        color=data['color'],
                        alpha=0.9)
        data_lines.append(line)
        legend_labels.append(data['label'])
    if reference_data:
        line, = ax.plot(reference_data['aoa'], reference_data['cd'],
                        marker="*",
                        markersize=14,
                        markeredgewidth=1.5,
                        markeredgecolor='white',
                        linestyle="--",
                        linewidth=3,
                        color='#222222',
                        alpha=0.9)
        data_lines.append(line)
        legend_labels.append(reference_data['label'])
        if all_data:
            first_sim_key = list(all_data.keys())[0]
            sim_data = all_data[first_sim_key]
            error_aoa, cd_abs_error = calculate_absolute_error(sim_data, reference_data, 'cd')
            if error_aoa.size > 0:
                line, = ax.plot(error_aoa, cd_abs_error,
                                marker='x',
                                markersize=8,
                                linestyle="-.",
                                linewidth=2.0,
                                color=error_color,
                                alpha=0.8)
                data_lines.append(line)
                legend_labels.append(f"{sim_data['label']} CD Abs Error (Sim - Ref)")
    apply_professional_styling(
        ax,
        "Drag Coefficient vs Angle of Attack (with Absolute Error)",
        "Angle of Attack (α) [degrees]",
        "Drag Coefficient (CD) / Absolute Error"
    )
    ax.set_facecolor('#f8f9fa')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_ylim(0, 0.3)
    legend = ax.legend(data_lines, legend_labels,
        loc='best',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        edgecolor='#cccccc'
    )
    legend.get_frame().set_facecolor('#ffffff')
    fig.text(0.99, 0.01, 'OpenFOAM Airfoil Analysis',
             fontsize=9, color='gray', ha='right', fontstyle='italic')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig

def create_combined_plot(all_data, reference_data=None, error_color='red', save_path='cl_cd_vs_alpha_with_errors.png', show_plot=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=120, sharex=True)
    fig.suptitle('Aerodynamic Coefficients vs Angle of Attack (with Absolute Errors)',
                 fontsize=20, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.92, hspace=0.15)
    lines_ax1, labels_ax1 = [], []
    lines_ax2, labels_ax2 = [], []
    for directory, data in all_data.items():
        line1, = ax1.plot(data['aoa'], data['cl'],
                         marker=data['marker'], markersize=8, markeredgewidth=1.2, markeredgecolor='white',
                         linestyle=":", linewidth=2.5, color=data['color'], alpha=0.9)
        lines_ax1.append(line1)
        labels_ax1.append(data['label'])
        line2, = ax2.plot(data['aoa'], data['cd'],
                         marker=data['marker'], markersize=8, markeredgewidth=1.2, markeredgecolor='white',
                         linestyle=":", linewidth=2.5, color=data['color'], alpha=0.9)
        lines_ax2.append(line2)
        labels_ax2.append(data['label'])
    if reference_data:
        line1_ref, = ax1.plot(reference_data['aoa'], reference_data['cl'],
                             marker="*", markersize=10, markeredgewidth=1.2, markeredgecolor='white',
                             linestyle="--", linewidth=3, color='#222222', alpha=0.9)
        lines_ax1.append(line1_ref)
        labels_ax1.append(reference_data['label'])
        line2_ref, = ax2.plot(reference_data['aoa'], reference_data['cd'],
                             marker="*", markersize=10, markeredgewidth=1.2, markeredgecolor='white',
                             linestyle="--", linewidth=3, color='#222222', alpha=0.9)
        lines_ax2.append(line2_ref)
        labels_ax2.append(reference_data['label'])
        if all_data:
            first_sim_key = list(all_data.keys())[0]
            sim_data = all_data[first_sim_key]
            error_aoa_cl, cl_abs_error = calculate_absolute_error(sim_data, reference_data, 'cl')
            if error_aoa_cl.size > 0:
                line_err_cl, = ax1.plot(error_aoa_cl, cl_abs_error, marker='x', markersize=6,
                                        linestyle="-.", linewidth=2.0, color=error_color, alpha=0.8)
                lines_ax1.append(line_err_cl)
                labels_ax1.append(f"{sim_data['label']} CL Abs Error")
            error_aoa_cd, cd_abs_error = calculate_absolute_error(sim_data, reference_data, 'cd')
            if error_aoa_cd.size > 0:
                line_err_cd, = ax2.plot(error_aoa_cd, cd_abs_error, marker='x', markersize=6,
                                        linestyle="-.", linewidth=2.0, color=error_color, alpha=0.8)
                lines_ax2.append(line_err_cd)
                labels_ax2.append(f"{sim_data['label']} CD Abs Error")
    apply_professional_styling(ax1, "Lift Coefficient (CL)", "", "CL / Abs Error")
    if all_data:
        first_key_ax1 = list(all_data.keys())[0]
        ax1.set_ylim(min(np.min(all_data[first_key_ax1]['cl']), 0)-0.1, np.max(all_data[first_key_ax1]['cl'])*1.1)
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.set_facecolor('#f8f9fa')
    legend1 = ax1.legend(lines_ax1, labels_ax1, loc='best', frameon=True, fancybox=True, framealpha=0.9, shadow=True, edgecolor='#cccccc', fontsize=9)
    legend1.get_frame().set_facecolor('#ffffff')
    apply_professional_styling(ax2, "Drag Coefficient (CD)", "Angle of Attack (α) [degrees]", "CD / Abs Error")
    if all_data:
        first_key_ax2 = list(all_data.keys())[0]
        ax2.set_ylim(min(np.min(all_data[first_key_ax2]['cd']), 0)-0.02, np.max(all_data[first_key_ax2]['cd'])*1.2)
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.set_facecolor('#f8f9fa')
    legend2 = ax2.legend(lines_ax2, labels_ax2, loc='best', frameon=True, fancybox=True, framealpha=0.9, shadow=True, edgecolor='#cccccc', fontsize=9)
    legend2.get_frame().set_facecolor('#ffffff')
    fig.text(0.99, 0.01, 'OpenFOAM Airfoil Analysis',
             fontsize=9, color='gray', ha='right', fontstyle='italic')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig

def create_error_percentage_plot(all_data, reference_data, save_path='validation_error_percentage.png', show_plot=True):
    if not all_data or not reference_data:
        print("Error: Both simulation and reference data are required for percentage error plot.")
        return None
    first_sim_key = list(all_data.keys())[0]
    sim_data = all_data[first_sim_key]
    sim_aoa = sim_data['aoa']
    sim_cl = sim_data['cl']
    sim_cd = sim_data['cd']
    ref_aoa = np.array(reference_data['aoa'])
    ref_cl = np.array(reference_data['cl'])
    ref_cd = np.array(reference_data['cd'])
    common_angles = []
    cl_perc_errors = []
    cd_perc_errors = []
    ref_cl_interp = np.interp(sim_aoa, ref_aoa, ref_cl, left=np.nan, right=np.nan)
    ref_cd_interp = np.interp(sim_aoa, ref_aoa, ref_cd, left=np.nan, right=np.nan)
    for i, aoa in enumerate(sim_aoa):
        if not np.isnan(ref_cl_interp[i]) and not np.isnan(ref_cd_interp[i]) and ref_cl_interp[i] != 0 and ref_cd_interp[i] != 0:
            common_angles.append(aoa)
            cl_perc_error = abs(sim_cl[i] - ref_cl_interp[i]) / abs(ref_cl_interp[i]) * 100
            cd_perc_error = abs(sim_cd[i] - ref_cd_interp[i]) / abs(ref_cd_interp[i]) * 100
            cl_perc_errors.append(cl_perc_error)
            cd_perc_errors.append(cd_perc_error)
    if not common_angles:
        print("No common angles with valid non-zero reference data found for percentage error calculation.")
        return None
    fig, ax = plt.subplots(figsize=(12, 9))
    x = np.arange(len(common_angles))
    width = 0.35
    cl_color = '#1f77b4'
    cd_color = '#ff7f0e'
    rects1 = ax.bar(x - width/2, cl_perc_errors, width,
                   label='CL Error %',
                   color=cl_color,
                   alpha=0.8,
                   edgecolor='white',
                   linewidth=1,
                   zorder=10)
    rects2 = ax.bar(x + width/2, cd_perc_errors, width,
                   label='CD Error %',
                   color=cd_color,
                   alpha=0.8,
                   edgecolor='white',
                   linewidth=1,
                   zorder=10)
    apply_professional_styling(
        ax,
        "Percentage Error Comparison with Reference Data",
        "Angle of Attack (α) [degrees]",
        "Error Percentage (%)"
    )
    max_err = max(max(cl_perc_errors) if cl_perc_errors else 0, max(cd_perc_errors) if cd_perc_errors else 0)
    ax.set_ylim(0, min(max(100, max_err * 1.1), 200))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{aoa:.1f}°" for aoa in common_angles], fontsize=12, rotation=45, ha="right")
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9,
                            fontweight='bold',
                            color='#333333',
                            zorder=15)
    add_value_labels(rects1)
    add_value_labels(rects2)
    legend = ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        edgecolor='#cccccc'
    )
    legend.get_frame().set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig

def create_polar_plot(all_data, reference_data=None, save_path='cl_vs_cd_polar.png', show_plot=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    data_lines = []
    legend_labels = []
    for directory, data in all_data.items():
        line, = ax.plot(data['cd'], data['cl'],
                        marker=data['marker'],
                        markersize=8,
                        markeredgewidth=1.2,
                        markeredgecolor='white',
                        linestyle=":",
                        linewidth=2.0,
                        color=data['color'],
                        alpha=0.9)
        data_lines.append(line)
        legend_labels.append(data['label'])
        for i, aoa in enumerate(data['aoa']):
            if aoa % 4 == 0 or (i == 0) or (i == len(data['aoa']) - 1):
                ax.annotate(f'{aoa:.0f}°',
                           (data['cd'][i], data['cl'][i]),
                           xytext=(3, -3) if aoa % 8 == 0 else (3, 3),
                           textcoords='offset points',
                           fontsize=8,
                           color='#555555')
    if reference_data:
        line_ref, = ax.plot(reference_data['cd'], reference_data['cl'],
                        marker="*",
                        markersize=10,
                        markeredgewidth=1.2,
                        markeredgecolor='white',
                        linestyle="--",
                        linewidth=2.5,
                        color='#222222',
                        alpha=0.9)
        data_lines.append(line_ref)
        legend_labels.append(reference_data['label'])
    apply_professional_styling(
        ax,
        "Lift-Drag Polar (CL vs CD)",
        "Drag Coefficient (CD)",
        "Lift Coefficient (CL)"
    )
    all_cd_values = []
    all_cl_values = []
    for data in all_data.values():
        all_cd_values.extend(data['cd'])
        all_cl_values.extend(data['cl'])
    if reference_data:
        all_cd_values.extend(reference_data['cd'])
        all_cl_values.extend(reference_data['cl'])
    max_cd_val = max(all_cd_values) if all_cd_values else 0.1
    max_cl_val = max(all_cl_values) if all_cl_values else 1.0
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1.2)
    ax.set_facecolor('#f8f9fa')
    legend = ax.legend(data_lines, legend_labels,
        loc='lower right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        shadow=True,
        edgecolor='#cccccc',
        fontsize=9
    )
    legend.get_frame().set_facecolor('#ffffff')
    fig.text(0.99, 0.01, 'OpenFOAM UAV Analysis',
             fontsize=9, color='gray', ha='right', fontstyle='italic')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig

def create_dashboard(all_data, reference_data=None, error_color='red', save_path='UAV_analysis_dashboard.png', show_plot=True):
    if not all_data:
        print("Cannot create dashboard: No simulation data provided.")
        return None
    fig = plt.figure(figsize=(18, 14), dpi=120)
    fig.suptitle('Validation against Meftah et. al.', fontsize=22, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.30, top=0.92)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    lines_ax1, labels_ax1 = [], []
    for directory, data in all_data.items():
        l, = ax1.plot(data['aoa'], data['cl'], marker=data['marker'], markersize=8, linestyle=":", linewidth=2, color=data['color'], alpha=0.9)
        lines_ax1.append(l)
        labels_ax1.append(data['label'])
    if reference_data:
        l_ref, = ax1.plot(reference_data['aoa'], reference_data['cl'], marker="*", markersize=10, linestyle="--", linewidth=2.5, color='#222222', alpha=0.9)
        lines_ax1.append(l_ref)
        labels_ax1.append(reference_data['label'])
        first_sim_key = list(all_data.keys())[0]
        sim_data = all_data[first_sim_key]
        error_aoa_cl, cl_abs_error = calculate_absolute_error(sim_data, reference_data, 'cl')
        if error_aoa_cl.size > 0:
            l_err, = ax1.plot(error_aoa_cl, cl_abs_error, marker='x', markersize=6, linestyle="-.", linewidth=1.8, color=error_color, alpha=0.8)
            lines_ax1.append(l_err)
            labels_ax1.append(f"{sim_data['label']} CL Abs Error")
    apply_professional_styling(ax1, "Lift Coefficient vs AoA", "Angle of Attack (α) [degrees]", "CL / Abs Error")
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.set_ylim(0, 1.2)
    ax1.set_xlim(0, 18)
    ax1.legend(lines_ax1, labels_ax1, loc='best', fontsize=9, fancybox=True, shadow=True, framealpha=0.9, edgecolor='#cccccc').get_frame().set_facecolor('#ffffff')
    ax1.set_facecolor('#f8f9fa')
    lines_ax2, labels_ax2 = [], []
    for directory, data in all_data.items():
        l, = ax2.plot(data['aoa'], data['cd'], marker=data['marker'], markersize=8, linestyle=":", linewidth=2, color=data['color'], alpha=0.9)
        lines_ax2.append(l)
        labels_ax2.append(data['label'])
    if reference_data:
        l_ref, = ax2.plot(reference_data['aoa'], reference_data['cd'], marker="*", markersize=10, linestyle="--", linewidth=2.5, color='#222222', alpha=0.9)
        lines_ax2.append(l_ref)
        labels_ax2.append(reference_data['label'])
        first_sim_key = list(all_data.keys())[0]
        sim_data = all_data[first_sim_key]
        error_aoa_cd, cd_abs_error = calculate_absolute_error(sim_data, reference_data, 'cd')
        if error_aoa_cd.size > 0:
            l_err, = ax2.plot(error_aoa_cd, cd_abs_error, marker='x', markersize=6, linestyle="-.", linewidth=1.8, color=error_color, alpha=0.8)
            lines_ax2.append(l_err)
            labels_ax2.append(f"{sim_data['label']} CD Abs Error")
    apply_professional_styling(ax2, "Drag Coefficient vs AoA", "Angle of Attack (α) [degrees]", "CD / Abs Error")
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.set_ylim(0, 0.3)
    ax1.set_xlim(0, 18)
    ax2.legend(lines_ax2, labels_ax2, loc='best', fontsize=9, fancybox=True, shadow=True, framealpha=0.9, edgecolor='#cccccc').get_frame().set_facecolor('#ffffff')
    ax2.set_facecolor('#f8f9fa')
    lines_ax3, labels_ax3 = [], []
    for directory, data in all_data.items():
        l, = ax3.plot(data['cd'], data['cl'], marker=data['marker'], markersize=8, linestyle=":", linewidth=2.0, color=data['color'], alpha=0.9)
        lines_ax3.append(l)
        labels_ax3.append(data['label'])
        for i, aoa in enumerate(data['aoa']):
            if aoa % 4 == 0 or (i == 0) or (i == len(data['aoa']) - 1):
                ax3.annotate(f'{aoa:.0f}°', (data['cd'][i], data['cl'][i]), xytext=(3, 3), textcoords='offset points', fontsize=8, color='#555555')
    if reference_data:
        l_ref, = ax3.plot(reference_data['cd'], reference_data['cl'], marker="*", markersize=10, linestyle="--", linewidth=2.5, color='#222222', alpha=0.9)
        lines_ax3.append(l_ref)
        labels_ax3.append(reference_data['label'])
    apply_professional_styling(ax3, "Lift-Drag Polar", "Drag Coefficient (CD)", "Lift Coefficient (CL)")
    ax3.set_xlim(0, 0.3)
    ax3.set_ylim(0, 1.2)
    ax3.legend(lines_ax3, labels_ax3, loc='lower right', fontsize=9, fancybox=True, shadow=True, framealpha=0.9, edgecolor='#cccccc').get_frame().set_facecolor('#ffffff')
    ax3.set_facecolor('#f8f9fa')
    if reference_data:
        first_sim_key = list(all_data.keys())[0]
        sim_data = all_data[first_sim_key]
        sim_aoa = sim_data['aoa']
        sim_cl = sim_data['cl']
        sim_cd = sim_data['cd']
        ref_aoa = np.array(reference_data['aoa'])
        ref_cl = np.array(reference_data['cl'])
        ref_cd = np.array(reference_data['cd'])
        common_angles_err = []
        cl_perc_errors_err = []
        cd_perc_errors_err = []
        ref_cl_interp_err = np.interp(sim_aoa, ref_aoa, ref_cl, left=np.nan, right=np.nan)
        ref_cd_interp_err = np.interp(sim_aoa, ref_aoa, ref_cd, left=np.nan, right=np.nan)
        for i, aoa in enumerate(sim_aoa):
            if not np.isnan(ref_cl_interp_err[i]) and not np.isnan(ref_cd_interp_err[i]) and ref_cl_interp_err[i] != 0 and ref_cd_interp_err[i] != 0:
                common_angles_err.append(aoa)
                cl_perc_errors_err.append(abs(sim_cl[i] - ref_cl_interp_err[i]) / abs(ref_cl_interp_err[i]) * 100)
                cd_perc_errors_err.append(abs(sim_cd[i] - ref_cd_interp_err[i]) / abs(ref_cd_interp_err[i]) * 100)
        if common_angles_err:
            x_err = np.arange(len(common_angles_err))
            width_err = 0.35
            cl_color_err = '#1f77b4'
            cd_color_err = '#ff7f0e'
            rects1_err = ax4.bar(x_err - width_err/2, cl_perc_errors_err, width_err, label='CL Error %', color=cl_color_err, alpha=0.8, edgecolor='white', linewidth=1, zorder=10)
            rects2_err = ax4.bar(x_err + width_err/2, cd_perc_errors_err, width_err, label='CD Error %', color=cd_color_err, alpha=0.8, edgecolor='white', linewidth=1, zorder=10)
            apply_professional_styling(ax4, "Percentage Error vs Reference", "Angle of Attack (α) [degrees]", "Error Percentage (%)")
            ax4.set_xticks(x_err)
            ax4.set_xticklabels([f"{aoa:.1f}°" for aoa in common_angles_err], rotation=45, ha="right")
            max_err_dash = max(max(cl_perc_errors_err) if cl_perc_errors_err else 0, max(cd_perc_errors_err) if cd_perc_errors_err else 0)
            ax4.set_ylim(0, min(max(100, max_err_dash * 1.1), 200))
            ax4.legend(loc='upper right', fontsize=9, fancybox=True, shadow=True, framealpha=0.9, edgecolor='#cccccc').get_frame().set_facecolor('#ffffff')
        else:
             ax4.text(0.5, 0.5, 'No common data for % error analysis', ha='center', va='center', fontsize=12, color='#666666')
             apply_professional_styling(ax4, "Percentage Error vs Reference", "Angle of Attack (α) [degrees]", "Error Percentage (%)")
             ax4.set_xticks([])
             ax4.set_yticks([])
        def add_value_labels(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax4.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width()/2, height),
                                xytext=(0, 4),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=9,
                                fontweight='bold',
                                color='#333333',
                                zorder=15)
        add_value_labels(rects1_err)
        add_value_labels(rects2_err)
    else:
        ax4.text(0.5, 0.5, 'Reference data needed for % error', ha='center', va='center', fontsize=12, color='#666666')
        apply_professional_styling(ax4, "Percentage Error vs Reference", "Angle of Attack (α) [degrees]", "Error Percentage (%)")
        ax4.set_xticks([])
        ax4.set_yticks([])
    ax4.set_facecolor('#f8f9fa')
    if reference_data and all_data and 'cl_perc_errors_err' in locals() and cl_perc_errors_err:
        first_dir = list(all_data.keys())[0]
        sim_data_summary = all_data[first_dir]
        cl_avg_error = np.mean(cl_perc_errors_err)
        cd_avg_error = np.mean(cd_perc_errors_err)
        valid_ld = (sim_data_summary['cd'] != 0) & (~np.isnan(sim_data_summary['cl'])) & (~np.isnan(sim_data_summary['cd']))
        if np.any(valid_ld):
            ld_ratios = sim_data_summary['cl'][valid_ld] / sim_data_summary['cd'][valid_ld]
            max_ld_idx = np.argmax(ld_ratios)
            max_ld = ld_ratios[max_ld_idx]
            max_ld_aoa = sim_data_summary['aoa'][valid_ld][max_ld_idx]
        else:
            max_ld = np.nan
            max_ld_aoa = np.nan
        summary_text = (
            f"Summary ({sim_data_summary['label']}):\n"
            f"• Max L/D: {max_ld:.2f} at α = {max_ld_aoa:.1f}°\n"
            f"• Avg. CL Error: {cl_avg_error:.1f}%\n"
            f"• Avg. CD Error: {cd_avg_error:.1f}%"
        )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig

def main():
    colors, markers, error_color = setup_plot_style()
    base_dir = os.getcwd()
    directories = ['baseline']
    labels = ['current study']
    reference_data = {
        'aoa': [3, 8, 13, 17],
        'cl': [0.4511, 0.8000, 1.0496, 0.9532],
        'cd': [0.0495, 0.0747, 0.1498, 0.2753],
        'label': "Data from Meftah et al."
    }
    num_dirs = len(directories)
    plot_colors = [colors[i % len(colors)] for i in range(num_dirs)]
    plot_markers = [markers[i % len(markers)] for i in range(num_dirs)]
    all_data = process_openfoam_data(base_dir, directories, labels, plot_colors, plot_markers)
    if not all_data:
        print("No data was processed. Exiting. Check directories and file paths (e.g., 'sa/AoA_*/postProcessing/forceCoeffs1/0/coefficient.dat').")
        return
    create_cl_alpha_plot(all_data, reference_data, error_color=error_color, save_path='cl_vs_alpha_with_error.png', show_plot=False)
    create_cd_alpha_plot(all_data, reference_data, error_color=error_color, save_path='cd_vs_alpha_with_error.png', show_plot=False)
    create_polar_plot(all_data, reference_data, save_path='cl_vs_cd_polar.png', show_plot=False)
    create_combined_plot(all_data, reference_data, error_color=error_color, save_path='cl_cd_vs_alpha_with_errors.png', show_plot=False)
    if reference_data:
        create_error_percentage_plot(all_data, reference_data, save_path='validation_error_percentage.png', show_plot=False)
    else:
        print("Skipping percentage error plot: Reference data not provided.")
    create_dashboard(all_data, reference_data, error_color=error_color, save_path='validation.png', show_plot=False)

if __name__ == "__main__":
    main()
