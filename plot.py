import matplotlib.pyplot as plt
import numpy as np

# Set up style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

# Data
methods = ['Sensor\nOnly', 'DANN', 'CACTUS', 'Ours\nw/o Cache', 'Ours\nw/ Cache']
n_methods = len(methods)

# Speech (Jetson Orin Nano) - Blue
speech_latency = [648, 1286, 1465, 712, 673]
speech_throughput = [1.36, 0.70, 0.61, 1.27, 1.36]
speech_energy = [1944, 3859, 4395, 2135, 2018]

# IMU (Samsung Mobile) - Orange
imu_latency = [5.69, 6.76, 8.05, 193, 9.45]
imu_throughput = [144, 125, 108, 5.11, 91.9]
imu_energy = [17.1, 20.3, 24.2, 579, 28.4]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

width = 0.7
gap = 1.5  # gap between two modality groups

# X positions: IMU first (0-4), then gap, then Speech (5.5-9.5)
x_imu = np.arange(n_methods)
x_speech = np.arange(n_methods) + n_methods + gap

# Colors
color_imu = '#F5B041'
color_imu_ours = '#D35400'
color_speech = '#7FB3D5'
color_speech_ours = '#2E86AB'

def get_colors(base_color, ours_color, n):
    colors = [base_color] * n
    colors[-1] = ours_color  # Last one is "Ours w/ Cache"
    return colors

colors_imu = get_colors(color_imu, color_imu_ours, n_methods)
colors_speech = get_colors(color_speech, color_speech_ours, n_methods)

# Subplot 1: Latency
ax = axes[0]
bars_imu = ax.bar(x_imu, imu_latency, width, color=colors_imu, edgecolor='white', linewidth=0.8)
bars_speech = ax.bar(x_speech, speech_latency, width, color=colors_speech, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Latency (ms)', fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(list(x_imu) + list(x_speech))
ax.set_xticklabels(methods + methods, fontsize=7, rotation=30, ha='right')
ax.set_title('(a) Latency ↓', fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add group labels
ax.text(np.mean(x_imu), ax.get_ylim()[0] * 0.3, 'IMU (Mobile)', ha='center', fontsize=9, fontweight='bold', color='#D35400')
ax.text(np.mean(x_speech), ax.get_ylim()[0] * 0.3, 'Speech (Orin)', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')

# Subplot 2: Throughput
ax = axes[1]
bars_imu = ax.bar(x_imu, imu_throughput, width, color=colors_imu, edgecolor='white', linewidth=0.8)
bars_speech = ax.bar(x_speech, speech_throughput, width, color=colors_speech, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Throughput (smp/s)', fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(list(x_imu) + list(x_speech))
ax.set_xticklabels(methods + methods, fontsize=7, rotation=30, ha='right')
ax.set_title('(b) Throughput ↑', fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.text(np.mean(x_imu), ax.get_ylim()[0] * 0.3, 'IMU (Mobile)', ha='center', fontsize=9, fontweight='bold', color='#D35400')
ax.text(np.mean(x_speech), ax.get_ylim()[0] * 0.3, 'Speech (Orin)', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')

# Subplot 3: Energy
ax = axes[2]
bars_imu = ax.bar(x_imu, imu_energy, width, color=colors_imu, edgecolor='white', linewidth=0.8)
bars_speech = ax.bar(x_speech, speech_energy, width, color=colors_speech, edgecolor='white', linewidth=0.8)
ax.set_ylabel('Energy (mJ)', fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(list(x_imu) + list(x_speech))
ax.set_xticklabels(methods + methods, fontsize=7, rotation=30, ha='right')
ax.set_title('(c) Energy ↓', fontweight='bold', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.text(np.mean(x_imu), ax.get_ylim()[0] * 0.3, 'IMU (Mobile)', ha='center', fontsize=9, fontweight='bold', color='#D35400')
ax.text(np.mean(x_speech), ax.get_ylim()[0] * 0.3, 'Speech (Orin)', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')

plt.tight_layout()
plt.savefig('modality_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('modality_comparison.png', dpi=300, bbox_inches='tight')
plt.show()