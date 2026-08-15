import os
import numpy as np
import matplotlib.pyplot as plt

# Apply consistent styling
def thesis_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   '#FAFBFC',
        'axes.edgecolor':   '#DEE2E6',
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'grid.color':       '#CED4DA',
        'font.size':        11,
        'axes.titlesize':   14,
        'axes.labelsize':   12,
        'legend.fontsize':  11,
        'figure.dpi':       120,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',
    })

thesis_style()

# Data
metric_names = ['mAP@50', 'mAP@50-95', 'Recall']
v8_vals       = [55.54, 28.98, 50.82]
v11_base_vals = [56.82, 30.64, 52.37]

C_V8 = '#2F80ED'        # Blue
C_V11_BASE = '#EB5757'  # Red

x = np.arange(len(metric_names))
w = 0.32  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_axisbelow(True) # Put grid lines behind bars

b1 = ax.bar(x - w/2, v8_vals,       w, label='YOLOv8s Baseline', color=C_V8,       edgecolor='white', linewidth=1.2)
b2 = ax.bar(x + w/2, v11_base_vals, w, label='YOLO11s Baseline', color=C_V11_BASE, edgecolor='white', linewidth=1.2)

# Add exact value labels on top of the bars
for bars in [b1, b2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.0,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11, fontweight='semibold')
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_ylim(0, 60)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))

ax.set_title('YOLOv8s vs. YOLO11s Baseline Model Comparison', fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()

# Save to destination folders
workspace_path = r'c:\Users\ihsan\Documents\GitHub\ML2\model_comparison_results\yolov8s_vs_yolo11s_baseline.png'
artifact_dir = r'C:\Users\ihsan\.gemini\antigravity-ide\brain\cdf00dd8-b356-4de2-b40f-daae9200960c'
artifact_path = os.path.join(artifact_dir, 'yolov8s_vs_yolo11s_baseline.png')

# Ensure directories exist
os.makedirs(os.path.dirname(workspace_path), exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

plt.savefig(workspace_path, dpi=300)
plt.savefig(artifact_path, dpi=300)
plt.close()

print(f"Saved workspace plot to: {workspace_path}")
print(f"Saved artifact plot to: {artifact_path}")
