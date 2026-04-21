import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Parameter counts (in millions)
components = ['DistilBERT\n(Text)', 'Wav2Vec2\n(Audio)', 'Fusion\nLayer']
params = [66.4, 94.4, 0.394]  # in millions
trainable = [0, 0, 0.394]  # only fusion is trainable
frozen = [66.4, 94.4, 0]

# Colors
color_frozen = '#e1f5ff'
color_trainable = '#fff9c4'
edge_frozen = '#0288d1'
edge_trainable = '#f57f17'

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===== LEFT PLOT: Stacked Bar Chart =====
x = np.arange(len(components))
width = 0.6

bars1 = ax1.bar(x, frozen, width, label='Frozen (Pretrained)',
                color=color_frozen, edgecolor=edge_frozen, linewidth=2)
bars2 = ax1.bar(x, trainable, width, bottom=frozen, label='Trainable',
                color=color_trainable, edgecolor=edge_trainable, linewidth=2)

# Add value labels
for i, (f, t) in enumerate(zip(frozen, trainable)):
    if f > 0:
        ax1.text(i, f/2, f'{f:.1f}M\n🔒 Frozen',
                ha='center', va='center', fontsize=11, fontweight='bold')
    if t > 0:
        ax1.text(i, f + t/2, f'{t:.3f}M\n⚡ Trainable',
                ha='center', va='center', fontsize=11, fontweight='bold')

ax1.set_ylabel('Parameters (Millions)', fontsize=13, fontweight='bold')
ax1.set_title('Model Components: Parameter Breakdown', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(components, fontsize=12)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ===== RIGHT PLOT: Pie Chart =====
total_params = sum(params)
frozen_total = sum(frozen)
trainable_total = sum(trainable)

sizes = [frozen_total, trainable_total]
labels = [f'Frozen\n{frozen_total:.1f}M\n(99.76%)',
          f'Trainable\n{trainable_total:.3f}M\n(0.24%)']
colors_pie = [color_frozen, color_trainable]
explode = (0, 0.1)  # explode trainable slice

wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels,
                                     colors=colors_pie, autopct='',
                                     shadow=True, startangle=90,
                                     textprops=dict(fontsize=12, fontweight='bold'))

# Add edge colors
for w, edge in zip(wedges, [edge_frozen, edge_trainable]):
    w.set_edgecolor(edge)
    w.set_linewidth(3)

ax2.set_title('Total Model: Parameter Efficiency', fontsize=14, fontweight='bold')

# Add total in center
ax2.text(0, 0, f'Total:\n{total_params:.1f}M\nparams',
         ha='center', va='center', fontsize=13, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='black', linewidth=2))

# Add efficiency note
efficiency_text = (
    'Extreme Parameter Efficiency:\n'
    '• 99.76% of parameters frozen\n'
    '• Only 394K parameters trained\n'
    '• Minimal overfitting risk\n'
    '• Fast training on limited data'
)

fig.text(0.5, 0.02, efficiency_text,
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen',
                  alpha=0.3, edgecolor='darkgreen', linewidth=2))

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save
plt.savefig('parameter_efficiency.pdf', dpi=300, bbox_inches='tight')
plt.savefig('parameter_efficiency.png', dpi=300, bbox_inches='tight')

print("✓ Parameter efficiency visualization saved as:")
print("  - parameter_efficiency.pdf (for paper)")
print("  - parameter_efficiency.png (for preview)")

print("\n" + "="*50)
print("PARAMETER SUMMARY")
print("="*50)
print(f"Total Parameters:     {total_params:.2f}M")
print(f"Frozen Parameters:    {frozen_total:.2f}M ({frozen_total/total_params*100:.2f}%)")
print(f"Trainable Parameters: {trainable_total:.3f}M ({trainable_total/total_params*100:.2f}%)")
print(f"\nThis extreme efficiency enables:")
print("  ✓ Training on small datasets (107 samples)")
print("  ✓ Fast convergence (3-4 epochs)")
print("  ✓ Low memory footprint (8GB GPU)")
print("  ✓ CPU inference without GPU")

plt.show()