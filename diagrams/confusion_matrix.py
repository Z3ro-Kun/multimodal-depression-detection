import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Your best model confusion matrix (Seed 999)
cm = np.array([[19, 4],
               [3, 9]])

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Depressed', 'Depressed'],
            yticklabels=['Non-Depressed', 'Depressed'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, linecolor='white',
            vmin=0, vmax=20,
            ax=ax)

# Labels and title
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix - Best Fusion Model\n(Seed 999, Macro F1 = 0.782)',
             fontsize=15, fontweight='bold', pad=20)

# Add metrics as text
metrics_text = (
    f'TN = {cm[0,0]}  |  FP = {cm[0,1]}\n'
    f'FN = {cm[1,0]}  |  TP = {cm[1,1]}\n\n'
    f'Non-Depressed Recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.3f}\n'
    f'Depressed Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.3f}'
)

ax.text(0.5, -0.25, metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Tight layout
plt.tight_layout()

# Save in multiple formats
plt.savefig('confusion_matrix.pdf', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

print("✓ Confusion matrix saved as:")
print("  - confusion_matrix.pdf (for paper)")
print("  - confusion_matrix.png (for preview)")

plt.show()