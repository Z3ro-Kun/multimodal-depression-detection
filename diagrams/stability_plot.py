import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Your actual results
audio_f1 = [0.2553, 0.2553, 0.2553, 0.2553, 0.3467]  # 5 seeds
text_f1 = [0.5333, 0.7472, 0.6500, 0.8381, 0.7200]   # 5 seeds
fusion_f1 = [0.7348, 0.7348, 0.7822, 0.7200, 0.7086,  # 10 seeds
             0.7464, 0.7200, 0.7464, 0.7086, 0.7200]

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Create boxplot
data = [audio_f1, text_f1, fusion_f1]
labels = ['Audio-only\n(3×20s random)', 'Text-only\n(Hierarchical)', 'Fusion (Ours)\n(Text+Audio)']
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

bp = ax.boxplot(data, labels=labels, patch_artist=True,
                widths=0.6, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red',
                              markersize=8, markeredgecolor='darkred'),
                boxprops=dict(linewidth=2),
                whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2),
                medianprops=dict(linewidth=2, color='darkblue'))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add grid
ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_axisbelow(True)

# Labels and title
ax.set_ylabel('Macro F1 Score', fontsize=14, fontweight='bold')
ax.set_title('Model Stability Comparison Across Random Seeds',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([0.2, 0.9])

# Add statistics annotations
means = [np.mean(d) for d in data]
stds = [np.std(d) for d in data]

for i, (mean, std, color) in enumerate(zip(means, stds, colors)):
    # Mean ± std annotation
    ax.text(i+1, mean + 0.08,
            f'μ = {mean:.3f}\nσ = {std:.3f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor=color, alpha=0.3,
                     edgecolor='black', linewidth=1.5))

# Add horizontal line at fusion mean
fusion_mean = np.mean(fusion_f1)
ax.axhline(y=fusion_mean, color='#45b7d1', linestyle='--',
           linewidth=2, alpha=0.5, label=f'Fusion Mean: {fusion_mean:.3f}')

# Add variance comparison text
variance_text = (
    f'Variance Reduction:\n'
    f'• Fusion vs Text: {np.var(text_f1)/np.var(fusion_f1):.1f}× more stable\n'
    f'• Fusion vs Audio: {np.var(audio_f1)/np.var(fusion_f1):.1f}× more stable'
)

ax.text(0.02, 0.98, variance_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3,
                 edgecolor='black', linewidth=1.5))

# Legend
ax.legend(loc='lower right', fontsize=10)

# Tight layout
plt.tight_layout()

# Save in multiple formats
plt.savefig('stability_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('stability_comparison.png', dpi=300, bbox_inches='tight')

print("✓ Stability comparison saved as:")
print("  - stability_comparison.pdf (for paper)")
print("  - stability_comparison.png (for preview)")

# Print summary statistics
print("\n" + "="*50)
print("STABILITY STATISTICS")
print("="*50)
print(f"Audio-only:  F1 = {np.mean(audio_f1):.4f} ± {np.std(audio_f1):.4f}")
print(f"Text-only:   F1 = {np.mean(text_f1):.4f} ± {np.std(text_f1):.4f}")
print(f"Fusion:      F1 = {np.mean(fusion_f1):.4f} ± {np.std(fusion_f1):.4f}")
print(f"\nVariance comparison:")
print(f"  Text/Fusion:  {np.var(text_f1)/np.var(fusion_f1):.2f}× improvement")
print(f"  Audio/Fusion: {np.var(audio_f1)/np.var(fusion_f1):.2f}× improvement")

plt.show()