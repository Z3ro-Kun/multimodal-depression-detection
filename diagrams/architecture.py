import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

# Create figure (fits IEEE 2-column: 3.5 inches wide)
fig, ax = plt.subplots(figsize=(7, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Colors
frozen_color = '#e1f5ff'
frozen_edge = '#0288d1'
trainable_color = '#fff9c4'
trainable_edge = '#f57f17'

# Helper function for boxes
def add_box(ax, x, y, width, height, text, color, edge_color, fontsize=9, bold=False):
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color,
                          edgecolor=edge_color,
                          linewidth=2.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=weight)

# Helper function for arrows
def add_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->,head_width=0.4,head_length=0.5',
                           color='black', linewidth=2)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x+0.3, mid_y, label, fontsize=8, style='italic')

# ===== INPUT LAYER =====
ax.text(5, 15.5, 'Input', ha='center', fontsize=12, fontweight='bold')
add_box(ax, 0.5, 14, 4, 1, 'Interview Transcript\n(~1,643 tokens)',
        'lightgray', 'black', 9)
add_box(ax, 5.5, 14, 4, 1, 'Interview Audio\n(16 minutes, 16kHz)',
        'lightgray', 'black', 9)

# ===== TEXT BRANCH =====
ax.text(2.5, 12.8, 'Text Branch', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=frozen_color, alpha=0.3))

add_box(ax, 0.5, 11.5, 4, 1, 'Hierarchical Chunking\n512-token segments',
        'white', 'gray', 8)
add_arrow(ax, 2.5, 14, 2.5, 12.5)

add_box(ax, 0.5, 9.5, 4, 1.5, '🔒 DistilBERT-base\n66.4M parameters\n(FROZEN)',
        frozen_color, frozen_edge, 9, True)
add_arrow(ax, 2.5, 11.5, 2.5, 11)

add_box(ax, 0.5, 7.8, 4, 1.2, '⚡ Attention Pooling\n(TRAINABLE)',
        trainable_color, trainable_edge, 8, True)
add_arrow(ax, 2.5, 9.5, 2.5, 9)

add_box(ax, 0.5, 6.3, 4, 1, 'Text Embedding\n768-dimensional',
        'white', 'gray', 9)
add_arrow(ax, 2.5, 7.8, 2.5, 7.3)

# ===== AUDIO BRANCH =====
ax.text(7.5, 12.8, 'Audio Branch', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=frozen_color, alpha=0.3))

add_box(ax, 5.5, 11.5, 4, 1, 'Random Sampling\n3 × 20-second segments',
        'white', 'gray', 8)
add_arrow(ax, 7.5, 14, 7.5, 12.5)

add_box(ax, 5.5, 9.5, 4, 1.5, '🔒 Wav2Vec2-base\n94.4M parameters\n(FROZEN)',
        frozen_color, frozen_edge, 9, True)
add_arrow(ax, 7.5, 11.5, 7.5, 11)

add_box(ax, 5.5, 7.8, 4, 1.2, 'Average Pooling\nacross segments',
        'white', 'gray', 8)
add_arrow(ax, 7.5, 9.5, 7.5, 9)

add_box(ax, 5.5, 6.3, 4, 1, 'Audio Embedding\n768-dimensional',
        'white', 'gray', 9)
add_arrow(ax, 7.5, 7.8, 7.5, 7.3)

# ===== FUSION LAYER =====
ax.text(5, 5.2, 'Fusion Module', ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=trainable_color, alpha=0.3))

add_arrow(ax, 2.5, 6.3, 3.5, 4.8, 'concat')
add_arrow(ax, 7.5, 6.3, 6.5, 4.8, 'concat')

add_box(ax, 2, 3.5, 6, 1.2, 'Concatenate: 1,536-d',
        'white', 'gray', 9)

add_box(ax, 2, 1.8, 6, 1.2, '⚡ MLP (394K params)\nLinear → LayerNorm → ReLU\n(TRAINABLE)',
        trainable_color, trainable_edge, 8, True)
add_arrow(ax, 5, 3.5, 5, 3)

# ===== OUTPUT =====
add_box(ax, 2.5, 0.2, 5, 1.2, '🎯 Classification\nNon-Depressed / Depressed',
        'lightgreen', 'darkgreen', 10, True)
add_arrow(ax, 5, 1.8, 5, 1.4)

# ===== LEGEND =====
frozen_patch = mpatches.Patch(facecolor=frozen_color, edgecolor=frozen_edge,
                              linewidth=2, label='🔒 Frozen (99.76%)')
trainable_patch = mpatches.Patch(facecolor=trainable_color, edgecolor=trainable_edge,
                                 linewidth=2, label='⚡ Trainable (0.24%)')
ax.legend(handles=[frozen_patch, trainable_patch],
         loc='upper center', bbox_to_anchor=(0.5, -0.02),
         ncol=2, fontsize=10, framealpha=0.9)

# Title
fig.suptitle('Lightweight Multimodal Fusion Architecture',
             fontsize=13, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
plt.savefig('architecture_compact.pdf', dpi=300, bbox_inches='tight')
plt.savefig('architecture_compact.png', dpi=300, bbox_inches='tight')

print("✓ Compact architecture diagram saved as:")
print("  - architecture_compact.pdf (for paper)")
print("  - architecture_compact.png (for preview)")
print("\nDimensions: 7×9 inches (fits IEEE 2-column)")

plt.show()