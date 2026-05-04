import matplotlib.pyplot as plt

epochs = range(1, 51)
actual_train_loss = [0.19814, 0.10044, 0.09574, 0.09341, 0.0919, 0.09127, 0.09056, 0.08972, 0.08933, 0.08842, 0.08815, 0.08791, 0.08748, 0.08754, 0.08687, 0.08684, 0.08672, 0.08619, 0.08603, 0.08603, 0.0855, 0.08544, 0.08532, 0.08508, 0.08474, 0.0845, 0.08434, 0.08396, 0.08386, 0.08369, 0.08361, 0.08335, 0.08324, 0.08317, 0.08293, 0.08303, 0.08296, 0.08288, 0.08259, 0.08248, 0.08242, 0.08211, 0.08219, 0.08211, 0.08234, 0.08216, 0.08186, 0.08182, 0.08171, 0.08166]
actual_val_loss = [0.10436, 0.09868, 0.09487, 0.09425, 0.09401, 0.0934, 0.09102, 0.09169, 0.08996, 0.08944, 0.08887, 0.08876, 0.0892, 0.08853, 0.08857, 0.08791, 0.0876, 0.08794, 0.08766, 0.08687, 0.08748, 0.08717, 0.08699, 0.08619, 0.08638, 0.08662, 0.08619, 0.08547, 0.08636, 0.08493, 0.08486, 0.08475, 0.08465, 0.08494, 0.08483, 0.08632, 0.08443, 0.08474, 0.08436, 0.08418, 0.08439, 0.08433, 0.08434, 0.08437, 0.08465, 0.08468, 0.08419, 0.08403, 0.08444, 0.08381]

plt.figure(figsize=(10, 6))
plt.plot(epochs, actual_train_loss, label='Training Loss', color='#1f77b4', linewidth=2, marker='o', markersize=3, alpha=0.8)
plt.plot(epochs, actual_val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2, marker='s', markersize=3, alpha=0.8)

plt.title('GCN-GRU Model: Training vs Validation Loss (50 Epochs)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True, shadow=True)

plt.annotate(f'Final Val Loss: {actual_val_loss[-1]:.4f}',
             xy=(50, actual_val_loss[-1]), xytext=(35, 0.12),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
plt.savefig('final_results_graph.png', dpi=300)
plt.show()
