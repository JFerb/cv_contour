from matplotlib import pyplot as plt
import numpy as np

canny_stats = {
    "precision_avg": 0.39362,
    "recall_avg": 0.42644,
    "f1_avg": 0.36234,
    "precision_best": 0.51534,
    "recall_best": 0.50434,
    "f1_best": 0.44884
}

mlp_stats = {
    "precision_avg": 0.30985,
    "recall_avg": 0.98355,
    "f1_avg": 0.45733,
    "precision_best": 0.39646,
    "recall_best": 1.0,
    "f1_best": 0.55002,
}

unet_stats = {
    "precision_avg": 0.61753,
    "recall_avg": 0.69363,
    "f1_avg": 0.63834,
    "precision_best": 0.72631,
    "recall_best": 0.76380,
    "f1_best": 0.70667
}

# All 6 stats and their display names
stats = [
    ("precision_avg", "Precision (avg)"),
    ("recall_avg", "Recall (avg)"),
    ("f1_avg", "F1 (avg)"),
    ("precision_best", "Precision (best)"),
    ("recall_best", "Recall (best)"),
    ("f1_best", "F1 (best)"),
]

algorithms = ["Canny", "MLP", "UNet"]
x = np.arange(len(algorithms))

# Metric groups: (avg_key, best_key) with fill color
metric_groups = [
    ("precision_avg", "precision_best", "precision"),
    ("recall_avg", "recall_best", "recall"),
    ("f1_avg", "f1_best", "f1"),
]
fill_colors = ["#2ecc71", "#3498db", "#e74c3c"]  # green, blue, red

plt.figure(figsize=(8, 5))

# Fill between avg and best for each metric
for (avg_key, best_key, label), color in zip(metric_groups, fill_colors):
    avg_vals = [canny_stats[avg_key], mlp_stats[avg_key], unet_stats[avg_key]]
    best_vals = [canny_stats[best_key], mlp_stats[best_key], unet_stats[best_key]]
    plt.fill_between(x, avg_vals, best_vals, alpha=0.3, color=color, label=f"{label.title()} (avg ↔ opt)")

# Plot the 6 lines on top (same colors as fills)
stat_colors = {
    "precision_avg": fill_colors[0], "precision_best": fill_colors[0],
    "recall_avg": fill_colors[1], "recall_best": fill_colors[1],
    "f1_avg": fill_colors[2], "f1_best": fill_colors[2],
}
for stat_key, stat_label in stats:
    values = [canny_stats[stat_key], mlp_stats[stat_key], unet_stats[stat_key]]
    marker = "o" if "avg" in stat_key else "x"
    plt.plot(x, values, marker=marker, markersize=8, color=stat_colors[stat_key])

plt.xticks(x, algorithms)
plt.ylim(0, 1)
plt.xlabel("Algorithmus")
plt.ylabel("Wert")
plt.title("Alle Statistiken von Canny, MLP und UNet")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("all_statistics_comparison.png")
plt.close()