import matplotlib.pyplot as plt

features = [
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "ConvexArea",
    "Extent",
    "Perimeter"
]

accuracies = [73.8, 80.3, 62.8, 61.8, 76.2, 49.8, 79.8] #accuracies in order for each of the features

plt.figure(figsize=(9, 4))
plt.bar(features, accuracies)
plt.ylabel("Accuracy (%)")
plt.title("Single-Feature Nearest Neighbor Accuracy")
plt.xticks(rotation=35, ha="right")
plt.ylim(40, 85)

#add accuracy value above each bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.7, f"{acc:.1f}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("single_feature_accuracy.png", dpi=200)
plt.show()