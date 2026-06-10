import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Raisin_Dataset.csv")

plt.figure(figsize=(6, 5))

for cls in ["Kecimen", "Besni"]: #plot each raisin class seperately 
    subset = df[df["Class"] == cls]
    plt.scatter(
        subset["MajorAxisLength"],
        subset["Perimeter"],
        alpha=0.6,
        label=cls
    )

plt.xlabel("MajorAxisLength")
plt.ylabel("Perimeter")
plt.title("Raisin Classes by MajorAxisLength and Perimeter")
plt.legend()
plt.tight_layout() #clean layout
plt.savefig("scatter_majoraxis_perimeter.png", dpi=200)
plt.show()