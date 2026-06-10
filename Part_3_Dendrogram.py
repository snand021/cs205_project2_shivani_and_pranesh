import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

def load_and_prepare(csv_path:str, n_samples:int, seed:int):
    df = pd.read_csv(csv_path)
    
    # grab an even number of samples from each class
    df_ke = df[df["Class"] == "Kecimen"].sample(n=n_samples // 2, random_state=seed)
    df_be = df[df["Class"] == "Besni"].sample(n=n_samples // 2, random_state=seed)

    df_sample = pd.concat([df_ke, df_be]).reset_index(drop=True) # merge back into one df

    # everything except the class column is a feature
    feature_cols = [c for c in df_sample.columns if c!="Class"]
    X = StandardScaler().fit_transform(df_sample[feature_cols]) # scale so no single feature dominates

    # shorter labels for the dendrogram leaves
    labels = []

    for i, r in df_sample.iterrows():
        if r["Class"] == "Kecimen":
            labels.append(f"Ke{i:02d}")
        else:
            labels.append(f"Be{i:02d}")
    
    return X, labels, df_sample, feature_cols

def draw_dendrogram(Z, labels, ax, color_threshold):
    dendrogram(Z, labels=labels, orientation="right", leaf_font_size=6.5, color_threshold=color_threshold, above_threshold_color="lightgray", ax=ax)
    # dashed line marks where we cut the tree
    ax.axvline(x=color_threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.55, label=f"cut = {color_threshold:.2f}")

    ax.set_xlabel("Cluster Distance", fontsize=11)
    ax.legend(fontsize=9)
    
    return ax
    
def plot(Z, labels, df_sample, color_threshold, save_path):
    # map each leaf label back to its true class
    label_to_class = {lbl: df_sample.loc[i, "Class"] for i, lbl in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(10, max(8, len(labels)*0.22))) # taller fig for more leaves
    draw_dendrogram(Z, labels, ax, color_threshold)
    
    # color each leaf label by its ground-truth class (Kecimen = red, Besni = blue)
    for tick in ax.get_ymajorticklabels():
        cls = label_to_class.get(tick.get_text())
        if cls == "Kecimen":
            tick.set_color("red")
        elif cls == "Besni":
            tick.set_color("blue")
            
    ax.set_title("Figure 2: Clustering of Raisins with True-Class Annotations (color)", fontsize=13, fontweight="bold")
    # legend for the class colors
    legend_patches = [mpatches.Patch(facecolor="red", label="Kecimen"),
                       mpatches.Patch(facecolor="blue", label="Besni")]

    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
 
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    plt.close(fig)      
    
def report_observations(Z, df_sample, feature_cols):
    # cut the tree into 2 clusters and tack the labels on
    cluster_ids = fcluster(Z, t=2, criterion="maxclust")
    df_sample = df_sample.copy()
    df_sample["cluster"] = cluster_ids

    # how do the clusters line up with the real classes
    print("\nCluster composition vs. true class")
    ct = pd.crosstab(df_sample["cluster"], df_sample["Class"])
    print(ct)
 
    purity = (ct.max(axis=1).sum()) / len(df_sample)
    print(f"\nCluster purity (2 clusters): {purity:.1%}")
 
    print("\nFeature means per cluster (original scale)")
    print(df_sample.groupby("cluster")[feature_cols].mean().round(2).T)

def main():
    X, labels, df_sample, feature_cols = load_and_prepare("./Raisin_Dataset.csv", 100, 42)
    print(f"  {100} samples drawn ({100 // 2} Kecimen, {100 // 2} Besni)")
    print("  Linkage method: ward")
    Z = linkage(X, method="ward") # build the hierarchical clustering


    # cut threshold = 55% of the tallest merge, just from eyeballing the tree
    max_distance = float(Z[:, 2].max())
    color_threshold = 0.55 * max_distance

    print("\nGenerating dendrograms")
    plot(Z, labels, df_sample, color_threshold,
                 save_path="./raisin_dendrogram_colored.png")
 
    report_observations(Z, df_sample, feature_cols)
    
if __name__ == "__main__":
    main()