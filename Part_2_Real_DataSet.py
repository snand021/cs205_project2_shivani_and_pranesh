import pandas as pd
from sklearn.preprocessing import StandardScaler
from Part_1_Nearest_Neighbour import forward_selection, backward_elimination, loo_accuracy

def fmt(features) -> str:
    return '{' + ','.join(map(str, sorted(features))) + '}'

def main():
    df = pd.read_csv(r"C:\Users\shiva\cs205_project2_shivani_and_pranesh\Raisin_Dataset.csv")

    #remove accidential spaces, then convert class labels to 1 and 2
    df["Class"] = df["Class"].astype(str).str.strip()
    df["Class"] = df["Class"].map({"Kecimen": 1, "Besni": 2})

    #check if mapping failed
    if df["Class"].isna().any():
        print("Error: Some class labels did not map correctly.")
        print(df["Class"].unique())
        return

    #moove Class to first column
    cols = ["Class"] + [c for c in df.columns if c != "Class"]
    df = df[cols]

    #normalize only the feature columns, not Class
    feature_cols = df.columns[1:]
    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols].astype(float))

    data = df.to_numpy(dtype=float)
    n_features = data.shape[1] - 1
    n_instances = data.shape[0]

    print(f"\nThis dataset has {n_features} features"
          f"(not including the class attribute), with {n_instances} instances.\n")

    all_acc = loo_accuracy(data, list(range(1, n_features + 1)))
    print(f'Running nearest neighbor with all {n_features} features, '
          f'using "leaving-one-out" evaluation, I get an accuracy of {all_acc * 100:.1f}%')

    print("\nType the number of the algorithm you want to run.\n")
    print("   1) Forward Selection")
    print("   2) Backward Elimination\n")

    choice = int(input().strip())

    if choice == 1:
        best_features, best_acc = forward_selection(data, n_features)
    elif choice == 2:
        best_features, best_acc = backward_elimination(data, n_features)
    else:
        print("Invalid choice — please enter 1 or 2.")
        return

    print(f"Finished search!The best feature subset is{fmt(best_features)},"
          f"which has an accuracy of {best_acc * 100:.1f}%")

if __name__ == "__main__":
    main()