import sys
import numpy as np


def load_data(filename: str) -> np.ndarray:
    rows = []
    with open(filename, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append([float(v) for v in line.split()])
    return np.array(rows, dtype=float)


def loo_accuracy(data: np.ndarray, feature_indices: list) -> float:
    if not feature_indices:
        return 0.0
    labels = data[:, 0]
    features = data[:, feature_indices]
    gram = features @ features.T
    sq_norms = np.diag(gram)
    sq_dists = np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(sq_dists, np.inf)
    return float(np.sum(labels[np.argmin(sq_dists, axis=1)] == labels)) / len(data)


def fmt(features) -> str:
    return '{' + ','.join(map(str, sorted(features))) + '}'


def forward_selection(data: np.ndarray, n_features: int):
    current, best_overall, best_acc_all, prev_acc = set(), set(), 0.0, 0.0
    print("\nBeginning search.\n")
    for _ in range(n_features):
        best_feature, best_acc_lvl = None, -1.0
        for f in range(1, n_features + 1):
            if f in current:
                continue
            acc = loo_accuracy(data, sorted(current | {f}))
            print(f"\t\tUsing feature(s) {fmt(current | {f})} accuracy is {acc * 100:.1f}%")
            if acc > best_acc_lvl:
                best_acc_lvl, best_feature = acc, f
        current.add(best_feature)
        if best_acc_lvl < prev_acc:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        if best_acc_lvl > best_acc_all:
            best_acc_all, best_overall = best_acc_lvl, current.copy()
        print(f"Feature set {fmt(current)} was best, accuracy is {best_acc_lvl * 100:.1f}%\n")
        prev_acc = best_acc_lvl
    return best_overall, best_acc_all


def backward_elimination(data: np.ndarray, n_features: int):
    current = set(range(1, n_features + 1))
    prev_acc = loo_accuracy(data, sorted(current))
    best_overall, best_acc_all = current.copy(), prev_acc
    print("\nBeginning search.\n")
    for _ in range(n_features - 1):
        best_to_remove, best_acc_lvl = None, -1.0
        for f in sorted(current):
            candidate = sorted(current - {f})
            if not candidate:
                continue
            acc = loo_accuracy(data, candidate)
            print(f"\t\tUsing feature(s) {fmt(set(candidate))} accuracy is {acc * 100:.1f}%")
            if acc > best_acc_lvl:
                best_acc_lvl, best_to_remove = acc, f
        current.remove(best_to_remove)
        if best_acc_lvl < prev_acc:
            print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        if best_acc_lvl > best_acc_all:
            best_acc_all, best_overall = best_acc_lvl, current.copy()
        print(f"Feature set {fmt(current)} was best, accuracy is {best_acc_lvl * 100:.1f}%\n")
        prev_acc = best_acc_lvl
    return best_overall, best_acc_all


def main():
    print("Welcome to Bertie Woosters Feature Selection Algorithm.")
    print("\nType the number of the algorithm you want to run.\n")
    print("   1) Small dataset")
    print("   2) Large dataset\n")
    
    choice = int(input().strip())
    if choice == 1:
        filename = ".\CS170_Small_DataSet__17.txt"
    else:
        filename = ".\CS170_Large_DataSet__3.txt"
    data = load_data(filename)

    n_features, n_instances = data.shape[1] - 1, data.shape[0]

    print("\nType the number of the algorithm you want to run.\n")
    print("   1) Forward Selection")
    print("   2) Backward Elimination\n")
    choice = int(input().strip())

    print(f"\nThis dataset has {n_features} features "
          f"(not including the class attribute), with {n_instances} instances.\n")

    all_acc = loo_accuracy(data, list(range(1, n_features + 1)))
    print(f"Running nearest neighbor with all {n_features} features, "
          f'using "leaving-one-out" evaluation, I get an accuracy of {all_acc * 100:.1f}%')

    if choice == 1:
        best_features, best_acc = forward_selection(data, n_features)
    elif choice == 2:
        best_features, best_acc = backward_elimination(data, n_features)
    else:
        print("Invalid choice — please enter 1 or 2.")
        sys.exit(1)

    print(f"Finished search!! The best feature subset is {fmt(best_features)}, "
          f"which has an accuracy of {best_acc * 100:.1f}%")


if __name__ == "__main__":
    main()