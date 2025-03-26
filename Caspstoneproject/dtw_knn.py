import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Load Raw Time Series (X, Y, Z axes)

def load_inertial_signals(signal_type, dataset_type, base_path):
    file_path = os.path.join(base_path, dataset_type, "Inertial Signals", f"{signal_type}_{dataset_type}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" Missing file: {file_path}")
    return np.loadtxt(file_path)

def load_labels(dataset_type, base_path):
    label_file = os.path.join(base_path, dataset_type, f"y_{dataset_type}.txt")
    return pd.read_csv(label_file, header=None).values.flatten()


# DTW Implementation

def dtw_distance(ts_a, ts_b):
    M, N = len(ts_a), len(ts_b)
    dtw_matrix = np.full((M + 1, N + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = np.linalg.norm(ts_a[i - 1] - ts_b[j - 1])  # multivariate DTW
            last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[M, N]

def dtw_knn(X_train, y_train, X_test, k=3):
    predictions = []
    for test_sample in X_test:
        distances = [(dtw_distance(test_sample, train_sample), y_train[i]) for i, train_sample in enumerate(X_train)]
        k_nearest = sorted(distances, key=lambda x: x[0])[:k]
        labels = [label for _, label in k_nearest]
        predictions.append(Counter(labels).most_common(1)[0][0])
    return predictions


# Combine X, Y, Z signals into multivariate time series

def stack_axes(dataset_type, base_path, num_samples):
    x = load_inertial_signals("body_acc_x", dataset_type, base_path)[:num_samples]
    y = load_inertial_signals("body_acc_y", dataset_type, base_path)[:num_samples]
    z = load_inertial_signals("body_acc_z", dataset_type, base_path)[:num_samples]

    stacked = []
    for i in range(num_samples):
        # Shape: (128 timesteps, 3 axes)
        sample = np.stack([x[i], y[i], z[i]], axis=1)
        stacked.append(sample)
    return np.array(stacked)


# Main Execution

if __name__ == "__main__":
    base_path = os.path.join("human_activity_data", "UCI HAR Dataset")

    # Load 3-axis multivariate data
    X_train = stack_axes("train", base_path, 500)
    y_train = load_labels("train", base_path)[:500]
    X_test = stack_axes("test", base_path, 100)
    y_test = load_labels("test", base_path)[:100]

    print(" Loaded 3-axis multivariate signals (500 train / 100 test)")

    # DTW-KNN 
    print("\n🔄 Running DTW-KNN (this may take a moment)...")
    dtw_preds = dtw_knn(X_train, y_train, X_test, k=3)
    print("\n📊 DTW-KNN Accuracy:", accuracy_score(y_test, dtw_preds))
    print(classification_report(y_test, dtw_preds))

    # Euclidean KNN 
    print("\n⚡ Flattening data for Euclidean KNN...")
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train_flat, y_train)
    knn_preds = knn.predict(X_test_flat)
    print("\n📊 Euclidean KNN Accuracy:", accuracy_score(y_test, knn_preds))
    print(classification_report(y_test, knn_preds))

    # Visualizations
    dtw_cm = confusion_matrix(y_test, dtw_preds)
    knn_cm = confusion_matrix(y_test, knn_preds)

    # DTW Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(dtw_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("DTW-KNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Euclidean Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Euclidean KNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # Accuracy Comparison
    plt.figure(figsize=(5, 4))
    models = ['DTW-KNN', 'Euclidean KNN']
    accuracies = [accuracy_score(y_test, dtw_preds), accuracy_score(y_test, knn_preds)]
    sns.barplot(x=models, y=accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
