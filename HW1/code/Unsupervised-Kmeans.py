import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

def load_texts_and_labels(directory: str):
    """
    Loads .txt files from subdirectories of the given directory.
    The subdirectory name is used as the label for each .txt file.
    
    :param directory: Path to the directory containing labeled subfolders.
    :return: (list of document strings, numpy array of integer labels)
    """
    texts, labels = [], []
    for label in os.listdir(directory):
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            for filepath in glob.glob(os.path.join(folder, '*.txt')):
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    texts.append(content)
                    # Assumes folder names are valid integers (e.g., "0", "1", "2", ...)
                    labels.append(int(label))
    return texts, np.array(labels)

###############################################################################
# KMeans Clustering Analysis with One-to-One Mapping
###############################################################################
def analyze_kmeans_clusters(documents, true_labels, n_clusters=None, top_n=5):
    """
    Performs TF-IDF vectorization, runs KMeans clustering, and prints detailed analysis including:
      - Cluster distribution.
      - Top TF-IDF features per cluster.
      - Representative document snippet (closest to centroid).
      - One-to-one mapping of predicted clusters to true labels via Hungarian algorithm.
      - Overall cluster purity (accuracy).
      - Confusion matrix of mapped predictions vs. true labels.
    
    :param documents: List of document strings.
    :param true_labels: Ground-truth labels as a numpy array of integers.
    :param n_clusters: Number of clusters for KMeans. If None, set to number of unique true labels.
    :param top_n: Number of top TF-IDF features to display per cluster.
    """
    # If n_clusters not provided, use number of unique true labels
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))
        
    # TF-IDF vectorization using character-level n-grams
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(X)
    
    print(f"\n=== KMeans with k={n_clusters} ===")
    
    # Print cluster distribution
    for c in range(n_clusters):
        count = np.sum(pred_labels == c)
        print(f"Cluster {c}: {count} documents")
    
    # Top TF-IDF features for each cluster
    centroids = kmeans.cluster_centers_
    print(f"\nTop {top_n} TF-IDF features per cluster:")
    for c in range(n_clusters):
        centroid = centroids[c]
        top_indices = np.argsort(centroid)[::-1][:top_n]
        top_features = feature_names[top_indices]
        print(f"  Cluster {c}: " + ", ".join(top_features))
    
    # Representative Documents: find document closest to each centroid
    print("\nRepresentative documents for each cluster (closest to centroid):")
    distances = euclidean_distances(X, centroids)
    for c in range(n_clusters):
        cluster_mask = (pred_labels == c)
        if not np.any(cluster_mask):
            print(f"  Cluster {c} has no documents!")
            continue
        cluster_distances = distances[cluster_mask, c]
        closest_local_idx = np.argmin(cluster_distances)
        global_idx = np.where(cluster_mask)[0][closest_local_idx]
        snippet = documents[global_idx][:200].replace("\n", " ")
        print(f"  Cluster {c}: doc index={global_idx}, snippet:")
        print("    ", snippet, "...\n")
    
    # Create the contingency table (confusion matrix between clusters and true labels)
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Use Hungarian algorithm to find one-to-one mapping between predicted clusters and true labels
    # We want to maximize correct assignments; convert to cost matrix by taking negative
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    # Ensure all clusters are mapped; if a cluster wasn't assigned, map it arbitrarily (should not happen if numbers match)
    for c in range(n_clusters):
        if c not in mapping:
            mapping[c] = -1  # assign a dummy label if missing

    print("\nOne-to-one mapping (predicted cluster -> true label):")
    for c in range(n_clusters):
        print(f"  Cluster {c} is mapped to true label: {mapping[c]}")
    
    # Map each predicted cluster to its corresponding true label using the mapping
    mapped_pred = np.array([mapping[c] for c in pred_labels])
    
    # Compute overall purity (accuracy)
    overall_accuracy = np.sum(mapped_pred == true_labels) / len(true_labels)
    print(f"\nOverall cluster purity (accuracy): {overall_accuracy:.4f}")
    
    # Compute and display confusion matrix of true labels vs. mapped predictions
    cm_mapped = confusion_matrix(true_labels, mapped_pred)
    all_labels = sorted(set(true_labels) | set(mapped_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_mapped, display_labels=all_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (True Labels vs. Mapped Cluster Labels)")
    plt.show()

###############################################################################
# Main Function
###############################################################################
def main():
    train_dir = "./data/train"
    test_dir = "./data/test"
    
    # Load documents and true labels from both training and test directories
    train_docs, train_labels = load_texts_and_labels(train_dir)
    test_docs, test_labels = load_texts_and_labels(test_dir)
    
    # Combine both datasets
    all_docs = train_docs + test_docs
    all_true_labels = np.concatenate((train_labels, test_labels))
    
    print(f"Loaded {len(all_docs)} documents from both train and test sets.")
    print("True label distribution:", dict(zip(*np.unique(all_true_labels, return_counts=True))))
    
    # Analyze KMeans clustering with one-to-one mapping and confusion matrix
    analyze_kmeans_clusters(all_docs, all_true_labels, n_clusters=None, top_n=5)

if __name__ == "__main__":
    main()
