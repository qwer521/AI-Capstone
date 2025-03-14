import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

# Import necessary modules from scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

def load_dataset(directory):
    """
    Loads text files from subdirectories.
    Each subdirectory name is used as the label.
    
    Parameters:
        directory (str): Path to the main directory.
        use_preprocessing (bool): Whether to remove spaces from the text.
        
    Returns:
        texts (list): List of document strings.
        labels (np.array): Array of labels (as strings).
    """
    texts = []
    labels = []
    # Loop over each subdirectory (each represents a label)
    for label in os.listdir(directory):
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            # Load all .txt files in the folder
            for filepath in glob.glob(os.path.join(folder, '*.txt')):
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(label)
    return texts, np.array(labels)

###############################################################################
# Main Function for SVM-based Classification
###############################################################################
def main():
    # Define dataset paths for training and test sets
    train_dir = './data/train'
    test_dir = './data/test'
    
    # Load training and test data
    X_train, y_train = load_dataset(train_dir)
    X_test, y_test = load_dataset(test_dir)
    
    # Display basic information about the dataset
    print(f"Loaded {len(X_train)} training documents and {len(X_test)} test documents.")
    
    # ---------------------------
    # SVM Classification using TF-IDF
    # ---------------------------
    # Set TF-IDF parameters to capture character-level n-grams (3 to 5 characters)
    tfidf_params = {
        'stop_words': None,
        'analyzer': 'char_wb',  # Use word-boundary constrained character n-grams
        'ngram_range': (3, 5)
    }
    
    # Create a pipeline that vectorizes text and applies a linear SVM
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LinearSVC(max_iter=10000))
    ])
    
    # Evaluate using 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("=== Cross-Validation on Training Data (SVM) ===")
    scores = cross_val_score(svm_pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"SVM CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Train the SVM pipeline on the full training set and evaluate on the test set
    print("\n=== Evaluation on Test Data (SVM) ===")
    svm_pipeline.fit(X_train, y_train)
    y_pred = svm_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {acc:.4f}")
    
    # Plot confusion matrix for SVM predictions
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("SVM Confusion Matrix")
    plt.show()
    
    # ---------------------------
    # SVM with PCA for Dimensionality Reduction
    # ---------------------------
    print("=== Experiment: SVM with PCA ===")
    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Convert sparse matrices to dense arrays for PCA processing
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()
    
    # Apply PCA to reduce dimensionality; n_components can be tuned based on dataset
    pca = PCA(n_components=70, random_state=42)
    X_train_pca = pca.fit_transform(X_train_dense)
    X_test_pca = pca.transform(X_test_dense)
    
    # Train a new SVM on the PCA-transformed data
    svm_pca = LinearSVC(max_iter=10000)
    svm_pca.fit(X_train_pca, y_train)
    y_pred_pca = svm_pca.predict(X_test_pca)
    acc_pca = accuracy_score(y_test, y_pred_pca)
    print(f"SVM with PCA Test Accuracy: {acc_pca:.4f}")
    
    # Plot confusion matrix for SVM with PCA predictions
    cm_pca = confusion_matrix(y_test, y_pred_pca)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Oranges)
    plt.title("SVM with PCA Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
