import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Metrics for confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_dataset(directory):
    """
    Loads text files from subdirectories.
    Returns texts and corresponding labels.
    """
    texts = []
    labels = []
    for label in os.listdir(directory):
        folder = os.path.join(directory, label)
        if os.path.isdir(folder):
            for filepath in glob.glob(os.path.join(folder, '*.txt')):
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(label)
    return texts, np.array(labels)

def main():
    train_dir = './data/train'
    test_dir = './data/test'

    X_train, y_train = load_dataset(train_dir)
    X_test, y_test = load_dataset(test_dir)

    y_train_int = np.array([int(label) for label in y_train])
    y_test_int = np.array([int(label) for label in y_test])
    num_classes = len(np.unique(y_train_int))
    print(f"Number of classes detected: {num_classes}")

    # Debug: Print label distributions for training and test sets
    unique_train, counts_train = np.unique(y_train_int, return_counts=True)
    print("Training label distribution:", dict(zip(unique_train, counts_train)))
    unique_test, counts_test = np.unique(y_test_int, return_counts=True)
    print("Test label distribution:", dict(zip(unique_test, counts_test)))

    # Shuffle training data to ensure validation split is representative
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = [X_train[i] for i in indices]
    y_train_int = y_train_int[indices]

    # Tokenize and pad sequences
    max_words = 10000  # Vocabulary size
    max_len = 500      # Maximum sequence length
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    # Build a CNN for multi-class classification with modifications to reduce overfitting
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.6),  # Increased dropout to reduce overfitting
        Dense(num_classes, activation='softmax')  # Multi-class output
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    # Train the model with a validation split
    epochs = 25
    batch_size = 8
    history = model.fit(
        X_train_pad, y_train_int,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate on test data
    loss, accuracy = model.evaluate(X_test_pad, y_test_int, verbose=1)
    print(f"Deep Learning Model Test Accuracy: {accuracy:.4f}")

    # Generate predictions on test data for confusion matrix
    test_pred_probs = model.predict(X_test_pad)
    test_pred_labels = np.argmax(test_pred_probs, axis=1)

    # Plot confusion matrix for test data
    cm = confusion_matrix(y_test_int, test_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_int))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Test Data")
    plt.show()

    # Debug: Check a few predictions on validation samples from training data
    val_split = 0.2
    split_index = int(len(X_train_pad) * (1 - val_split))
    X_val = X_train_pad[split_index:]
    y_val = y_train_int[split_index:]
    pred_probs = model.predict(X_val)
    pred_labels = np.argmax(pred_probs, axis=1)
    print("Sample validation predictions:", pred_labels[:10])
    print("Actual validation labels:", y_val[:10])

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
