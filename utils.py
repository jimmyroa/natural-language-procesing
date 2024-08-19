import matplotlib.pyplot as plt

import re
import string

import pandas as pd
import tensorflow as tf


def merge_reviews():
    """
    Merge all positive and negative reviews into two files
    """
    categories = ['books', 'dvd', 'electronics', 'kitchen_&_housewares']
    # load all type of review in each category into one file
    pos_all = pd.DataFrame()
    neg_all = pd.DataFrame()
    unlabeled_all = pd.DataFrame()
    for category in categories:
        pos = pd.read_csv(f"./format_data/{category}/positive.review.csv")
        neg = pd.read_csv(f"./format_data/{category}/negative.review.csv")
        unlabel = pd.read_csv(f"./format_data/{category}/unlabeled.review.csv")

        pos_all = pd.concat([pos_all, pos], ignore_index=True)
        neg_all = pd.concat([neg_all, neg], ignore_index=True)
        unlabeled_all = pd.concat([unlabeled_all, unlabel], ignore_index=True)

    # save unlabeled data for later use
    unlabeled_all.to_csv("./format_data/unlabeled.review.csv", index=False)

    # label the positive and negative reviews
    pos['label'] = 1  # positive
    neg['label'] = 0  # negative

    # save the labeled data into one file

    all_reviews = pd.concat([pos, neg], ignore_index=True)

    all_reviews.to_csv("./format_data/all.review.csv", index=False)

    print("All reviews are merged and saved into one file")


def clean_text(text):
    """
    Clean the text by removing punctuation, converting to lowercase, and removing extra whitespace
    """
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def filter_short_reviews(df, min_length=5):
    """
    Filter out reviews that are shorter than the specified length
    """
    return df[df['review_text'].apply(lambda x: len(x.split()) >= min_length)]


def get_dataset(X, y, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def plot_training_history(history):
    # Get the data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()


def label_data(row):
    if row['rating'] in [4, 5]:
        return 1  # Positive
    elif row['rating'] in [1, 2]:
        return 0  # Negative
    else:
        return None  # Neutral or invalid rating (e.g., rating 3)


def get_sparse_dataset(X, y, batch_size=32, shuffle=True):
    """
    Create a Dataset from a sparse matrix X and dense vector y
    """
    dataset = tf.data.Dataset.from_tensor_slices((tf.sparse.from_dense(X.toarray()), y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
