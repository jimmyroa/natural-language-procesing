import pickle
import re
import string

import pandas as pd
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Model, layers


# Model definition using embeddings and LSTM
class TextClassifier(Model):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(TextClassifier, self).__init__()
        # Embedding layer to convert words into dense vectors
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)

        # LSTM layer to capture sequential dependencies in the text
        self.lstm = layers.LSTM(128, return_sequences=False)

        # Fully connected layer
        self.fc1 = layers.Dense(64, activation='relu')

        # Dropout layer to prevent overfitting
        self.dropout = layers.Dropout(0.5)

        # Output layer for binary classification
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Forward pass through the network
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.output_layer(x)

    def build(self, input_shape):
        super(TextClassifier, self).build(input_shape)


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


# Main function
def main():
    # merge all reviews
    # merge_reviews()

    # load the dataset
    dataset = pd.read_csv("./all.review.csv")

    # Randomly shuffle the dataset to ensure that the data is not ordered
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Apply the cleaning function to the review_text column
    dataset['cleaned_review_text'] = dataset['review_text'].apply(clean_text)

    # Conduct outlier removal to eliminate really short or wrong reviews.
    dataset = filter_short_reviews(dataset)

    # Pad/truncate remaining data
    max_length = 100
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(dataset['cleaned_review_text'])
    X = np.array(X.toarray())
    if X.shape[1] > max_length:
        X = X[:, :max_length]
    elif X.shape[1] < max_length:
        X = np.pad(X, ((0, 0), (0, max_length - X.shape[1])), mode='constant')

    # Save the vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Split the data into training, validation and test sets

    X_train, X_test, y_train, y_test = train_test_split(X, dataset['label'], test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create datasets for training, validation, and testing
    batch_size = 32
    train_dataset = get_dataset(X_train, y_train, batch_size=batch_size)
    val_dataset = get_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_dataset = get_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Define hyperparameters
    vocab_size = 5000  # Number of unique words in the vocabulary
    embedding_dim = 128  # Size of word embeddings
    max_length = 100  # Maximum length of input sequences

    # Instantiate the model
    model = TextClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, max_length=max_length)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print model summary
    model.build((None, max_length))  # Specifying the input shape for the model summary
    model.summary()

    # Train the model
    epochs = 10

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint('model.keras', save_best_only=True)

    # Train the model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        callbacks=[early_stopping, checkpoint])

    # Evaluate the model
    model.load_weights('model.keras')
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Call the function to plot the training history
    plot_training_history(history)


if __name__ == "__main__":
    main()
