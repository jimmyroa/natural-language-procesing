import pickle

import pandas as pd
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from model import TextClassifier
from utils import merge_reviews, clean_text, filter_short_reviews, get_dataset, plot_training_history


def main():
    # merge all reviews
    merge_reviews()

    # load the dataset
    dataset = pd.read_csv("./format_data/all.review.csv")

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
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    # Train the model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        callbacks=[early_stopping, checkpoint])

    # Evaluate the model
    model.load_weights('best_model.keras')
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Call the function to plot the training history
    plot_training_history(history)

    # Save the model
    model.save('text_classifier.keras')
    print("Model saved successfully")


if __name__ == "__main__":
    main()
