import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
import pickle

from model import TextClassifier
from utils import clean_text, label_data, plot_training_history


def main():
    # Load the unlabeled CSV file
    unlabeled_dataset = pd.read_csv('./format_data/unlabeled.review.csv')

    # Apply the cleaning function to the review_text column
    unlabeled_dataset['cleaned_review_text'] = unlabeled_dataset['review_text'].apply(clean_text)

    # Apply the labeling function
    unlabeled_dataset['label'] = unlabeled_dataset.apply(label_data, axis=1)

    # Drop rows with neutral/invalid ratings
    labeled_dataset = unlabeled_dataset.dropna(subset=['label'])

    # Load the saved vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Vectorize the text data
    X_new = vectorizer.transform(labeled_dataset['cleaned_review_text']).toarray()

    # Pad/truncate data to max_length (100)
    max_length = 100
    if X_new.shape[1] > max_length:
        X_new = X_new[:, :max_length]
    elif X_new.shape[1] < max_length:
        X_new = np.pad(X_new, ((0, 0), (0, max_length - X_new.shape[1])), mode='constant')

    # Split the data into training and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_new, labeled_dataset['label'], test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Load the saved model
    max_length = 100
    model = TextClassifier(max_length=max_length)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Load the best model weights
    model.load_weights('best_model.keras')

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    # Continue training the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=1,
                        callbacks=[early_stopping, checkpoint])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Call the function to plot the training history
    plot_training_history(history)

    # Save the model
    model.save('text_classifier.keras')
    print("Model saved successfully")


if __name__ == "__main__":
    main()
