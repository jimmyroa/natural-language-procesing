import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. Load and Combine Data
positive_reviews = pd.read_csv('https://github.com/jimmyroa/natural-language-procesing/blob/main/format_data/books/positive.review.csv?raw=true', names=['review_text'])
negative_reviews = pd.read_csv('https://github.com/jimmyroa/natural-language-procesing/blob/main/format_data/books/negative.review.csv?raw=true', names=['review_text'])

positive_reviews['sentiment'] = 1
negative_reviews['sentiment'] = 0

data = pd.concat([positive_reviews, negative_reviews], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Clean the data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

data['review_text'] = data['review_text'].apply(clean_text)

# 3. Remove short reviews
data = data[data['review_text'].apply(lambda x: len(x.split()) > 2)]

# 4. Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['review_text'])
sequences = tokenizer.texts_to_sequences(data['review_text'])
max_sequence_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# 5. Split the data
X = padded_sequences
y = data['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Define the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Train the model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 8. Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# 9. Function to predict sentiment
def predict_sentiment(review_text):
    cleaned_text = clean_text(review_text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(padded)
    return "Positive" if prediction[0][0] > 0.5 else "Negative"

# Example usage
sample_review = "This book was fantastic! I loved every page."
print(f"Sample review: {sample_review}")
print(f"Predicted sentiment: {predict_sentiment(sample_review)}")