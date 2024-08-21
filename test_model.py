import pickle
from flask import Flask
from flask import request, render_template

from main import TextClassifier, clean_text

# Define hyperparameters
vocab_size = 5000  # Number of unique words in the vocabulary
embedding_dim = 128  # Size of word embeddings
max_length = 100  # Maximum length of input sequences
model = TextClassifier(vocab_size, embedding_dim, max_length)

model.load_weights('model.keras')

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Create a Flask app
app = Flask(__name__)


# create a route with a text area for user input
@app.route('/')
def home():
    return render_template('home.html')


# create a route to handle the user input and return the prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    processed_text = vectorizer.transform([clean_text(text)]).toarray()
    prediction = model.predict(processed_text)
    result = "Positive" if prediction[0] > 0.5 else "Negative"
    # return json
    return {'result': result}


# Get user input and predict sentiment
if __name__ == "__main__":
    app.run(debug=True, port=5005, host='0.0.0.0')
