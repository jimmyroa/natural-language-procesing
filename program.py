import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from tqdm import tqdm
from flask import Flask, request, render_template

# Load the dataset
url = "http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html"
data = pd.read_csv(url, header=None, names=['review', 'sentiment'])

# Clean the data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

data['review'] = data['review'].apply(clean_text)

# Encode the words in the review
vocab = Counter()
for review in data['review']:
    vocab.update(review.split())
vocab = sorted(vocab, key=vocab.get, reverse=True)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

def encode_review(review):
    return [word_to_idx[word] for word in review.split()]

data['encoded_review'] = data['review'].apply(encode_review)

# Encode the labels for 'positive' and 'negative'
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Conduct outlier removal
min_length = 5
data = data[data['encoded_review'].apply(len) >= min_length]

# Pad/truncate remaining data
max_length = 200
def pad_or_truncate(encoded_review):
    if len(encoded_review) < max_length:
        return encoded_review + [0] * (max_length - len(encoded_review))
    else:
        return encoded_review[:max_length]

data['padded_review'] = data['encoded_review'].apply(pad_or_truncate)

# Split the data into training, validation and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Create a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.reviews = torch.tensor(data['padded_review'].tolist())
        self.labels = torch.tensor(data['sentiment'].tolist())

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

# Obtain batches of training data using DataLoaders
train_dataset = SentimentDataset(train_data)
val_dataset = SentimentDataset(val_data)
test_dataset = SentimentDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the network architecture
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        output = self.fc(hidden)
        return output

# Define the model class
class SentimentModel:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, device):
        self.device = device
        self.model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = 0
            train_acc = 0
            self.model.train()
            for reviews, labels in tqdm(train_loader):
                reviews, labels = reviews.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(reviews)
                loss = self.criterion(output, labels.unsqueeze(1).float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_acc += (output.sigmoid() > 0.5).eq(labels.unsqueeze(1)).sum().item() / labels.size(0)
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            val_loss = 0
            val_acc = 0
            self.model.eval()
            with torch.no_grad():
                for reviews, labels in val_loader:
                    reviews, labels = reviews.to(self.device), labels.to(self.device)
                    output = self.model(reviews)
                    loss = self.criterion(output, labels.unsqueeze(1).float())
                    val_loss += loss.item()
                    val_acc += (output.sigmoid() > 0.5).eq(labels.unsqueeze(1)).sum().item() / labels.size(0)
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    def test(self, test_loader):
        test_acc = 0
        self.model.eval()
        with torch.no_grad():
            for reviews, labels in test_loader:
                reviews, labels = reviews.to(self.device), labels.to(self.device)
                output = self.model(reviews)
                test_acc += (output.sigmoid() > 0.5).eq(labels.unsqueeze(1)).sum().item() / labels.size(0)
        test_acc /= len(test_loader)
        print(f'Test Accuracy: {test_acc:.4f}')

    def predict(self, review):
        self.model.eval()
        with torch.no_grad():
            encoded_review = torch.tensor([encode_review(clean_text(review))]).to(self.device)
            encoded_review = pad_or_truncate(encoded_review[0])
            encoded_review = encoded_review.unsqueeze(0)
            output = self.model(encoded_review)
            return 'Positive' if output.sigmoid() > 0.5 else 'Negative'

# Instantiate the network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentModel(len(vocab), 128, 256, 1, device)

# Train the model
model.train(train_loader, val_loader, 10)

# Test the model
model.test(test_loader)

# Create a simple web page
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = model.predict(review)
        return render_template('index.html', sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)