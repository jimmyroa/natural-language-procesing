from keras import Model, layers


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
