from keras import Model, layers


class TextClassifier(Model):
    def __init__(self, max_length):
        super(TextClassifier, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')  # Input layer
        self.dropout = layers.Dropout(0.5)  # Dropout layer to prevent overfitting
        self.fc2 = layers.Dense(64, activation='relu')  # Hidden layer
        self.output_layer = layers.Dense(1, activation='sigmoid')  # Output layer for binary classification

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.output_layer(x)

    def build(self, input_shape):
        super(TextClassifier, self).build(input_shape)
