import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    data = data_dict['data']
    labels = data_dict['labels']

# Encode labels (assuming they are strings)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Convert data to numpy arrays
import numpy as np
X_train = np.array(X_train)
X_test = np.array(X_test)

# Reshape input data to have a single channel for 1D convolutions
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the CNN model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(42, 1)),  # First convolutional layer
    MaxPooling1D(2),  # Max pooling layer to reduce dimensionality
    Conv1D(128, 3, activation='relu'),  # Second convolutional layer
    MaxPooling1D(2),  # Max pooling layer
    Flatten(),  # Flatten the output to feed into fully connected layers
    Dropout(0.5),  # Dropout for regularization
    Dense(128, activation='relu'),  # Fully connected layer
    Dense(len(set(labels_encoded)), activation='softmax')  # Output layer (number of classes)
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")