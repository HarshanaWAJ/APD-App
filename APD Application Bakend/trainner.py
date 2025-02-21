import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset (data.pickle)
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels from the pickle file
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode the labels (e.g., converting strings to integers)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# Define hyperparameters to tune using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],  # Depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Minimum samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for best split
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV with Cross-Validation (5 folds)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Print the best hyperparameters found by GridSearchCV
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-validation Score: {grid_search.best_score_:.4f}")

# Use the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(x_test)

# Evaluate the model's accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Final Test Accuracy: {accuracy * 100:.2f}%')

# Save the fine-tuned model to a file
with open('fine_tuned_model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)

# Optionally, save the label encoder to decode the labels back later
with open('label_encoder.p', 'wb') as f:
    pickle.dump({'label_encoder': label_encoder}, f)
