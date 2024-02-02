import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Synthetic data generation (simplified for demonstration)
def generate_data(num_samples=1000, num_nodes=5):
    """
    Generate synthetic training data for ANN.
    Each sample represents a network state and a target optimal route (simplified).
    """
    X = np.random.randint(0, 2, (num_samples, num_nodes * num_nodes))  # Random network states
    y = np.random.randint(0, 2, (num_samples, num_nodes))  # Random "optimal" routes
    return X, y

X, y = generate_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the ANN model
model = Sequential([
    Dense(64, activation='relu', input_dim=X.shape[1]),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

# Save the model
model.save('ann_hanet_model.h5')
