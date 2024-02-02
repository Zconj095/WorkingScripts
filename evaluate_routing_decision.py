import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('ann_hanet_model.h5')

# Function to simulate a new network condition and evaluate routing decision
def evaluate_routing_decision(num_nodes=5):
    """
    Simulate a new network condition, use the ANN model to predict the route,
    and evaluate its decision (simplified evaluation).
    """
    # Simulate a new network condition
    new_network_state = np.random.randint(0, 2, (1, num_nodes * num_nodes))
    
    # Predict the optimal route using the ANN model
    predicted_route = model.predict(new_network_state)
    
    # For demonstration, evaluate the route based on a simplistic criterion
    # Here we assume the more 1s in predicted_route, the more efficient it is considered
    efficiency_score = np.sum(predicted_route)
    print("Predicted route:", predicted_route)
    print("Efficiency score:", efficiency_score)

# Evaluate a routing decision
evaluate_routing_decision()
