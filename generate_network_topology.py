import numpy as np

def generate_network_topology(num_nodes):
    """
    Generates a random network topology.

    Parameters:
    - num_nodes: int, the number of nodes in the network

    Returns:
    - A numpy array representing the adjacency matrix of the network
    """
    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Populate the adjacency matrix with random connections
    # Ensuring that the matrix is symmetric and has no self-loops
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            connection = np.random.randint(0, 2)  # Randomly create a connection
            adjacency_matrix[i, j] = connection
            adjacency_matrix[j, i] = connection  # Mirror the connection

    return adjacency_matrix

# Example usage
num_nodes = 5  # Number of nodes in the network
network_topology = generate_network_topology(num_nodes)
print("Generated Network Topology:\n", network_topology)
