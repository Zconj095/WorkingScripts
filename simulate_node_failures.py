import numpy as np
from generate_network_topology import network_topology
def simulate_node_failures(network_topology, num_failures=1):
    """
    Simulates node failures in the network.

    Parameters:
    - network_topology: numpy array, the adjacency matrix representing the network
    - num_failures: int, the number of nodes to fail

    Returns:
    - A tuple containing the modified network topology and the list of failed nodes
    """
    num_nodes = network_topology.shape[0]
    failed_nodes = np.random.choice(num_nodes, size=num_failures, replace=False)
    
    # Disable the failed nodes by setting their connections to 0
    for node in failed_nodes:
        network_topology[node, :] = 0
        network_topology[:, node] = 0

    return network_topology, failed_nodes

# Example usage with the previously generated topology
num_failures = 1  # Number of nodes to fail
modified_topology, failed_nodes = simulate_node_failures(network_topology, num_failures)
print("Modified Network Topology After Node Failure:\n", modified_topology)
print("Failed Nodes:", failed_nodes)
