import random
from ml.incident_detector import (
    incident_from_autoencoder,
    simulate_synthetic_incident
)

def detect_incident(queue_length, speed):
    auto_incident = incident_from_autoencoder(queue_length, speed)
    rand_incident, reason = simulate_synthetic_incident()

    if auto_incident or rand_incident:
        return True, "Incident Detected"
    return False, "Normal Traffic"


def apply_incident_to_graph(G, incident_detected):
    if not incident_detected:
        return None

    edges = list(G.edges())
    u, v = random.choice(edges)

    G[u][v]["dynamic_weight"] = float("inf")
    G[u][v]["incident"] = True

    return (u, v)
