import random
import numpy as np
import torch
from models.autoencoder_incident import TrafficAutoencoder, detect_incident_autoencoder

# Synthetic tiny autoencoder model
auto_model = TrafficAutoencoder(input_dim=2)
auto_model.eval()

def incident_from_autoencoder(queue_length, speed):
    sample = np.array([queue_length, speed], dtype=np.float32)
    incident = detect_incident_autoencoder(auto_model, sample, threshold=0.15)
    return incident


def simulate_synthetic_incident():
    """
    Returns True/False + reason string
    """
    if random.random() < 0.35:
        return True, "Random Synthetic Incident"
    return False, "No Incident"
