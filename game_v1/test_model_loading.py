#!/usr/bin/env python3
"""Quick test to verify model loads correctly"""
import os
import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

# Load model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "dqn_cartpole_gymnasium.pth")
print(f"Looking for model at: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    model = DQN(4, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Test with a sample state
    test_state = torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)
    with torch.no_grad():
        q_values = model(test_state)
        action = torch.argmax(q_values).item()
    
    print(f"Q-values for balanced state: {q_values[0].numpy()}")
    print(f"Recommended action: {action} ({'LEFT' if action == 0 else 'RIGHT'})")
    
    # Test with tilted state
    test_state2 = torch.FloatTensor([0.0, 0.0, 0.1, 0.0]).unsqueeze(0)  # Tilted right
    with torch.no_grad():
        q_values2 = model(test_state2)
        action2 = torch.argmax(q_values2).item()
    
    print(f"\nQ-values for right-tilted pole: {q_values2[0].numpy()}")
    print(f"Recommended action: {action2} ({'LEFT' if action2 == 0 else 'RIGHT'})")
else:
    print("✗ Model file not found!")
