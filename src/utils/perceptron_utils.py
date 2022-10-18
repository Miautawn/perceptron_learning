from math import exp

def step_activation(x: float) -> float:
    return 1.0 if x >= 0.0 else 0.0

def sigmoid_activation(x: float) -> float:
    return (1.0 / (1.0 + exp(-x)))