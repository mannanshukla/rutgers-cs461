import numpy as np

# Given CPTs
P_C = 0.5  # P(C = T)
P_S_given_C = {True: 0.1, False: 0.5}
P_R_given_C = {True: 0.8, False: 0.2}
P_W_given_S_R = {
    (True, True): 0.99,  # P(W = T | S = T, R = T)
    (True, False): 0.9,  # P(W = T | S = T, R = F)
    (False, True): 0.9,  # P(W = T | S = F, R = T)
    (False, False): 0.0  # P(W = T | S = F, R = F)
}

# Fixed evidence
S = True  # Sprinkler = T
W = True  # WetGrass = T

# Initialize variables
C = np.random.choice([True, False])  # Cloudy
R = np.random.choice([True, False])  # Rain

# Conditional Sampling Functions
def sample_C(R):
    """Sample C given R, S = T, W = T."""
    P_C_T = P_C * P_S_given_C[True] * (P_R_given_C[True] if R else (1 - P_R_given_C[True]))
    P_C_F = (1 - P_C) * P_S_given_C[False] * (P_R_given_C[False] if R else (1 - P_R_given_C[False]))
    return np.random.choice([True, False], p=[P_C_T / (P_C_T + P_C_F), P_C_F / (P_C_T + P_C_F)])

def sample_R(C):
    """Sample R given C, S = T, W = T."""
    P_R_T = P_R_given_C[C] * P_W_given_S_R[(S, True)]
    P_R_F = (1 - P_R_given_C[C]) * P_W_given_S_R[(S, False)]
    return np.random.choice([True, False], p=[P_R_T / (P_R_T + P_R_F), P_R_F / (P_R_T + P_R_F)])

# Gibbs Sampling
iterations = 10000
cloudy_samples = []

for _ in range(iterations):
    C = sample_C(R)  # Sample C given R, S = T, W = T
    R = sample_R(C)  # Sample R given C, S = T, W = T
    cloudy_samples.append(C)

# Compute approximate posterior
P_C_T_approx = sum(cloudy_samples) / len(cloudy_samples)
P_C_F_approx = 1 - P_C_T_approx

print("Gibbs Sampling Results:")
print(f"P(Cloudy = T | Sprinkler = T, WetGrass = T): {P_C_T_approx:.4f}")
print(f"P(Cloudy = F | Sprinkler = T, WetGrass = T): {P_C_F_approx:.4f}")
