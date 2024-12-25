import numpy as np

# Given CPTs
P_C = 0.5  # P(Cloudy = T) = 0.5

P_S_given_C = {
    True: 0.1,   # P(S = T | C = T)
    False: 0.5   # P(S = T | C = F)
}

P_R_given_C = {
    True: 0.8,   # P(R = T | C = T)
    False: 0.2   # P(R = T | C = F)
}

P_W_given_S_R = {
    (True, True): 0.99,   # P(W = T | S = T, R = T)
    (True, False): 0.9,   # P(W = T | S = T, R = F)
    (False, True): 0.9,   # P(W = T | S = F, R = T)
    (False, False): 0.0   # P(W = T | S = F, R = F)
}

# Evidence
S = True  # Sprinkler = T
W = True  # WetGrass = T

# Step 1: Compute unnormalized probabilities for C = T and C = F
def compute_posterior_cloudy():
    # Case 1: Cloudy = T
    P_C_true_R_true = P_C * P_S_given_C[True] * P_R_given_C[True] * P_W_given_S_R[(S, True)]
    P_C_true_R_false = P_C * P_S_given_C[True] * (1 - P_R_given_C[True]) * P_W_given_S_R[(S, False)]
    P_C_true = P_C_true_R_true + P_C_true_R_false

    # Case 2: Cloudy = F
    P_C_false_R_true = (1 - P_C) * P_S_given_C[False] * P_R_given_C[False] * P_W_given_S_R[(S, True)]
    P_C_false_R_false = (1 - P_C) * P_S_given_C[False] * (1 - P_R_given_C[False]) * P_W_given_S_R[(S, False)]
    P_C_false = P_C_false_R_true + P_C_false_R_false

    # Step 2: Normalize
    total = P_C_true + P_C_false
    P_C_true_normalized = P_C_true / total
    P_C_false_normalized = P_C_false / total

    return P_C_true_normalized, P_C_false_normalized

# Compute the posterior probabilities
P_C_true, P_C_false = compute_posterior_cloudy()

print("P(Cloudy = T | Sprinkler = T, WetGrass = T):", round(P_C_true, 4))
print("P(Cloudy = F | Sprinkler = T, WetGrass = T):", round(P_C_false, 4))
