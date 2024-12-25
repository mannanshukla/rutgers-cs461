import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Given parameters of the RBM (from the figure in the problem)
# W is a 3x3 matrix: rows = visible units (m1, m2, m3), columns = hidden units (h1, h2, h3)
W = np.array([[-1,  1,  0],
              [ 0,  1,  2],
              [-1, -2,  3]])

# Visible biases (b) and hidden biases (c)
b = np.array([-1, -1, -2])  # biases for m1, m2, m3
c = np.array([-2,  1,  1])  # biases for h1, h2, h3

# Known user preferences: liked m1 (+1) and disliked m3 (-1)
# We don't know m2, so we temporarily set it to 0, just to run computations.
v_known = np.array([+1, 0, -1])

# Step 1: Compute hidden probabilities given the known visible units
# Using bipolar {+1,-1}, we just use the linear energy term W^T v + c
hidden_input = c + W.T @ v_known
h_prob = sigmoid(hidden_input)

# Step 2: Deterministic hidden state: h_j = +1 if P(h_j=+1)>0.5 else -1
h = np.where(h_prob > 0.5, 1, -1)

# Step 3: Now reconstruct m2 given h
# Compute the visible probabilities: P(m = +1|h) = sigmoid(W h + b)
visible_input = b + W @ h
v_prob = sigmoid(visible_input)

# The probability for m2 (index 1) being +1:
m2_probability = v_prob[1]

# Decide if user likes m2:
m2_preference = 1 if m2_probability > 0.5 else -1

print("Probability that user likes m2:", m2_probability)
print("Predicted preference for m2 (+1=like, -1=dislike):", m2_preference)

