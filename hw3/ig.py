import numpy as np
from collections import Counter

# Data: Weather, Temperature, Humidity, Wind, Play
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Cloudy', 'Hot', 'High', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Cloudy', 'Mild', 'High', 'Strong', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Strong', 'No'],
    ['Rainy', 'Cool', 'Normal', 'Strong', 'No'],
    ['Rainy', 'Mild', 'High', 'Weak', 'Yes'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Cloudy', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rainy', 'Mild', 'High', 'Strong', 'No']
]

# Convert to NumPy array for easier processing
data = np.array(data)
features = ['Weather', 'Temperature', 'Humidity', 'Wind']
target = data[:, -1]  # Play column

# Calculate entropy
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

# Information gain calculation
def information_gain(data, feature_index, target):
    total_entropy = entropy(target)
    values, counts = np.unique(data[:, feature_index], return_counts=True)
    conditional_entropy = 0.0
    for value, count in zip(values, counts):
        subset = target[data[:, feature_index] == value]
        conditional_entropy += (count / len(target)) * entropy(subset)
    return total_entropy - conditional_entropy

# Calculate information gain for each feature
gains = {}
for i, feature in enumerate(features):
    gains[feature] = information_gain(data, i, target)

# Sort features by information gain
sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)

# Output the results
print("Information Gain for each feature:")
for feature, gain in sorted_gains:
    print(f"{feature}: {gain:.4f}")

# Select the feature with the highest gain
best_feature = sorted_gains[0][0]
print(f"\nBest feature to split: {best_feature}")

