import numpy as np

# Define the dataset
data = np.array([
    ["Sunny", "Hot", "High", "Weak", "No"],
    ["Cloudy", "Hot", "High", "Weak", "Yes"],
    ["Sunny", "Mild", "Normal", "Strong", "Yes"],
    ["Cloudy", "Mild", "High", "Strong", "Yes"],
    ["Rainy", "Mild", "High", "Strong", "No"],
    ["Rainy", "Cool", "Normal", "Strong", "No"],
    ["Rainy", "Mild", "High", "Weak", "Yes"],
    ["Sunny", "Hot", "High", "Strong", "No"],
    ["Cloudy", "Hot", "Normal", "Weak", "Yes"],
    ["Rainy", "Mild", "High", "Strong", "No"],
])

# Feature names and possible values
feature_names = ["Weather", "Temperature", "Humidity", "Wind"]
feature_values = {
    "Weather": ["Sunny", "Cloudy", "Rainy"],
    "Temperature": ["Hot", "Cool", "Mild"],
    "Humidity": ["Normal", "High"],
    "Wind": ["Weak", "Strong"]
}

# Function to calculate entropy
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))

# Function to calculate information gain
def information_gain(feature_column, target):
    # Total entropy of the target
    total_entropy = entropy(target)

    # Weighted entropy after splitting by the feature
    unique_values, counts = np.unique(feature_column, return_counts=True)
    total_samples = len(feature_column)
    weighted_entropy = 0

    for value, count in zip(unique_values, counts):
        subset_target = target[feature_column == value]
        weighted_entropy += (count / total_samples) * entropy(subset_target)

    # Information Gain
    return total_entropy - weighted_entropy

# Recursive function to build the decision tree
def build_tree(data, feature_names):
    target = data[:, -1]

    # If all target values are the same, return a leaf node
    if len(np.unique(target)) == 1:
        return f"Leaf: {target[0]}"

    # If no features are left, return the majority class
    if len(feature_names) == 0:
        values, counts = np.unique(target, return_counts=True)
        majority_class = values[np.argmax(counts)]
        return f"Leaf: {majority_class}"

    # Calculate Information Gain for each feature
    IGs = []
    for i in range(len(feature_names)):
        IG = information_gain(data[:, i], target)
        IGs.append((feature_names[i], IG))

    # Select the best feature
    best_feature_name, best_IG = max(IGs, key=lambda x: x[1])
    best_feature_index = feature_names.index(best_feature_name)

    # Split the dataset based on the best feature
    unique_values = feature_values[best_feature_name]
    tree = {best_feature_name: {}}
    remaining_features = [f for i, f in enumerate(feature_names) if i != best_feature_index]

    for value in unique_values:
        subset = data[data[:, best_feature_index] == value]
        if len(subset) > 0:
            tree[best_feature_name][value] = build_tree(subset, remaining_features)
        else:
            # Assign the majority class in the remaining subset
            majority_class = np.unique(target, return_counts=True)[0][np.argmax(np.unique(target, return_counts=True)[1])]
            tree[best_feature_name][value] = f"Leaf: {majority_class}"

    return tree

# Build the decision tree
decision_tree = build_tree(data, feature_names)

# Function to print the tree in a readable format
def print_tree(tree, depth=0):
    if isinstance(tree, str):
        print("  " * depth + tree)
    else:
        for key, value in tree.items():
            print("  " * depth + f"If {key}:")
            if isinstance(value, dict):
                for subkey, subtree in value.items():
                    print("  " * (depth + 1) + f"{subkey} ->")
                    print_tree(subtree, depth + 2)
            else:
                print_tree(value, depth + 1)

# Print the decision tree
print("Decision Tree:")
print_tree(decision_tree)

