import csv
from classify import classify

def test_accuracy():
    test_data = []
    with open('test.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            glucose, bp, actual_outcome = float(row[1]), float(row[2]), int(row[3])
            test_data.append((glucose, bp, actual_outcome))

    correct_predictions = 0
    for glucose, bp, actual_outcome in test_data:
        predicted_outcome = classify(glucose, bp)
        if predicted_outcome == actual_outcome:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    return accuracy

accuracy = 100 * test_accuracy()
print(f'The accuracy of this classification is {accuracy}%')
