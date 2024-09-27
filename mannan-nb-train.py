
import numpy as np
import csv

def calculate_statistics():
    data = []
    with open('train.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            glucose, bp, outcome = float(row[1]), float(row[2]), int(row[3])
            data.append((glucose, bp, outcome))

    pos_data = [d for d in data if d[2] == 1]
    neg_data = [d for d in data if d[2] == 0]

    glucose_pos = [d[0] for d in pos_data]
    bp_pos = [d[1] for d in pos_data]
    glucose_neg = [d[0] for d in neg_data]
    bp_neg = [d[1] for d in neg_data]

    mu_g_pos, sigma2_g_pos = np.mean(glucose_pos), np.var(glucose_pos)
    mu_g_neg, sigma2_g_neg = np.mean(glucose_neg), np.var(glucose_neg)
    mu_b_pos, sigma2_b_pos = np.mean(bp_pos), np.var(bp_pos)
    mu_b_neg, sigma2_b_neg = np.mean(bp_neg), np.var(bp_neg)

    P_D_pos = len(pos_data) / len(data)  # Prior for diabetes-positive
    P_D_neg = len(neg_data) / len(data)  # Prior for diabetes-negative

    return mu_g_pos, sigma2_g_pos, mu_g_neg, sigma2_g_neg, mu_b_pos, sigma2_b_pos, mu_b_neg, sigma2_b_neg, P_D_pos, P_D_neg
