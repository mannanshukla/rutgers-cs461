
import numpy as np
from train import calculate_statistics

def gaussian_pdf(x, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

def classify(glucose, bp):
    mu_g_pos, sigma2_g_pos, mu_g_neg, sigma2_g_neg, mu_b_pos, sigma2_b_pos, mu_b_neg, sigma2_b_neg, P_D_pos, P_D_neg = calculate_statistics()

    likelihood_g_pos = gaussian_pdf(glucose, mu_g_pos, sigma2_g_pos)
    likelihood_b_pos = gaussian_pdf(bp, mu_b_pos, sigma2_b_pos)
    posterior_pos = likelihood_g_pos * likelihood_b_pos * P_D_pos

    likelihood_g_neg = gaussian_pdf(glucose, mu_g_neg, sigma2_g_neg)
    likelihood_b_neg = gaussian_pdf(bp, mu_b_neg, sigma2_b_neg)
    posterior_neg = likelihood_g_neg * likelihood_b_neg * P_D_neg

    # Normalize the posteriors
    denominator = posterior_pos + posterior_neg
    P_pos_given_data = posterior_pos / denominator
    P_neg_given_data = posterior_neg / denominator

    # Return 1 if positive, 0 if negative
    return 1 if P_pos_given_data > P_neg_given_data else 0
