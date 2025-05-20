import numpy as np
from scipy.special import gammaln, digamma

def kl_div_dirichlet(alpha_1, alpha_2): return gammaln(np.sum(alpha_1,axis=1)) - np.sum(gammaln(alpha_1),axis=1) - gammaln(np.sum(alpha_2, axis=1)) + np.sum(gammaln(alpha_2),axis=1) + np.sum((alpha_1-alpha_2)*(digamma(alpha_1)-digamma(np.sum(alpha_1, axis=1))[:,np.newaxis]),axis=1)
def sym_kl_div_dirichlet(alpha_1, alpha_2): return 0.5*(kl_div_dirichlet(alpha_1, alpha_2) + kl_div_dirichlet(alpha_2, alpha_1))


def bhattacharyya_distance_dirichlet(gamma1, gamma2):
    gamma1 = np.array(gamma1)
    gamma2 = np.array(gamma2)
    norm1 = np.sum(gamma1, axis=1)
    norm2 = np.sum(gamma2, axis=1) 
    term1 = gammaln(0.5 * (norm1 + norm2))
    term2 = 0.5 * np.sum(gammaln(gamma1) + gammaln(gamma2), axis=1)
    term3 = np.sum(gammaln(0.5 * (gamma1 + gamma2)),axis=1)
    term4 = 0.5 * (gammaln(norm1) + gammaln(norm2))
    distance = term1 + term2 - term3 - term4
    return distance


def calculate_quantile_loss(targets, quantile_predictions, quantiles, nonzero=False):
    quantiles_losses = np.array([np.maximum(q * (targets - quantile_predictions[i]), (q - 1) * (targets - quantile_predictions[i])) for i, q in enumerate(quantiles)])
    if nonzero: 
        all_zero_idx = np.all(np.isnan(quantiles_losses) | (quantiles_losses == 0), axis=0)
        quantiles_losses[:, all_zero_idx] = np.nan
    quantiles_losses = np.nanmean(quantiles_losses, axis=-1)
    total_loss = np.nanmean(quantiles_losses)
    return total_loss, quantiles_losses

def calculate_interval_score(targets, quantile_predictions, quantiles, weighted=False):
    quantiles = np.array(quantiles)
    iqrs, scores = [], []

    upper_quantiles = quantiles[quantiles >= 0.5]
    lower_quantiles = quantiles[quantiles <= 0.5]

    upper_quantile_predictions = quantile_predictions[quantiles >= 0.5]
    lower_quantile_predictions = quantile_predictions[quantiles <= 0.5]

    for i, q_lower in enumerate(lower_quantiles):
        upper_idx = np.where(upper_quantiles+q_lower==1)[0]
        q_upper = upper_quantiles[upper_idx[0]]
        if len(upper_idx) == 0: continue
        penalties_lower = np.maximum(0, lower_quantile_predictions[i] - targets)
        penalties_upper = np.maximum(0, targets - upper_quantile_predictions[upper_idx[0]])
        iqrs.append((q_upper - q_lower))
        scores.append((upper_quantile_predictions[upper_idx[0]] - lower_quantile_predictions[i]) + (2 / (1-iqrs[-1])) * penalties_lower + (2 / (1-iqrs[-1])) * penalties_upper)
        if weighted: scores[-1] *= 1-iqrs[-1]
    
    scores = np.array(scores)
    scores = np.nanmean(scores, axis=0)
    score = np.nanmean(scores)

    return score, scores, iqrs

def calculate_mean_absolute_deviation(targets, quantile_predictions, quantiles):
    mean_absolute_deviations = np.array([np.nanmean(np.abs(targets - quantile_predictions[i]), axis=-1) for i in range(len(quantiles))])

    score = np.nanmean(mean_absolute_deviations)

    return score, mean_absolute_deviations