import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
num_buckets = 5

def mse_buckets(df, num_buckets):
    df_sorted = df.sort_values('fico_score')
    bucket_size = len(df) // num_buckets
    
    boundaries = []
    for i in range(1, num_buckets):
        lower_bound = df_sorted.iloc[(i - 1) * bucket_size]['fico_score']
        upper_bound = df_sorted.iloc[i * bucket_size]['fico_score']
        boundaries.append((lower_bound + upper_bound) / 2)
    
    return boundaries

def log_likelihood(df_slice):
    k_i = sum(df_slice['default'])
    n_i = len(df_slice)
    
    if n_i == 0:
        return 0
    
    p_i = k_i / n_i
    
    if p_i == 0:
        return (n_i - k_i) * np.log(1 - p_i)
    elif p_i == 1:
        return k_i * np.log(p_i)
    else:
        return k_i * np.log(p_i) + (n_i - k_i) * np.log(1 - p_i)

# Dynamic Programming to find boundaries maximizing Log-likelihood
def find_dp_boundaries(df, num_buckets):
    df = df.sort_values('fico_score').reset_index(drop=True)
    n = len(df)
    
    dp = np.zeros((n+1, num_buckets+1))
    dp[:, :] = -np.inf
    dp[0, :] = 0
    
    prev = np.zeros((n+1, num_buckets+1), dtype=int)
    
    for j in range(1, num_buckets + 1):
        for i in range(1, n + 1):
            for k in range(i):
                cur_ll = log_likelihood(df.iloc[k:i])
                if dp[k, j-1] + cur_ll > dp[i, j]:
                    dp[i, j] = dp[k, j-1] + cur_ll
                    prev[i, j] = k

    boundaries = []
    i, j = n, num_buckets
    while j > 0 and i > 0:
        k = prev[i, j]
        boundaries.append(df.iloc[k]['fico_score'])
        i, j = k, j-1

    return sorted(boundaries)

# Example usage
mse_boundaries = mse_buckets(df, num_buckets)
print(f"MSE boundaries: {mse_boundaries}")
# log likelihood not working, stuck in infinite loop
log_likelihood_boundaries = find_dp_boundaries(df, num_buckets)
print(f"Log-likelihood boundaries: {log_likelihood_boundaries}")
