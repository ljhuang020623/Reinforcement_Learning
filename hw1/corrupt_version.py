from pp1starter import prepFrozen
from VI_PI_MPI import value_iteration, policy_evaluation
import numpy as np
import matplotlib.pyplot as plt
import copy

def corrupt_transition(P, alpha):
    P_noisy = copy.deepcopy(P)
    for s in range(len(P_noisy)):
        for a in range(len(P_noisy[s])):
            outcomes = P_noisy[s][a]
            p_orig = []
            nonzero_indices = []

            for i, outcome in enumerate(outcomes):
                prob, s_, R, done = outcome
                if prob > 0:
                    nonzero_indices.append(i)
                    p_orig.append(prob)
            if len(nonzero_indices) == 0:
                continue
            q = np.random.dirichlet(np.ones(len(p_orig)))
            p_orig = np.array(p_orig)
            p_new = alpha * q + (1 - alpha) * np.array(p_orig)
            
            for idx, i in enumerate(nonzero_indices):
                prob, s_, R, done = outcomes[i]
                outcomes[i] = (p_new[idx], s_, R, done)
    return P_noisy

def run_experiment(P, env, nS, nA, discount, tolerance, alphas, num_experiment=10):
    # store results in dictionary
    results = {alpha: [] for alpha in alphas}
    for alpha in alphas:
        print(f"Alpha = {alpha:.1f}")
        for exp in range(num_experiment):
            P_noisy = corrupt_transition(P, alpha)
            noisy_policy, _, _, _ = value_iteration(P_noisy, nS, nA, discount, tolerance, max_iter=500, env = env)
            eval_score = policy_evaluation(env, noisy_policy, discount, episodes = 500)
            results[alpha].append(eval_score)
            print(f" Experiment {exp + 1} : Score = {eval_score:.4f}")
    return results




if __name__ == '__main__':
    env, P, nS, nA, dname = prepFrozen()  
    tolerance = 0.001
    discount = 1.0 - 1E-3 
    alphas = np.arange(0.0, 0.8, 0.1)
    results = run_experiment(P, env, nS, nA, discount, tolerance, alphas)

    # Compute means and standard deviations:
    alpha_means = []
    alpha_stds = []
    for alpha in alphas:
        scores = np.array(results[alpha])
        alpha_means.append(np.mean(scores))
        alpha_stds.append(np.std(scores))

    plt.figure(figsize=(8, 6))
    plt.errorbar(alphas, alpha_means, yerr=alpha_stds, fmt='o-', capsize=5)
    plt.xlabel('Noise Level α')
    plt.ylabel('Mean Discounted Return')
    plt.title('Performance vs. Noise Level in the Transition Model')
    plt.grid(True)
    plt.show()