import numpy as np
import matplotlib.pyplot as plt
from ghmm import GaussianHMM

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data with 3 regimes:
    0: Low mean, low vol (Normal market)
    1: High mean, medium vol (Bull market)
    2: Low mean, high vol (Volatile/Bear market)
    """
    n_states = 3
    # Transition matrix
    A = np.array([
        [0.95, 0.03, 0.02],
        [0.05, 0.90, 0.05],
        [0.10, 0.10, 0.80]
    ])
    
    # Emission parameters
    means = np.array([[0.0], [0.5], [-0.2]])
    covs = np.array([[[0.01]], [[0.04]], [[0.1]]])
    
    states = np.zeros(n_samples, dtype=int)
    obs = np.zeros((n_samples, 1))
    
    # Initial state
    states[0] = 0
    obs[0] = np.random.normal(means[states[0]], np.sqrt(covs[states[0]]))
    
    for t in range(1, n_samples):
        states[t] = np.random.choice(n_states, p=A[states[t-1]])
        obs[t] = np.random.normal(means[states[t]], np.sqrt(covs[states[t]]))
        
    return obs, states

def test_ghmm():
    # 1. Generate data
    obs, true_states = generate_synthetic_data(1000)
    
    # 2. Initialize and fit HMM
    hmm = GaussianHMM(n_states=3, n_features=1)
    hmm.fit(obs, n_iter=50, tol=1e-4)
    
    # 3. Predict states
    pred_states = hmm.predict(obs)
    
    # 4. Evaluate (Note: states might be permuted)
    print("\nModel Parameters after fitting:")
    print("Means:\n", hmm.means)
    print("Transitions:\n", hmm.trans_mat)
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(obs, color='gray', alpha=0.5, label='Observations')
    plt.title('Observations and True Regimes')
    for i in range(3):
        mask = (true_states == i)
        plt.scatter(np.where(mask)[0], obs[mask], label=f'State {i}', s=10)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(obs, color='gray', alpha=0.5, label='Observations')
    plt.title('Predicted Regimes')
    for i in range(3):
        mask = (pred_states == i)
        plt.scatter(np.where(mask)[0], obs[mask], label=f'State {i}', s=10)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hmm_test_results.png')
    print("\nResults saved to 'hmm_test_results.png'")
    
    # Check log-likelihood history
    plt.figure()
    plt.plot(hmm.history)
    plt.title('Log-likelihood History')
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.savefig('hmm_convergence.png')
    print("Convergence plot saved to 'hmm_convergence.png'")

if __name__ == "__main__":
    test_ghmm()
