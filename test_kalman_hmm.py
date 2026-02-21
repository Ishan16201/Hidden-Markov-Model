import numpy as np
import matplotlib.pyplot as plt
from kalman_hmm import KalmanHMM

def generate_switching_data(n_samples=500):
    """
    Generate data from two regimes:
    Regime 0: Constant velocity (Trending)
    Regime 1: Oscillating (Mean-reverting)
    """
    n_regimes = 2
    state_dim = 2
    obs_dim = 1
    
    # Transition matrix
    A = np.array([[0.98, 0.02],
                  [0.05, 0.95]])
    
    # Dynamics
    F0 = np.array([[1.0, 1.0], [0.0, 1.0]]) # Trending
    F1 = np.array([[0.8, -0.5], [0.5, 0.8]]) # Oscillating
    
    H = np.array([[1.0, 0.0]])
    
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.5
    
    states = np.zeros(n_samples, dtype=int)
    x = np.zeros((n_samples, 2))
    z = np.zeros((n_samples, 1))
    
    curr_regime = 0
    curr_x = np.array([[0.0], [1.0]])
    
    for t in range(n_samples):
        states[t] = curr_regime
        x[t] = curr_x.flatten()
        z[t] = np.dot(H, curr_x) + np.random.normal(0, np.sqrt(R[0,0]))
        
        # Switch regime
        curr_regime = np.random.choice(n_regimes, p=A[curr_regime])
        
        # State transition
        F = F0 if curr_regime == 0 else F1
        curr_x = np.dot(F, curr_x) + np.random.multivariate_normal([0,0], Q).reshape(-1, 1)
        
    return z, states, x

def test_kalman_hmm():
    # 1. Generate data
    obs, true_regimes, true_states = generate_switching_data(500)
    
    # 2. Initialize and Fit
    khmm = KalmanHMM(n_regimes=2, state_dim=2, obs_dim=1)
    khmm.fit(obs, n_iter=10, tol=1e-3)
    
    # 3. Filter
    res = khmm.filter(obs)
    pred_regimes = np.argmax(res['prob'], axis=1)
    pred_states = res['combined_mu']
    
    # 4. Plot
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(obs, color='gray', alpha=0.5, label='Observations')
    plt.plot(true_states[:, 0], color='green', label='True State 0 (Pos)')
    plt.plot(pred_states[:, 0], color='blue', linestyle='--', label='Filtered State 0')
    plt.title('KalmanHMM Position Tracking')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(true_regimes, color='green', label='True Regimes', alpha=0.5)
    plt.plot(pred_regimes, color='red', linestyle='--', label='Predicted Regimes')
    plt.title('Regime Detection')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(res['prob'][:, 0], label='P(Regime 0)')
    plt.plot(res['prob'][:, 1], label='P(Regime 1)')
    plt.title('Regime Probabilities')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kalman_hmm_results.png')
    print("Results saved to 'kalman_hmm_results.png'")
    
    # AIC/BIC
    criteria = khmm.get_model_selection_criteria(obs)
    print(f"Model Selection: {criteria}")

if __name__ == "__main__":
    test_kalman_hmm()
