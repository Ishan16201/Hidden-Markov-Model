import numpy as np
import matplotlib.pyplot as plt
from initializer import KalmanHMMInitializer
from kalman_hmm import KalmanHMM

def test_initialization():
    # 1. Generate multi-regime switching data
    n_samples = 400
    z = np.zeros((n_samples, 1))
    true_regimes = np.zeros(n_samples, dtype=int)
    
    # Dynamics for 2 regimes
    curr_x = np.zeros((2, 1))
    curr_regime = 0
    A_true = np.array([[0.95, 0.05], [0.05, 0.95]])
    
    for t in range(n_samples):
        if curr_regime == 0: # Trending
            F = np.array([[1.0, 0.5], [0.0, 1.0]])
        else: # Oscillating
            F = np.array([[0.8, -0.4], [0.4, 0.8]])
            
        curr_x = np.dot(F, curr_x) + np.random.normal(0, 0.1, (2, 1))
        z[t] = curr_x[0] + np.random.normal(0, 0.2)
        true_regimes[t] = curr_regime
        curr_regime = np.random.choice(2, p=A_true[curr_regime])

    # 2. Test Initializer
    config = {
        'n_regimes': 2,
        'state_dim': 2,
        'obs_dim': 1,
        'hmm_init': 'persistent',
        'F_init': 'ar_fit',
        'Q_init': 'sample_variance',
        'R_init': 'robust_mad'
    }
    
    initializer = KalmanHMMInitializer(config)
    init_params = initializer.initialize_from_data(z, method='clustering')
    
    print("\nInitial Parameters Estimated:")
    print("Transition Matrix A:\n", init_params['A'])
    print("Initial Distributions pi:\n", init_params['pi'])
    
    # 3. Compare with standard KalmanHMM fit convergence
    # Using random init vs smart init
    khmm_smart = KalmanHMM(2, 2, 1)
    # Inject smart parameters
    khmm_smart.A = init_params['A']
    khmm_smart.pi = init_params['pi']
    khmm_smart.F = init_params['F']
    khmm_smart.H = init_params['H']
    khmm_smart.Q = init_params['Q']
    khmm_smart.R = init_params['R']
    
    print("\nFitting KalmanHMM with Smart Initialization...")
    khmm_smart.fit(z, n_iter=5)
    
    khmm_random = KalmanHMM(2, 2, 1)
    print("\nFitting KalmanHMM with Random/Naive Initialization...")
    khmm_random.fit(z, n_iter=5)
    
    # Plot Likelihood comparison
    plt.figure(figsize=(10, 6))
    plt.plot(khmm_smart.history, label='Smart Initialization', marker='o')
    plt.plot(khmm_random.history, label='Naive Initialization', marker='x')
    plt.title('EM Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Likelihood Proxy')
    plt.legend()
    plt.savefig('init_convergence_comparison.png')
    print("\nConvergence plot saved to 'init_convergence_comparison.png'")
    
    # Validate
    is_valid = initializer.validate_parameters(init_params)
    print(f"Parameters Validated: {is_valid}")

if __name__ == "__main__":
    test_initialization()
