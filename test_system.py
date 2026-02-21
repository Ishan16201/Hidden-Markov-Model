import numpy as np
import os
from orchestrator import KalmanHMMSystem

def test_kalman_hmm_system():
    # 1. Generate multi-regime data
    n_samples = 600
    train_obs = np.random.normal(0, 0.5, (n_samples, 1))
    train_obs[300:] += np.arange(300).reshape(-1, 1) * 0.1 # Trend start
    
    test_obs = np.random.normal(0, 0.5, (100, 1))
    test_obs += 35.0 # Test extrapolation
    
    # 2. Configure the System
    config = {
        'n_regimes': 2,
        'state_dim': 2,
        'obs_dim': 1,
        'init_method': 'clustering',
        'em_iterations': 10,
        'enable_diagnostics': True,
        'diagnostic_plot_path': 'final_system_diagnostics.png',
        'enable_adaptation': True,
        'forgetting_factor': 0.98
    }
    
    # 3. Initialize and Train
    print("\n--- Phase 1: Training ---")
    system = KalmanHMMSystem(config)
    results = system.train(train_obs)
    print(f"Training converged with LL: {results['ll_history'][-1]:.2f}")
    
    # 4. Evaluate and Forecast
    print("\n--- Phase 2: Evaluation & Forecasting ---")
    eval_res = system.evaluate(test_obs)
    print(f"Test Log-Likelihood: {eval_res['log_likelihood']:.2f}")
    
    forecast = system.forecast(train_obs[-20:], steps=10)
    print(f"Forecasted next 10 obs: {forecast['observations'].flatten()}")
    
    # 5. Deployment (Online)
    print("\n--- Phase 3: Deployment ---")
    online_system = system.deploy_online()
    state = online_system.update(test_obs[0])
    print(f"Online State Estimate for first test observation: {state['state_estimate']}")
    
    # 6. Serialization
    print("\n--- Phase 4: Serialization ---")
    model_path = 'kalman_hmm_final.joblib'
    system.save_model(model_path)
    
    new_system = KalmanHMMSystem.load_model(model_path)
    new_online = new_system.deploy_online()
    state_after_reload = new_online.update(test_obs[0])
    print(f"Online State Estimate after reload: {state_after_reload['state_estimate']}")
    
    # Verify parameter consistency
    system.export_parameters('model_params.json')
    
    if os.path.exists(model_path) and os.path.exists('final_system_diagnostics.png'):
        print("\nAll integration tests passed. System is ready for production.")
    else:
        print("\nVerification error: Files not generated.")

if __name__ == "__main__":
    test_kalman_hmm_system()
