import numpy as np
from kalman_hmm import KalmanHMM
from diagnostics import KalmanHMMDiagnostics
from initializer import KalmanHMMInitializer

def test_diagnostics():
    n_samples = 500
    # Generate switching data (Regime 0: Low noise, Regime 1: High noise)
    obs = np.random.normal(0, 0.2, (n_samples, 1))
    obs[250:] = np.random.normal(0, 1.5, (n_samples - 250, 1))
    
    config = {
        'n_regimes': 2,
        'state_dim': 1,
        'obs_dim': 1,
    }
    
    print("Initializing and fitting model...")
    init = KalmanHMMInitializer(config)
    params = init.initialize_from_data(obs)
    
    model = KalmanHMM(2, 1, 1)
    model.A = params['A']
    model.pi = params['pi']
    model.F = params['F']
    model.H = params['H']
    model.Q = params['Q']
    model.R = params['R']
    model.fit(obs, n_iter=5)
    
    print("\nRunning Diagnostics...")
    diag = KalmanHMMDiagnostics(model, obs)
    
    # Run full diagnostics
    results = diag.run_full_diagnostics()
    
    print("\nFit Metrics:")
    for k, v in results['fit_metrics'].items():
        print(f"  {k}: {v:.4f}")
        
    print("\nResidual Analysis:")
    for k, v in results['residual_analysis'].items():
        print(f"  {k}: {v}")
        
    print("\nRegime Analysis:")
    print(f"  Avg Persistence: {results['regime_analysis']['avg_persistence']:.2f}")
    
    # Generate plots and report
    diag.plot_diagnostic_report('kalman_hmm_diagnostics.png')
    summary = diag.generate_summary_report()
    
    print("\nSummary Report:")
    print(summary)
    
    with open('diagnostic_summary.txt', 'w') as f:
        f.write(summary)
        
    print("\nDiagnostics complete. Plots saved to 'kalman_hmm_diagnostics.png'")

if __name__ == "__main__":
    test_diagnostics()
