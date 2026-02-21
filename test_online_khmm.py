import numpy as np
import time
import matplotlib.pyplot as plt
from kalman_hmm import KalmanHMM
from online_khmm import OnlineKalmanHMM
from initializer import KalmanHMMInitializer

def test_online_khmm():
    n_samples = 300
    regime_switch_at = 150
    
    # 1. Generate data for training a model first
    # Regime 0: Static Mean=0
    # Regime 1: Constant Velocity
    obs = np.random.normal(0, 0.5, (n_samples, 1))
    obs[regime_switch_at:] += np.arange(n_samples - regime_switch_at).reshape(-1, 1) * 0.5
    
    config = {
        'n_regimes': 2,
        'state_dim': 2,
        'obs_dim': 1,
        'F_init': 'random_walk'
    }
    
    print("Training base model...")
    init = KalmanHMMInitializer(config)
    params = init.initialize_from_data(obs, method='clustering')
    
    khmm = KalmanHMM(2, 2, 1)
    khmm.A = params['A']
    khmm.pi = params['pi']
    khmm.F = params['F']
    khmm.H = params['H']
    khmm.Q = params['Q']
    khmm.R = params['R']
    khmm.fit(obs, n_iter=5)
    
    # 2. Test Online Inference
    online_khmm = OnlineKalmanHMM(khmm)
    online_khmm.enable_adaptation(enabled=True, forgetting_factor=0.98)
    
    streaming_res = []
    latencies = []
    
    print("Simulating streaming data...")
    for t in range(n_samples):
        start_time = time.perf_counter()
        
        # Process new observation
        state = online_khmm.update(obs[t])
        
        # Get multi-step prediction
        forecast = online_khmm.get_prediction(steps=10)
        
        latency = (time.perf_counter() - start_time) * 1000
        latencies.append(latency)
        
        streaming_res.append({
            'prob': state['regime_probs'],
            'est': state['state_estimate'],
            'conf': state['confidence'],
            'forecast': forecast['observations']
        })
        
    print(f"Average latency: {np.mean(latencies):.2f} ms")
    print(f"Max latency: {np.max(latencies):.2f} ms")
    
    # 3. Plot Results
    probs = np.array([r['prob'] for r in streaming_res])
    ests = np.array([r['est'] for r in streaming_res])
    confs = np.array([r['conf'] for r in streaming_res])
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(obs, color='gray', alpha=0.3, label='Streaming Obs')
    plt.plot(ests[:, 0], color='blue', label='Online State Estimate')
    plt.axvline(x=regime_switch_at, color='red', linestyle='--', label='True Switch')
    plt.title('Online Position Tracking')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(probs[:, 0], label='P(Regime 0)')
    plt.plot(probs[:, 1], label='P(Regime 1)')
    plt.title('Real-time Regime Probabilities')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(confs, label='Inference Confidence', color='green')
    plt.title('Model Confidence')
    plt.xlabel('Time Step')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('online_khmm_streaming.png')
    print("Streaming results saved to 'online_khmm_streaming.png'")

if __name__ == "__main__":
    test_online_khmm()
