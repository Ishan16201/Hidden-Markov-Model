import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter

def test_kalman_filter():
    n_samples = 200
    dt = 1.0
    
    # Constant velocity model
    # Transition matrix F: [x, v]' = [1 dt; 0 1] * [x, v]'
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    
    # Observation matrix H: z = [1 0] * [x, v]' (measure position only)
    H = np.array([[1.0, 0.0]])
    
    # Noise covariances
    Q = np.array([[0.1, 0.01],
                  [0.01, 0.1]])
    R = np.array([[10.0]])
    
    # Initial state and covariance
    x0 = np.array([[0.0], [1.0]])
    P0 = np.eye(2) * 500.0
    
    # Generate synthetic data
    true_states = np.zeros((n_samples, 2))
    obs = np.zeros((n_samples, 1))
    
    curr_state = x0.copy()
    for t in range(n_samples):
        true_states[t] = curr_state.flatten()
        obs[t] = np.dot(H, curr_state) + np.random.normal(0, np.sqrt(R[0,0]))
        curr_state = np.dot(F, curr_state) + np.random.multivariate_normal([0, 0], Q).reshape(-1, 1)
    
    # 1. Kalman Filtering
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
    filtered_states, _ = kf.filter(obs)
    
    # 2. Kalman Smoothing (RTS)
    kf_smooth = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
    smoothed_states, _ = kf_smooth.smooth(obs)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Position plot
    plt.subplot(2, 1, 1)
    plt.plot(true_states[:, 0], label='True Position', color='green', alpha=0.8)
    plt.scatter(range(n_samples), obs, label='Observations', color='red', s=5, alpha=0.3)
    plt.plot(filtered_states[:, 0], label='Filtered Position', color='blue', linestyle='--')
    plt.plot(smoothed_states[:, 0], label='Smoothed Position', color='purple', linewidth=2)
    plt.title('Kalman Filter: Position Tracking')
    plt.legend()
    
    # Velocity plot
    plt.subplot(2, 1, 2)
    plt.plot(true_states[:, 1], label='True Velocity', color='green', alpha=0.8)
    plt.plot(filtered_states[:, 1], label='Filtered Velocity', color='blue', linestyle='--')
    plt.plot(smoothed_states[:, 1], label='Smoothed Velocity', color='purple', linewidth=2)
    plt.title('Kalman Filter: Velocity Estimation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kalman_test_results.png')
    print("Results saved to 'kalman_test_results.png'")
    
    # Check Likelihood
    log_likelihood = kf.get_likelihood(obs)
    print(f"Log-Likelihood: {log_likelihood:.4f}")
    
    # Check Mahalanobis distance for an outlier
    outlier = np.array([[obs[0, 0] + 50]])
    md = kf.get_mahalanobis_distance(outlier)
    print(f"Mahalanobis distance for outlier: {md:.4f}")

if __name__ == "__main__":
    test_kalman_filter()
