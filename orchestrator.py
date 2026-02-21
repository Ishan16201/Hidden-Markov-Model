import numpy as np
import joblib
import logging
import json
import os
from typing import Dict, List, Optional, Union, Tuple
from kalman_hmm import KalmanHMM
from online_khmm import OnlineKalmanHMM
from initializer import KalmanHMMInitializer
from diagnostics import KalmanHMMDiagnostics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KalmanHMMSystem:
    """
    Unified orchestrator for KalmanHMM lifecycle:
    Training, Evaluation, Deployment, and Monitoring.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_regimes = config.get('n_regimes', 2)
        self.state_dim = config.get('state_dim', 2)
        self.obs_dim = config.get('obs_dim', 1)
        
        self.model = None
        self.online_model = None
        self.training_history = []
        self.diagnostics_report = None
        
    def train(self, observations: np.ndarray, validation_data: Optional[np.ndarray] = None) -> Dict:
        """
        End-to-end training pipeline.
        """
        logger.info("Starting training pipeline...")
        
        # 1. Preprocessing (Basic validation)
        obs = self._preprocess(observations)
        
        # 2. Initialization
        logger.info("Initializing parameters...")
        initializer = KalmanHMMInitializer(self.config)
        init_params = initializer.initialize_from_data(obs, method=self.config.get('init_method', 'clustering'))
        
        # 3. Model Creation
        self.model = KalmanHMM(self.n_regimes, self.state_dim, self.obs_dim)
        self.model.A = init_params['A']
        self.model.pi = init_params['pi']
        self.model.F = init_params['F']
        self.model.H = init_params['H']
        self.model.Q = init_params['Q']
        self.model.R = init_params['R']
        
        # 4. EM Training
        logger.info("Starting EM algorithm...")
        self.model.fit(
            obs, 
            n_iter=self.config.get('em_iterations', 50), 
            tol=self.config.get('em_tol', 1e-4)
        )
        self.training_history = self.model.history
        
        # 5. Diagnostics
        if self.config.get('enable_diagnostics', True):
            logger.info("Running post-training diagnostics...")
            diag = KalmanHMMDiagnostics(self.model, obs)
            self.diagnostics_report = diag.run_full_diagnostics()
            diag.plot_diagnostic_report(self.config.get('diagnostic_plot_path', 'system_diagnostics.png'))
            
        return {
            'status': 'success',
            'll_history': self.training_history,
            'criteria': self.model.get_model_selection_criteria(obs)
        }

    def evaluate(self, test_observations: np.ndarray) -> Dict:
        """Evaluate model on held-out data."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")
            
        test_obs = self._preprocess(test_observations)
        ll = self.model.score(test_obs)
        criteria = self.model.get_model_selection_criteria(test_obs)
        
        return {
            'log_likelihood': ll,
            'BIC': criteria['BIC'],
            'AIC': criteria['AIC']
        }

    def deploy_online(self) -> OnlineKalmanHMM:
        """Convert trained model to online inference mode."""
        if self.model is None:
            raise ValueError("Model must be trained before deployment.")
            
        self.online_model = OnlineKalmanHMM(self.model)
        if self.config.get('enable_adaptation', False):
            self.online_model.enable_adaptation(
                enabled=True,
                forgetting_factor=self.config.get('forgetting_factor', 0.99)
            )
        return self.online_model

    def save_model(self, filepath: str):
        """Serialize the entire system."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        data = {
            'config': self.config,
            'model': self.model,
            'training_history': self.training_history,
            'diagnostics': self.diagnostics_report
        }
        joblib.dump(data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'KalmanHMMSystem':
        """Load a serialized system."""
        data = joblib.load(filepath)
        system = cls(data['config'])
        system.model = data['model']
        system.training_history = data['training_history']
        system.diagnostics_report = data['diagnostics']
        logger.info(f"Model loaded from {filepath}")
        return system

    def forecast(self, observations: np.ndarray, steps: int = 10) -> Dict:
        """Batch forecasting from a sequence of data."""
        if self.model is None:
            raise ValueError("Model must be trained before forecasting.")
        preds, regimes = self.model.predict_next(observations, steps=steps)
        return {
            'observations': preds,
            'regime_probs': regimes
        }

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is in correct numpy format."""
        data = np.array(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data

    def export_parameters(self, filepath: str):
        """Export model parameters to JSON."""
        params = {
            'A': self.model.A.tolist(),
            'pi': self.model.pi.tolist(),
            'F': [f.tolist() for f in self.model.F],
            'H': [h.tolist() for h in self.model.H],
            'Q': [q.tolist() for q in self.model.Q],
            'R': [r.tolist() for r in self.model.R]
        }
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Parameters exported to {filepath}")
