import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Optional, Union, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KalmanHMMDiagnostics:
    """
    Comprehensive evaluation and diagnostics for KalmanHMM models.
    """
    
    def __init__(self, model, observations: np.ndarray):
        self.model = model
        self.obs = observations
        self.T = observations.shape[0]
        self.res = model.filter(observations)
        self.smooth_res = model.smooth(observations)
        
    def run_full_diagnostics(self) -> Dict:
        """
        Run all diagnostic tests and return results.
        """
        results = {}
        results['fit_metrics'] = self.compute_fit_metrics()
        results['residual_analysis'] = self.analyze_residuals()
        results['regime_analysis'] = self.analyze_regimes()
        results['parameter_stability'] = self.check_stability()
        results['performance'] = self.benchmark_performance()
        
        return results

    def compute_fit_metrics(self) -> Dict:
        """Compute LL, AIC, BIC, Perplexity."""
        ll = self.model.score(self.obs)
        criteria = self.model.get_model_selection_criteria(self.obs)
        
        # Perplexity = exp(-LL / T)
        perplexity = np.exp(-ll / self.T)
        
        return {
            'log_likelihood': ll,
            'AIC': criteria['AIC'],
            'BIC': criteria['BIC'],
            'perplexity': perplexity
        }

    def analyze_residuals(self) -> Dict:
        """Analyze innovations for whiteness and normality."""
        # Simplified innovation sequence from filter
        # In a real IMM, innovations are regime-weighted or taken from combined estimate
        combined_mu = self.res['combined_mu']
        H = self.model.H[0] # Assumes same H for simplicity in residuals
        innovations = self.obs - np.dot(combined_mu, H.T)
        
        innov_flat = innovations.flatten()
        
        # 1. Normality (Jarque-Bera)
        jb_stat, jb_p = stats.jarque_bera(innov_flat)
        
        # 2. Whiteness (Ljung-Box)
        lb_res = acorr_ljungbox(innov_flat, lags=[10], return_df=True)
        lb_p = lb_res['lb_pvalue'].iloc[0]
        
        # 3. Standardized Residuals
        # (Simplified: normalize by overall variance)
        std_residuals = innov_flat / np.std(innov_flat)
        
        return {
            'jarque_bera_p': jb_p,
            'ljung_box_p': lb_p,
            'is_gaussian': jb_p > 0.05,
            'is_white': lb_p > 0.05,
            'max_std_residual': np.max(np.abs(std_residuals))
        }

    def analyze_regimes(self) -> Dict:
        """Regime stats: duration, persistence, transitions."""
        probs = self.res['prob']
        regimes = np.argmax(probs, axis=1)
        
        # 1. Transitions
        trans_counts = np.zeros((self.model.M, self.model.M))
        for i in range(self.T - 1):
            trans_counts[regimes[i], regimes[i+1]] += 1
            
        trans_freq = trans_counts / (trans_counts.sum(axis=1)[:, np.newaxis] + 1e-9)
        
        # 2. Persistence (Mean duration)
        durations = []
        curr_d = 0
        for i in range(1, self.T):
            if regimes[i] == regimes[i-1]:
                curr_d += 1
            else:
                durations.append(curr_d)
                curr_d = 0
        durations.append(curr_d)
        
        return {
            'avg_persistence': np.mean(durations),
            'transitions': trans_freq.tolist(),
            'regime_counts': np.bincount(regimes, minlength=self.model.M).tolist()
        }

    def check_stability(self) -> Dict:
        """Check eigenvalues of F matrices."""
        stability = []
        for j in range(self.model.M):
            eigs = np.abs(np.linalg.eigvals(self.model.F[j]))
            stability.append(np.max(eigs))
            
        return {
            'max_eigenvalues': stability,
            'all_stable': all(e < 1.05 for e in stability)
        }

    def benchmark_performance(self) -> Dict:
        """Measure filtering and training speed."""
        start = time.perf_counter()
        _ = self.model.filter(self.obs)
        filter_time = (time.perf_counter() - start) * 1000
        
        return {
            'filter_time_ms': filter_time,
            'avg_per_step_ms': filter_time / self.T
        }

    def plot_diagnostic_report(self, save_path: str = 'diagnostics_report.png'):
        """Generate a visual diagnostic report."""
        plt.figure(figsize=(15, 20))
        
        # 1. Regime Probabilities
        plt.subplot(4, 1, 1)
        for j in range(self.model.M):
            plt.plot(self.res['prob'][:, j], label=f'Regime {j}')
        plt.title('Regime Probabilities (Filtered)')
        plt.legend()
        
        # 2. Innovations
        combined_mu = self.res['combined_mu']
        H = self.model.H[0]
        innovations = self.obs - np.dot(combined_mu, H.T)
        plt.subplot(4, 1, 2)
        plt.plot(innovations, color='r', alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title('Innovations (Prediction Errors)')
        
        # 3. Innovation Histogram / QQ
        plt.subplot(4, 2, 5)
        plt.hist(innovations, bins=30, density=True, alpha=0.6)
        plt.title('Innovation Distribution')
        
        plt.subplot(4, 2, 6)
        stats.probplot(innovations.flatten(), dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        # 4. Transitions Heatmap
        stats_reg = self.analyze_regimes()
        plt.subplot(4, 2, 7)
        plt.imshow(stats_reg['transitions'], cmap='Blues')
        plt.colorbar()
        plt.title('Transition Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"Diagnostic report saved to {save_path}")

    def generate_summary_report(self) -> str:
        """Generate a text-based recommendation report."""
        res = self.run_full_diagnostics()
        
        report = []
        report.append("=== KalmanHMM Diagnostic Report ===")
        report.append(f"Model: {self.model.M} Regimes, {self.model.n} States")
        report.append(f"Log-Likelihood: {res['fit_metrics']['log_likelihood']:.2f}")
        report.append(f"BIC: {res['fit_metrics']['BIC']:.2f}")
        
        # Residuals
        if not res['residual_analysis']['is_white']:
            report.append("[WARNING] Non-white residuals detected (Ljung-Box p < 0.05). Suggest higher-order dynamics.")
        if not res['residual_analysis']['is_gaussian']:
            report.append("[NOTE] Non-Gaussian residuals detected. Consider robust estimation or student-t emissions.")
            
        # Stability
        if not res['parameter_stability']['all_stable']:
            report.append("[CAUTION] Potentially unstable dynamics detected (Eigenvalues > 1.05).")
            
        report.append(f"Avg Regime Persistence: {res['regime_analysis']['avg_persistence']:.2f} ticks")
        
        return "\n".join(report)
