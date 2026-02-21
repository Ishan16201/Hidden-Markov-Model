

# Regime-Kalman-OHLC

**Regime-Kalman-OHLC** is a state-space modeling framework that combines **Hidden Markov Models (HMM)** with **Kalman Filters** to decode the "hidden" logic of financial markets. Instead of using a one-size-fits-all moving average, this library detects the current market regime (Bull, Bear, or Sideways) and applies an adaptive filter to denoise OHLC (Open, High, Low, Close) data in real-time.

## 🚀 Overview

Financial time-series data is non-stationary—the rules of the game change constantly. This project implements a **Switching Kalman Filter** architecture:

1. **HMM Layer:** Identifies the latent market state (Regime Detection).
2. **Kalman Layer:** Estimates the "True Price" by filtering out microstructure noise.
3. **Adaptive Feedback:** The model updates its expectations based on the prediction error (innovation) of the filtered signal.

## 🛠️ Key Features

* **Multi-Regime Detection:** Automatically classify market environments using Gaussian HMMs.
* **Dynamic Denoising:** A Kalman Filter that adjusts its "gain" based on the detected regime's volatility.
* **Feature Engineering:** Includes built-in support for Log-Returns, ATR-based volatility, and OHLC range analysis.
* **Visual Analysis:** Color-coded charts showing price action mapped to hidden states.

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/Regime-Kalman-OHLC.git
cd Regime-Kalman-OHLC
pip install -r requirements.txt

```

## 💻 Quick Start

```python
from model import RegimeFilter
import yfinance as yf

# Load OHLC data
df = yf.download("SPY", start="2023-01-01")

# Initialize and fit the hybrid model
rf = RegimeFilter(n_regimes=3)
rf.fit(df)

# Get the denoised "Fair Value" estimate
denoised_price = rf.transform(df)

```

## 📊 Roadmap

* [ ] Implement Unscented Kalman Filter (UKF) for non-linear volatility tracking.
* [ ] Add Interacting Multiple Model (IMM) algorithm to smooth state transitions.
* [ ] Backtesting module integration for regime-based strategies.

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.

