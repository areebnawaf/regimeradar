"""
RegimeRadar Backend — Real R2-RD Implementation
Based on: Hirsa, Xu & Malhotra (2024) "Robust Rolling Regime Detection (R2-RD)"
SSRN 4729435

Run: python app.py
Then open: http://localhost:5000
"""

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# R2-RD Core Algorithm
# ─────────────────────────────────────────────

REGIME_MAP = {
    'bull': {'label': 'Bull / Expansion', 'color': '#4ade80', 'bg': '#052e16', 'text': '#4ade80'},
    'stress': {'label': 'Stress / Crisis', 'color': '#f87171', 'bg': '#2d0a0a', 'text': '#f87171'},
    'transition': {'label': 'Transition', 'color': '#fbbf24', 'bg': '#2d1a00', 'text': '#fbbf24'},
    'recovery': {'label': 'Recovery', 'color': '#94a3b8', 'bg': '#0f172a', 'text': '#94a3b8'},
}


def compute_features(prices: pd.Series, vix: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Build the feature matrix used in R2-RD.
    Two primary features as per Hirsa et al.:
      1. Rolling z-score of volatility proxy (VIX / VIXY)
      2. Momentum (n-day return)
    Plus supplementary features: realized vol, drawdown.
    """
    df = pd.DataFrame({'price': prices, 'vix': vix}).dropna()

    # Log returns
    df['ret'] = np.log(df['price'] / df['price'].shift(1))

    # Feature 1: VIX z-score (rolling)
    df['vix_z'] = (df['vix'] - df['vix'].rolling(window).mean()) / (df['vix'].rolling(window).std() + 1e-8)

    # Feature 2: Momentum (n-day price momentum)
    df['momentum'] = (df['price'] - df['price'].shift(window)) / df['price'].shift(window) * 100

    # Feature 3: Realized volatility (annualized)
    df['realized_vol'] = df['ret'].rolling(window).std() * np.sqrt(252) * 100

    # Feature 4: Drawdown from rolling 60d high
    df['drawdown'] = (df['price'] - df['price'].rolling(60).max()) / df['price'].rolling(60).max() * 100

    df = df.dropna()
    return df


def label_regime(cluster_id: int, cluster_stats: dict) -> str:
    """
    Map GMM cluster IDs to economic regime labels using cluster characteristics.
    Follows the economic interpretation from Hirsa et al. (2024):
    - High VIX + negative momentum → stress
    - Low VIX + positive momentum → bull
    - Medium VIX + recovering momentum → recovery
    - Ambiguous → transition
    """
    stats = cluster_stats[cluster_id]
    vix_z = stats['vix_z_mean']
    mom = stats['momentum_mean']
    vol = stats['vol_mean']

    if vix_z > 1.0 or mom < -8:
        return 'stress'
    elif vix_z > 0.3 and mom < 1:
        return 'transition'
    elif mom < 0 and vix_z < 0.3:
        return 'recovery'
    else:
        return 'bull'


def rolling_gmm_r2rd(df: pd.DataFrame, roll_window: int = 120,
                      n_components: int = 4, smooth_window: int = 5,
                      step: int = 1) -> pd.DataFrame:
    """
    Core R2-RD algorithm: Rolling GMM with temporal ensemble smoothing.

    Key innovations from Hirsa et al. (2024):
    1. Rolling retraining: GMM is retrained on each new window of data
    2. Temporal ensemble: labels are smoothed over a lookback to reduce noise
    3. Data-driven: cluster→regime mapping is inferred from cluster statistics

    Parameters:
        roll_window: Training window length in days
        n_components: Number of GMM mixture components (latent regimes)
        smooth_window: Ensemble smoothing lookback (reduces label chatter)
        step: Retraining frequency (1 = daily, 5 = weekly)
    """
    feature_cols = ['vix_z', 'momentum', 'realized_vol', 'drawdown']
    scaler = StandardScaler()

    n = len(df)
    raw_labels = [''] * n
    raw_confidence = [0.0] * n

    for i in range(roll_window, n, step):
        # Training window
        window_data = df[feature_cols].iloc[i - roll_window:i].values
        window_data_scaled = scaler.fit_transform(window_data)

        # Fit GMM on this window
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                n_init=5,
                max_iter=200,
                random_state=42
            )
            gmm.fit(window_data_scaled)
        except Exception:
            continue

        # Compute cluster stats for economic labeling
        cluster_labels_window = gmm.predict(window_data_scaled)
        cluster_stats = {}
        for c in range(n_components):
            mask = cluster_labels_window == c
            if mask.sum() == 0:
                cluster_stats[c] = {'vix_z_mean': 0, 'momentum_mean': 0, 'vol_mean': 0}
                continue
            cluster_stats[c] = {
                'vix_z_mean': df['vix_z'].iloc[i - roll_window:i].values[mask].mean(),
                'momentum_mean': df['momentum'].iloc[i - roll_window:i].values[mask].mean(),
                'vol_mean': df['realized_vol'].iloc[i - roll_window:i].values[mask].mean(),
            }

        # Classify current point
        end = min(i + step, n)
        for j in range(i, end):
            point = df[feature_cols].iloc[j:j+1].values
            point_scaled = scaler.transform(point)

            cluster = gmm.predict(point_scaled)[0]
            proba = gmm.predict_proba(point_scaled)[0]

            raw_labels[j] = label_regime(cluster, cluster_stats)
            raw_confidence[j] = float(proba.max() * 100)

    # Fill early period with first valid label
    first_valid = next((i for i, l in enumerate(raw_labels) if l), roll_window)
    for i in range(first_valid):
        raw_labels[i] = raw_labels[first_valid] if first_valid < n else 'bull'

    df = df.copy()
    df['raw_label'] = raw_labels
    df['raw_confidence'] = raw_confidence

    # Temporal ensemble smoothing (key R2-RD innovation)
    smoothed = []
    for i in range(n):
        window = raw_labels[max(0, i - smooth_window):i + 1]
        counts = {}
        for l in window:
            counts[l] = counts.get(l, 0) + 1
        smoothed.append(max(counts, key=counts.get) if counts else 'bull')

    df['regime'] = smoothed

    # Confidence: rolling label stability score (0–100)
    confidence_scores = []
    for i in range(n):
        window = smoothed[max(0, i - 10):i + 1]
        cur = smoothed[i]
        score = int(window.count(cur) / len(window) * 100)
        confidence_scores.append(score)
    df['confidence'] = confidence_scores

    return df


def fetch_market_data(ticker_price: str, ticker_vix: str, period: str = '2y') -> tuple:
    """Fetch price and VIX proxy data from Yahoo Finance."""
    price_data = yf.download(ticker_price, period=period, progress=False, auto_adjust=True)
    vix_data = yf.download(ticker_vix, period=period, progress=False, auto_adjust=True)

    price = price_data['Close'].squeeze()
    vix = vix_data['Close'].squeeze()

    return price, vix


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sp500')
def api_sp500():
    try:
        price, vix = fetch_market_data('SPY', 'VIXY', period='3y')
        df = compute_features(price, vix, window=20)
        result = rolling_gmm_r2rd(df, roll_window=120, n_components=4, smooth_window=5)

        # Build response
        recent = result.tail(252)  # 1 year for charts
        dates = [d.strftime('%Y-%m-%d') for d in recent.index]

        return jsonify({
            'success': True,
            'current': {
                'regime': result['regime'].iloc[-1],
                'confidence': int(result['confidence'].iloc[-1]),
                'spy_price': float(price.iloc[-1]),
                'spy_change_pct': float((price.iloc[-1] - price.iloc[-2]) / price.iloc[-2] * 100),
                'vixy': float(vix.iloc[-1]),
                'vixy_change_pct': float((vix.iloc[-1] - vix.iloc[-2]) / vix.iloc[-2] * 100),
                'realized_vol': float(result['realized_vol'].iloc[-1]),
            },
            'series': {
                'dates': dates,
                'spy': [round(float(v), 2) for v in recent['price']],
                'vixy': [round(float(v), 2) for v in recent['vix']],
                'regime': list(recent['regime']),
                'confidence': [int(v) for v in recent['confidence']],
                'momentum': [round(float(v), 2) for v in recent['momentum']],
                'realized_vol': [round(float(v), 2) for v in recent['realized_vol']],
            },
            'history': build_annual_history(result),
            'regime_stats': compute_regime_stats(result),
            'updated_at': pd.Timestamp.now().strftime('%b %d, %Y %H:%M UTC'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/btc')
def api_btc():
    try:
        price, vix = fetch_market_data('BTC-USD', 'VIXY', period='3y')

        # Align both series on common dates
        common_idx = price.index.intersection(vix.index)
        price = price.loc[common_idx]
        vix = vix.loc[common_idx]

        df = compute_features(price, vix, window=20)
        result = rolling_gmm_r2rd(df, roll_window=120, n_components=4, smooth_window=5)

        recent = result.tail(180)
        dates = [d.strftime('%Y-%m-%d') for d in recent.index]

        return jsonify({
            'success': True,
            'current': {
                'regime': result['regime'].iloc[-1],
                'confidence': int(result['confidence'].iloc[-1]),
                'btc_price': float(price.iloc[-1]),
                'btc_change_pct': float((price.iloc[-1] - price.iloc[-2]) / price.iloc[-2] * 100),
                'realized_vol': float(result['realized_vol'].iloc[-1]),
                'momentum_20d': float(result['momentum'].iloc[-1]),
            },
            'series': {
                'dates': dates,
                'btc': [round(float(v), 2) for v in recent['price']],
                'regime': list(recent['regime']),
                'confidence': [int(v) for v in recent['confidence']],
                'momentum': [round(float(v), 2) for v in recent['momentum']],
                'realized_vol': [round(float(v), 2) for v in recent['realized_vol']],
            },
            'history': build_annual_history(result),
            'updated_at': pd.Timestamp.now().strftime('%b %d, %Y %H:%M UTC'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def build_annual_history(result: pd.DataFrame) -> list:
    """Aggregate regime labels by year for the timeline visualization."""
    result = result.copy()
    result['year'] = result.index.year
    history = []
    for year, group in result.groupby('year'):
        counts = group['regime'].value_counts()
        total = len(group)
        segs = []
        for regime in ['bull', 'transition', 'stress', 'recovery']:
            if regime in counts:
                segs.append({'r': regime, 'w': round(counts[regime] / total * 100)})
        history.append({'year': int(year), 'segs': segs})
    return history


def compute_regime_stats(result: pd.DataFrame) -> dict:
    """Compute summary statistics per regime for the methodology panel."""
    stats = {}
    for regime in ['bull', 'stress', 'transition', 'recovery']:
        mask = result['regime'] == regime
        if mask.sum() == 0:
            continue
        r = result[mask]
        rets = r['price'].pct_change().dropna()
        stats[regime] = {
            'count_days': int(mask.sum()),
            'pct_of_time': round(mask.mean() * 100, 1),
            'avg_daily_return': round(float(rets.mean() * 100), 3),
            'avg_realized_vol': round(float(r['realized_vol'].mean()), 1),
            'avg_momentum': round(float(r['momentum'].mean()), 1),
        }
    return stats


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
