"""
Fetch hourly BTC/USDT data from Binance, build simple features, train an XGBoost multiclass classifier
(label: short=-1, neutral=0, long=1 -> mapped to 0,1,2), and explain predictions with SHAP.
Requires: ccxt, pandas, numpy, xgboost, scikit-learn, shap
"""

import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap

# -------------------------
# Indicator helpers
# -------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -------------------------
# Fetch data
# -------------------------
def fetch_ohlcv_binance(symbol='BTC/USDT', timeframe='1h', limit=1000):
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        raise RuntimeError(f"Error fetching OHLCV from Binance: {e}")
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df

# -------------------------
# Prepare features & labels
# -------------------------
def prepare_features_labels(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    # Indicators
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['atr'] = compute_atr(df, 14)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['price_change'] = df['close'].pct_change()
    df['vol_spike'] = df['volume'] / df['volume'].rolling(20).mean()
    df['ema_diff'] = (df['ema20'] - df['ema50']) / df['close']
    df['vwap_diff'] = (df['close'] - df['vwap']) / df['close']

    feature_cols = ['price_change', 'vol_spike', 'atr', 'ema_diff', 'vwap_diff', 'rsi']
    features = df[feature_cols]

    # Label: 1 if next close > close * 1.001, -1 if next close < close * 0.999 else 0
    df['label'] = np.where(df['close'].shift(-1) > df['close'] * 1.001, 1,
                           np.where(df['close'].shift(-1) < df['close'] * 0.999, -1, 0))

    # Drop NA rows from features and keep aligned labels
    features = features.dropna()
    labels = df.loc[features.index, 'label']

    # Map labels to [0,1,2] for XGBoost (keep mapping for interpretation)
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = labels.map(label_map).astype(int)

    return features, y, label_map, inv_label_map

# -------------------------
# Main flow
# -------------------------
def main():
    df = fetch_ohlcv_binance('BTC/USDT', '1h', limit=1200)

    features, y, label_map, inv_label_map = prepare_features_labels(df)
    X = features.values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Predict on latest available sample
    latest_df = features.iloc[[-1]]  # keep as DataFrame (1 x n_features)
    latest_X = latest_df.values
    prob = model.predict_proba(latest_X)[0]  # probabilities for [0,1,2] (mapped labels)
    # Map back to original labels
    prob_by_original_label = {inv_label_map[i]: float(prob[i]) for i in range(len(prob))}
    print("Probabilities by original label:", prob_by_original_label)
    print(f"Predicted mapped label: {model.predict(latest_X)[0]}, original label: "
          f"{inv_label_map[model.predict(latest_X)[0]]}")

    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    # Global: summary on training set
    shap_values_train = explainer.shap_values(X_train)  # list of arrays for multiclass
    try:
        shap.summary_plot(shap_values_train, features.iloc[:len(X_train)], show=False)
        print("Displayed SHAP summary plot (global importance on training set).")
    except Exception:
        # In case running headless or plotting backend issues
        pass

    # Local explanation for the predicted class of latest sample
    mapped_pred = int(model.predict(latest_X)[0])
    # shap_values for multiclass: shap_values[mapped_pred][0] is the contribution for this sample
    sv_local = shap_values_train[mapped_pred]
    # Use SHAP force plot (or bar) for single sample
    try:
        # Use the TreeExplainer's expected_value for the mapped class
        expected_value = explainer.expected_value[mapped_pred]
        shap.force_plot(expected_value, sv_local[-1], latest_df, matplotlib=True, show=True)
    except Exception:
        # fallback: print feature contributions numerically
        contribs = pd.Series(sv_local[-1], index=features.columns).sort_values(key=abs, ascending=False)
        print("Local SHAP contributions (sorted by absolute value):")
        print(contribs)

if __name__ == '__main__':
    main()