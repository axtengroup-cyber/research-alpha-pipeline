"""
US Equities 5-Minute Alpha Model: Data Ingestion & Preprocessing Pipeline
=========================================================================
A self-contained, vectorized pipeline for transforming flat OHLCV bars
into clean 2-D signal-ready matrices (timestamps × tickers).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings


# --------------------------------------------------------------------------- #
# 1. DATA INGESTION (Synthetic OHLCV Generator)
# --------------------------------------------------------------------------- #
def generate_synthetic_ohlcv(
    tickers: List[str],
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-01",
    freq: str = "5min",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic 5-minute OHLCV data for a universe of tickers.
    
    Parameters
    ----------
    tickers : List[str]
        Ticker symbols to simulate.
    start_date, end_date : str
        Date range (inclusive start, exclusive end).
    freq : str
        Pandas frequency string (default '5min' for 5-minute bars).
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Flat long-form DataFrame with columns:
        ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    """
    rng = np.random.default_rng(seed)
    
    # Generate market-hours timestamps (09:30 - 16:00 EST), Mon-Fri only
    full_idx = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive='left')
    valid_mask = (
        (full_idx.time >= pd.Timestamp("09:30").time()) &
        (full_idx.time <= pd.Timestamp("16:00").time()) &
        (full_idx.dayofweek < 5)
    )
    timestamps = full_idx[valid_mask]
    
    records = []
    for ticker in tickers:
        n = len(timestamps)
        
        # Random walk for close prices with slight drift
        returns = rng.normal(loc=0.00005, scale=0.0015, size=n)
        log_prices = np.cumsum(returns)
        close = 100.0 * np.exp(log_prices)
        
        # High / Low / Open around close
        intrabar_vol = rng.exponential(scale=0.0010, size=n)
        high = close * (1 + intrabar_vol)
        low = close * (1 - intrabar_vol * rng.uniform(0.5, 1.0, size=n))
        open_price = low + (high - low) * rng.uniform(0.2, 0.8, size=n)
        
        # Enforce OHLC consistency
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))
        
        # Volume: log-normal with intraday U-shape seasonality
        base_vol = rng.lognormal(mean=10.0, sigma=1.2, size=n)
        minutes_from_open = (timestamps.hour - 9) * 60 + (timestamps.minute - 30)
        minutes_from_open = np.where(minutes_from_open < 0, 0, minutes_from_open)
        session_shape = 1.0 + 0.5 * np.sin(np.pi * minutes_from_open / 390)
        volume = (base_vol * session_shape).astype(np.int64)
        
        df_ticker = pd.DataFrame({
            'timestamp': timestamps,
            'ticker': ticker,
            'open': open_price.astype(np.float32),
            'high': high.astype(np.float32),
            'low': low.astype(np.float32),
            'close': close.astype(np.float32),
            'volume': volume.astype(np.int64),
        })
        records.append(df_ticker)
    
    df = pd.concat(records, ignore_index=True)
    df.sort_values(['ticker', 'timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# --------------------------------------------------------------------------- #
# 2. LIQUIDITY FILTERING (30-Day ADV)
# --------------------------------------------------------------------------- #
def filter_liquid_tickers(
    df: pd.DataFrame,
    min_adv: float = 5_000_000,
    price_min: float = 5.0,
) -> pd.DataFrame:
    """
    Filter universe to liquid tickers based on 30-day Average Daily Volume.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flat OHLCV data.
    min_adv : float
        Minimum 30-day average daily volume (shares).
    price_min : float
        Minimum last close price to avoid penny stocks.
    
    Returns
    -------
    pd.DataFrame
        Subset of `df` containing only tickers that pass liquidity filters.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Daily aggregates
    daily = df.groupby(['ticker', 'date']).agg(
        daily_volume=('volume', 'sum'),
        last_close=('close', 'last')
    ).reset_index()
    
    # Rolling 30-day ADV per ticker
    daily = daily.sort_values(['ticker', 'date'])
    daily['adv_30'] = (
        daily.groupby('ticker')['daily_volume']
        .transform(lambda x: x.rolling(window=30, min_periods=15).mean())
    )
    
    # Screen on latest ADV and price
    latest = daily.groupby('ticker').last().reset_index()
    liquid_tickers = latest.loc[
        (latest['adv_30'] >= min_adv) & (latest['last_close'] >= price_min),
        'ticker'
    ].unique()
    
    print(f"[FILTER] Universe: {df['ticker'].nunique()} tickers -> "
          f"{len(liquid_tickers)} liquid tickers "
          f"(ADV≥{min_adv:,.0f}, Price≥${price_min:.2f})")
    
    return df[df['ticker'].isin(liquid_tickers)].copy()


# --------------------------------------------------------------------------- #
# 3. VWAP CALCULATION & MATRIX PIVOT
# --------------------------------------------------------------------------- #
def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bar cumulative VWAP using typical price (H+L+C)/3 × Volume.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flat OHLCV data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'vwap' column (float32).
    """
    df = df.copy()
    typical_price = (
        df['high'].astype(np.float64) + 
        df['low'].astype(np.float64) + 
        df['close'].astype(np.float64)
    ) / 3.0
    
    df['vwap'] = typical_price * df['volume'].astype(np.float64)
    
    # Cumulative VWAP per ticker per day (industry standard)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    cum_pv = df.groupby(['ticker', 'date'])['vwap'].cumsum()
    cum_vol = df.groupby(['ticker', 'date'])['volume'].cumsum()
    
    df['vwap'] = (cum_pv / cum_vol.replace(0, np.nan)).astype(np.float32)
    df.drop(columns=['date'], inplace=True)
    return df


def pivot_to_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pivot flat OHLCV bars into 2-D timestamp × ticker matrices.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flat DataFrame with ['timestamp', 'ticker', 'close', 'volume', 'vwap'].
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_close, df_volume, df_vwap) each indexed by timestamp, columns by ticker.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df_close = df.pivot(index='timestamp', columns='ticker', values='close')
    df_volume = df.pivot(index='timestamp', columns='ticker', values='volume')
    df_vwap = df.pivot(index='timestamp', columns='ticker', values='vwap')
    
    # Memory-efficient dtypes
    df_close = df_close.astype(np.float32)
    df_vwap = df_vwap.astype(np.float32)
    df_volume = df_volume.fillna(0).astype(np.int64)
    
    return df_close, df_volume, df_vwap


# --------------------------------------------------------------------------- #
# 4. DATA CLEANING (Forward Fill + Missing Value Handling)
# --------------------------------------------------------------------------- #
def clean_matrices(
    df_close: pd.DataFrame,
    df_volume: pd.DataFrame,
    df_vwap: pd.DataFrame,
    ffill_limit: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean signal-ready matrices.
    
    - Forward-fill missing price quotes up to `ffill_limit` bars.
    - Drop columns (tickers) that still contain NaNs after ffill.
    - Fill remaining volume gaps with 0.
    
    Parameters
    ----------
    df_close, df_volume, df_vwap : pd.DataFrame
        Raw pivoted matrices.
    ffill_limit : int
        Maximum consecutive NaNs to forward-fill for prices.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Cleaned (df_close, df_volume, df_vwap).
    """
    # Forward fill prices (limit=2 bars = 10 minutes)
    df_close = df_close.ffill(limit=ffill_limit)
    df_vwap = df_vwap.ffill(limit=ffill_limit)
    
    # Drop tickers that still have gaps (illiquid or recent listing)
    valid_tickers = df_close.columns[df_close.notna().all()].tolist()
    dropped = set(df_close.columns) - set(valid_tickers)
    if dropped:
        print(f"[CLEAN] Dropped {len(dropped)} tickers with persistent gaps: {sorted(dropped)}")
    
    df_close = df_close[valid_tickers].astype(np.float32)
    df_vwap = df_vwap[valid_tickers].astype(np.float32)
    df_volume = df_volume[valid_tickers].fillna(0).astype(np.int64)
    
    print(f"[CLEAN] Final matrix shape: {df_close.shape} "
          f"(timestamps={df_close.shape[0]}, tickers={df_close.shape[1]})")
    
    return df_close, df_volume, df_vwap


# --------------------------------------------------------------------------- #
# 5. MAIN ORCHESTRATOR
# --------------------------------------------------------------------------- #
def build_alpha_matrices(
    tickers: Optional[List[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-03-01",
    min_adv: float = 5_000_000,
    price_min: float = 5.0,
    ffill_limit: int = 2,
    use_yfinance: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    End-to-end pipeline: ingest → filter → pivot → clean → output.
    
    Parameters
    ----------
    tickers : List[str], optional
        Universe of tickers. Defaults to synthetic large-cap list.
    start_date, end_date : str
        Lookback window.
    min_adv : float
        30-day ADV threshold (shares).
    price_min : float
        Minimum price threshold.
    ffill_limit : int
        Forward-fill limit for prices.
    use_yfinance : bool
        If True, attempts to download from YFinance (requires internet).
        If False, uses high-fidelity synthetic generator.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        {'close': df_close, 'volume': df_volume, 'vwap': df_vwap}
    """
    if tickers is None:
        tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B",
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "ABBV", "PFE",
            "KO", "PEP", "COST", "TMO", "AVGO", "DIS", "CSCO", "VZ", "ADBE",
            "CRM", "ACN", "WMT", "MRK", "NKE", "ABT", "CMCSA", "XOM", "CVX",
            "TXN", "QCOM", "NEE", "RTX", "HON", "PM", "IBM", "PYPL", "INTC"
        ]
    
    # --- 1. Ingestion -------------------------------------------------------- #
    if use_yfinance:
        try:
            import yfinance as yf
            print("[INGEST] Downloading from YFinance (5m bars, max 60 days)...")
            df_list = []
            for t in tickers:
                try:
                    data = yf.download(t, start=start_date, end=end_date,
                                       interval="5m", progress=False)
                    if not data.empty:
                        data = data.reset_index()
                        data.columns = [c[0] if isinstance(c, tuple) else c 
                                        for c in data.columns]
                        data = data.rename(columns={
                            'Datetime': 'timestamp', 'Open': 'open', 'High': 'high',
                            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                        })
                        data['ticker'] = t
                        df_list.append(data[['timestamp', 'ticker', 'open',
                                             'high', 'low', 'close', 'volume']])
                except Exception as e:
                    print(f"[WARN] YFinance failed for {t}: {e}")
            df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
            if df.empty:
                raise ValueError("YFinance returned no data; falling back to synthetic.")
        except Exception as e:
            print(f"[WARN] YFinance ingestion failed ({e}). Switching to synthetic data.")
            df = generate_synthetic_ohlcv(tickers, start_date, end_date)
    else:
        print("[INGEST] Generating synthetic 5-minute OHLCV data...")
        df = generate_synthetic_ohlcv(tickers, start_date, end_date)
    
    print(f"[INGEST] Raw bars: {len(df):,}  |  Tickers: {df['ticker'].nunique()}")
    
    # --- 2. Liquidity Filter ------------------------------------------------- #
    df = filter_liquid_tickers(df, min_adv=min_adv, price_min=price_min)
    
    # --- 3. VWAP & Pivot ----------------------------------------------------- #
    df = compute_vwap(df)
    df_close, df_volume, df_vwap = pivot_to_matrices(df)
    
    # --- 4. Clean ------------------------------------------------------------ #
    df_close, df_volume, df_vwap = clean_matrices(
        df_close, df_volume, df_vwap, ffill_limit
    )
    
    return {
        'close': df_close,
        'volume': df_volume,
        'vwap': df_vwap
    }


# --------------------------------------------------------------------------- #
# 6. EXECUTION
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    result = build_alpha_matrices(
        tickers=None,
        start_date="2024-01-01",
        end_date="2024-03-01",
        min_adv=1_000_000,
        price_min=5.0,
        ffill_limit=2,
        use_yfinance=False
    )
    
    df_close = result['close']
    df_volume = result['volume']
    df_vwap = result['vwap']
    
    print("\n" + "="*70)
    print("OUTPUT MATRICES READY FOR SIGNAL GENERATION")
    print("="*70)
    print(f"\nClose Matrix  ->  Shape: {df_close.shape}  |  Dtype: {df_close.dtypes.iloc[0]}")
    print(f"Volume Matrix ->  Shape: {df_volume.shape}  |  Dtype: {df_volume.dtypes.iloc[0]}")
    print(f"VWAP Matrix   ->  Shape: {df_vwap.shape}  |  Dtype: {df_vwap.dtypes.iloc[0]}")
    print(f"\nNaN in Close: {df_close.isna().sum().sum()}  |  Zero Volume: {(df_volume==0).sum().sum()}")
    print("\nPreview (Close):")
    print(df_close.head(3))
    print("\nPreview (Volume):")
    print(df_volume.head(3))
