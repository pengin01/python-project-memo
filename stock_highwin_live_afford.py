# pip install pandas numpy yfinance ta

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import ta


@dataclass
class Params:
    lookback_years: int = 5

    # Signal
    pullback_ret3: float = -0.02
    rsi_min: float = 55.0

    # Market filter (N225)
    market_ticker: str = "^N225"
    market_ma: int = 200

    # Execution / A mode (same-day close out)
    tp: float = 0.012
    sl: float = 0.008
    slippage_oneway: float = 0.0005




    # Capital constraint (100-share lot)
    live_capital: float = 80000
    lot_size: int = 100
    max_positions: int = 1  # 8万なら1銘柄集中が現実的

P = Params()


def load_data(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{years}y", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().copy()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret3"] = df["Close"].pct_change(3)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    return df


def market_ok_series(years: int, ma_days: int) -> pd.Series:
    mkt = load_data(P.market_ticker, years)
    if mkt.empty or len(mkt) < ma_days + 10:
        return pd.Series(dtype=bool)
    close = mkt["Close"].astype(float)
    ma = close.rolling(ma_days).mean()
    ok = (close > ma).fillna(False)
    ok.name = "market_ok"
    return ok


def market_ok_on(market_ok: pd.Series, day: pd.Timestamp) -> bool:
    if market_ok is None or market_ok.empty:
        return True
    day = pd.Timestamp(day).normalize()
    if day in market_ok.index:
        return bool(market_ok.loc[day])
    prev = market_ok.loc[:day]
    if prev.empty:
        return False
    return bool(prev.iloc[-1])


def is_affordable(entry_price_sim: float) -> bool:
    alloc = P.live_capital / P.max_positions
    min_cost = entry_price_sim * P.lot_size
    return min_cost <= alloc


def simulate_same_day_tp_sl(df: pd.DataFrame, entry_i: int) -> float:
    """
    A固定：entry_i 当日に TP/SL を判定し、当たらなければ引け（Close）で決済。
    価格はスリッページ片道を考慮。
    戻り値：net_ret（%）
    """
    o = float(df["Open"].iloc[entry_i])
    h = float(df["High"].iloc[entry_i])
    l = float(df["Low"].iloc[entry_i])
    c = float(df["Close"].iloc[entry_i])

    entry = o * (1 + P.slippage_oneway)

    tp_px = entry * (1 + P.tp)
    sl_px = entry * (1 - P.sl)

    # ギャップ（寄りで飛んだ場合）
    if o >= tp_px:
        exit_px = o * (1 - P.slippage_oneway)
        return exit_px / entry - 1.0
    if o <= sl_px:
        exit_px = o * (1 - P.slippage_oneway)
        return exit_px / entry - 1.0

    # 同日中の到達（保守的に SL→TP 順）
    if l <= sl_px:
        exit_px = sl_px * (1 - P.slippage_oneway)
        return exit_px / entry - 1.0
    if h >= tp_px:
        exit_px = tp_px * (1 - P.slippage_oneway)
        return exit_px / entry - 1.0

    # TIME（引け）
    exit_px = c * (1 - P.slippage_oneway)
    return exit_px / entry - 1.0


def run_strategy(ticker: str, market_ok: pd.Series) -> np.ndarray:
    df = load_data(ticker, P.lookback_years)
    if df.empty or len(df) < 260:
        return np.array([], dtype=float)

    df = compute_indicators(df)

    rets = []
    for i in range(3, len(df) - 1):
        day = pd.Timestamp(df.index[i]).normalize()

        # 市場フィルター
        if not market_ok_on(market_ok, day):
            continue

        # シグナル
        if (df["ret3"].iloc[i] <= P.pullback_ret3) and (df["RSI"].iloc[i] > P.rsi_min):
            entry_i = i + 1
            if entry_i >= len(df):
                continue

            # 資金で買えるか（寄り価格で判定）
            entry_open = float(df["Open"].iloc[entry_i])
            entry_price_sim = entry_open * (1 + P.slippage_oneway)
            if not is_affordable(entry_price_sim):
                continue

            ret = simulate_same_day_tp_sl(df, entry_i)
            rets.append(ret)

    return np.array(rets, dtype=float)


def equity_curve_from_returns(returns: np.ndarray) -> pd.Series:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return pd.Series(eq[1:])


def max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


if __name__ == "__main__":
    # テスト用ユニバース（必要に応じて増減）
    universe = [
        "8306.T","7203.T","9984.T","6857.T",
        "8604.T","4188.T","5401.T","4183.T",
        "5713.T","5020.T","9501.T","8591.T"
    ]

    mkt_ok = market_ok_series(P.lookback_years, P.market_ma)

    all_rets = []
    for t in universe:
        try:
            r = run_strategy(t, mkt_ok)
            if len(r):
                all_rets.extend(r.tolist())
        except Exception as e:
            print("skip", t, ":", e)

    all_rets = np.array(all_rets, dtype=float)

    if len(all_rets) == 0:
        print("No trades (all unaffordable or no signals)")
    else:
        win_rate = float(np.mean(all_rets > 0))
        avg_ret = float(np.mean(all_rets))
        total_sum = float(np.sum(all_rets))

        eq = equity_curve_from_returns(all_rets)
        mdd = max_drawdown(eq)

        print("Trades:", len(all_rets))
        print("Win rate:", round(win_rate, 3))
        print("Avg return:", round(avg_ret, 4))
        print("Total return (sum):", round(total_sum, 3))
        print("Max drawdown (seq):", round(mdd, 3))