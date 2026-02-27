# pip install pandas numpy yfinance

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "7203.T"
PERIOD = "5y"

TP = 0.03
SL = 0.03
HOLD_DAYS = 5


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (EMA-like smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest(df: pd.DataFrame) -> pd.Series:
    trades = []
    i = 60  # 指標が揃うまで待つ

    while i < len(df) - HOLD_DAYS - 1:
        cond = (
            (df["Close"].iloc[i] > df["MA25"].iloc[i]) &
            (df["MA5"].iloc[i] > df["MA25"].iloc[i]) &
            (df["ret3"].iloc[i] <= -0.03) &
            (df["RSI"].iloc[i] > 50)
        )

        if not bool(cond):
            i += 1
            continue

        entry_i = i + 1
        entry_price = float(df["Open"].iloc[entry_i])
        entry_date = df.index[entry_i]

        exit_i = min(entry_i + HOLD_DAYS - 1, len(df) - 1)
        exit_reason = "TIME"

        for j in range(entry_i, exit_i + 1):
            high = float(df["High"].iloc[j])
            low = float(df["Low"].iloc[j])

            if high >= entry_price * (1 + TP):
                exit_i = j
                exit_reason = "TP"
                break

            if low <= entry_price * (1 - SL):
                exit_i = j
                exit_reason = "SL"
                break

        exit_price = float(df["Close"].iloc[exit_i])
        exit_date = df.index[exit_i]

        ret = exit_price / entry_price - 1.0
        trades.append(ret)

        # 同時保有なし
        i = exit_i + 1

    return pd.Series(trades, name="ret")


def main():
    df = yf.download(TICKER, period=PERIOD, interval="1d", auto_adjust=False, progress=False)

    # MultiIndex対策（念のため）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    # ここで「Series化」しておく（形状事故の予防）
    close = df["Close"].astype(float)

    df["MA5"] = close.rolling(5).mean()
    df["MA25"] = close.rolling(25).mean()
    df["ret3"] = close.pct_change(3)
    df["RSI"] = rsi_wilder(close, 14)

    trades = backtest(df)

    print("trades:", int(len(trades)))
    if len(trades) == 0:
        print("win_rate: n/a (no trades)")
        return

    print("win_rate:", float((trades > 0).mean()))
    print("avg_return:", float(trades.mean()))
    print("total_return:", float((1 + trades).prod() - 1))


if __name__ == "__main__":
    main()