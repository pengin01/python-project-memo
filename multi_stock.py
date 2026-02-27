# pip install pandas numpy yfinance

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# 設定
# =========================
@dataclass
class Params:
    period: str = "5y"
    vol_ratio_th: float = 2.0

    hold_days: int = 3
    take_profit: Optional[float] = 0.08   # 例: 0.08 = +8%
    stop_loss: Optional[float] = 0.06     # 例: 0.06 = -6%（Bが良かったので初期値は6%）

    fee_rate_roundtrip: float = 0.001     # 往復コスト（ざっくり）
    slippage_oneway: float = 0.0005       # 片道スリッページ（ざっくり）

    # 流動性フィルタ（任意：小さすぎる銘柄を除外）
    min_avg_value20: float = 50_000_000   # 例: 5000万円/日


PARAMS = Params()


# =========================
# インジケータ
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["VMA20"] = df["Volume"].rolling(20).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["HH20_prev"] = df["High"].rolling(20).max().shift(1)  # 今日を含めない
    df["MA25_slope"] = df["MA25"].diff(5)
    df["vol_ratio"] = df["Volume"] / df["VMA20"]

    # 流動性（売買代金の簡易）
    df["Value"] = df["Close"] * df["Volume"]
    df["AVG_Value20"] = df["Value"].rolling(20).mean()
    return df


def generate_signal(df: pd.DataFrame, vol_ratio_th: float) -> pd.Series:
    cond = (
        (df["vol_ratio"] >= vol_ratio_th) &
        (df["Close"] >= df["HH20_prev"]) &
        (df["Close"] >= df["MA25"]) &
        (df["MA25_slope"] > 0)
    )
    return cond.fillna(False)


# =========================
# yfinance 取得
# =========================
def download_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    # MultiIndex対策
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.dropna()
    return raw


# =========================
# バックテスト（現実寄り：TP/SL価格約定＋ギャップ考慮）
# =========================
def backtest_one(
    df: pd.DataFrame,
    signal: pd.Series,
    params: Params,
) -> Tuple[pd.DataFrame, Dict]:

    trades: List[Dict] = []
    idx = df.index
    i = 0

    # 同時保有なし（銘柄内）
    while i < len(df) - 1:
        if not bool(signal.iloc[i]):
            i += 1
            continue

        entry_i = i + 1
        if entry_i >= len(df):
            break

        # エントリー（翌日寄り）＋スリッページ
        entry_open = float(df["Open"].iloc[entry_i])
        entry_price = entry_open * (1 + params.slippage_oneway)
        entry_date = idx[entry_i]

        # エグジット候補最終日
        exit_i = min(entry_i + params.hold_days - 1, len(df) - 1)
        exit_reason = "TIME"
        exit_price = float(df["Close"].iloc[exit_i]) * (1 - params.slippage_oneway)  # TIMEの仮

        tp_price = entry_price * (1 + params.take_profit) if params.take_profit is not None else None
        sl_price = entry_price * (1 - params.stop_loss) if params.stop_loss is not None else None

        # 期間内を走査して、先にヒットしたものを採用
        for j in range(entry_i, exit_i + 1):
            o = float(df["Open"].iloc[j])
            h = float(df["High"].iloc[j])
            l = float(df["Low"].iloc[j])

            # ギャップで飛び越え
            if sl_price is not None and o <= sl_price:
                exit_i = j
                exit_reason = "SL_GAP"
                exit_price = o * (1 - params.slippage_oneway)
                break

            if tp_price is not None and o >= tp_price:
                exit_i = j
                exit_reason = "TP_GAP"
                exit_price = o * (1 - params.slippage_oneway)
                break

            # 日中ヒット（保守的にSL優先）
            if sl_price is not None and l <= sl_price:
                exit_i = j
                exit_reason = "SL"
                exit_price = sl_price * (1 - params.slippage_oneway)
                break

            if tp_price is not None and h >= tp_price:
                exit_i = j
                exit_reason = "TP"
                exit_price = tp_price * (1 - params.slippage_oneway)
                break

        exit_date = idx[exit_i]

        gross_ret = (exit_price / entry_price) - 1.0
        net_ret = gross_ret - params.fee_rate_roundtrip

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "hold_days": int(exit_i - entry_i + 1),
            "reason": exit_reason,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })

        # 次の探索位置：クローズ翌日から
        i = exit_i + 1

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, {"trades": 0}

    equity = (1 + trades_df["net_ret"]).cumprod()
    dd = equity / equity.cummax() - 1

    wins = trades_df.loc[trades_df["net_ret"] > 0, "net_ret"]
    losses = trades_df.loc[trades_df["net_ret"] <= 0, "net_ret"]

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["net_ret"] > 0).mean()),
        "avg_net_ret": float(trades_df["net_ret"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(dd.min()),
        "avg_hold_days": float(trades_df["hold_days"].mean()),
    }
    return trades_df, stats


# =========================
# 複数銘柄実行
# =========================
def run_multi(tickers: List[str], params: Params) -> None:
    summary_rows: List[Dict] = []
    all_trades: List[pd.DataFrame] = []

    for t in tickers:
        print(f"\n--- {t} ---")

        raw = download_ohlcv(t, params.period)
        if raw.empty or len(raw) < 80:
            print("skip: no data or too short")
            continue

        df = compute_indicators(raw)

        # 流動性フィルタ（任意）
        avg_val = float(df["AVG_Value20"].iloc[-1]) if not math.isnan(float(df["AVG_Value20"].iloc[-1])) else 0.0
        if avg_val < params.min_avg_value20:
            print(f"skip: low liquidity avg_value20={avg_val:,.0f}")
            continue

        sig = generate_signal(df, params.vol_ratio_th)
        sig_count = int(sig.sum())
        print("signal_count:", sig_count)

        trades_df, stats = backtest_one(df, sig, params)

        row = {
            "ticker": t,
            "signal_count": sig_count,
            "avg_value20": avg_val,
            **stats,
        }
        summary_rows.append(row)

        if not trades_df.empty:
            trades_df = trades_df.copy()
            trades_df.insert(0, "ticker", t)
            all_trades.append(trades_df)

        # 負荷軽減（レート制限回避）
        time.sleep(0.15)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        print("\nNo results.")
        return

    summary = summary.sort_values(["total_return", "win_rate", "trades"], ascending=False)

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary.to_csv("summary.csv", index=False, encoding="utf-8-sig")
    trades_all.to_csv("trades.csv", index=False, encoding="utf-8-sig")

    print("\nSaved: summary.csv, trades.csv")
    print("\nTop 20:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    # まずは手動リストでOK。好きなだけ追加してください（日本株は .T）
    tickers = [
        "6526.T",  # ソシオネクスト（動作確認で使ってた）
        "7203.T",  # トヨタ
        "7974.T",  # 任天堂
        "9984.T",  # ソフトバンクG
        "8306.T",  # 三菱UFJ
        "9432.T",  # NTT
        "8035.T",  # 東京エレクトロン
        "6857.T",  # アドバンテスト
        "3992.T",  #ニーズウェル
        "8267.T",  #イオン
    ]

    run_multi(tickers, PARAMS)