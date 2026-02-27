# pip install pandas numpy yfinance openpyxl requests tqdm

from __future__ import annotations
import io
import math
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm


# =========================
# 設定
# =========================
@dataclass
class Params:
    period: str = "5y"

    # シグナル条件
    vol_ratio_th: float = 2.0

    # 売買ルール
    hold_days: int = 3
    take_profit: Optional[float] = 0.08   # +8%
    stop_loss: Optional[float] = 0.06     # -6%（あなたの結果で効いてたので初期値6%）

    # コスト（ざっくり）
    fee_rate_roundtrip: float = 0.001     # 往復コスト
    slippage_oneway: float = 0.0005       # 片道スリッページ

    # 流動性フィルタ（低流動性を除外）
    min_avg_value20: float = 500_000_000  # 例: 5億円/日（必要なら上げる）

    # yfinance負荷対策
    sleep_sec: float = 0.12               # 速すぎると止まるので控えめに
    max_tickers: Optional[int] = 200     # 例: 300 で試運転。Noneで全件。


PARAMS = Params()

# JPX 上場銘柄一覧（英語ページ。Excelリンクを拾ってDL）
JPX_LIST_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"


# =========================
# JPX 銘柄一覧取得
# =========================
def download_jpx_list_xls() -> bytes:
    html = requests.get(JPX_LIST_URL, timeout=30).text
    m = re.search(r'href="([^"]+\.(?:xls|xlsx))"', html, re.IGNORECASE)
    if not m:
        raise RuntimeError("JPXの上場銘柄一覧Excelリンクが見つかりません（ページ構造が変わった可能性）。")
    xls_url = m.group(1)
    if xls_url.startswith("/"):
        xls_url = "https://www.jpx.co.jp" + xls_url
    return requests.get(xls_url, timeout=60).content


def load_codes_from_jpx_xls(xls_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(xls_bytes))

    # 列名はJPXファイル更新で変わることがあるので候補を複数用意
    code_candidates = ["Code", "Local Code", "Security Code", "コード"]
    name_candidates = ["Company Name", "Name", "銘柄名", "銘柄名称", "Company name"]

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    code_col = pick(code_candidates)
    name_col = pick(name_candidates)

    if code_col is None:
        raise RuntimeError(f"銘柄コード列が見つかりません。列一覧: {list(df.columns)}")
    if name_col is None:
        name_col = code_col

    out = df[[code_col, name_col]].copy()
    out.columns = ["code", "name"]
    out["code"] = (
        out["code"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )
    out = out.dropna().drop_duplicates(subset=["code"])
    out["ticker"] = out["code"] + ".T"
    # ETFっぽいコード帯を除外（保険）
    etf_prefixes = ("13", "14", "15", "16")
    out = out[~out["code"].str.startswith(etf_prefixes)]

    return out[["ticker", "code", "name"]]

# =========================
# 指標 & シグナル
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["VMA20"] = df["Volume"].rolling(20).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["HH20_prev"] = df["High"].rolling(20).max().shift(1)  # 今日を含めない
    df["MA25_slope"] = df["MA25"].diff(5)
    df["vol_ratio"] = df["Volume"] / df["VMA20"]

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

    return raw.dropna()


# =========================
# バックテスト（TP/SL価格約定 + ギャップ考慮）
# =========================
def backtest_one(
    df: pd.DataFrame,
    signal: pd.Series,
    p: Params,
) -> Tuple[pd.DataFrame, Dict]:

    trades: List[Dict] = []
    idx = df.index
    i = 0

    while i < len(df) - 1:
        if not bool(signal.iloc[i]):
            i += 1
            continue

        entry_i = i + 1
        if entry_i >= len(df):
            break

        entry_open = float(df["Open"].iloc[entry_i])
        entry_price = entry_open * (1 + p.slippage_oneway)
        entry_date = idx[entry_i]

        exit_i = min(entry_i + p.hold_days - 1, len(df) - 1)
        exit_reason = "TIME"
        exit_price = float(df["Close"].iloc[exit_i]) * (1 - p.slippage_oneway)

        tp_price = entry_price * (1 + p.take_profit) if p.take_profit is not None else None
        sl_price = entry_price * (1 - p.stop_loss) if p.stop_loss is not None else None

        for j in range(entry_i, exit_i + 1):
            o = float(df["Open"].iloc[j])
            h = float(df["High"].iloc[j])
            l = float(df["Low"].iloc[j])

            # ギャップ（始値で飛び越え）
            if sl_price is not None and o <= sl_price:
                exit_i = j
                exit_reason = "SL_GAP"
                exit_price = o * (1 - p.slippage_oneway)
                break

            if tp_price is not None and o >= tp_price:
                exit_i = j
                exit_reason = "TP_GAP"
                exit_price = o * (1 - p.slippage_oneway)
                break

            # 日中ヒット（保守的にSL優先）
            if sl_price is not None and l <= sl_price:
                exit_i = j
                exit_reason = "SL"
                exit_price = sl_price * (1 - p.slippage_oneway)
                break

            if tp_price is not None and h >= tp_price:
                exit_i = j
                exit_reason = "TP"
                exit_price = tp_price * (1 - p.slippage_oneway)
                break

        exit_date = idx[exit_i]

        gross_ret = (exit_price / entry_price) - 1.0
        net_ret = gross_ret - p.fee_rate_roundtrip

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

        i = exit_i + 1  # 同時保有なし（銘柄内）

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
# 全銘柄実行
# =========================
def run_all(p: Params) -> None:
    xls = download_jpx_list_xls()
    master = load_codes_from_jpx_xls(xls)

    if p.max_tickers is not None:
        master = master.head(p.max_tickers)

    summary_rows: List[Dict] = []
    trades_all: List[pd.DataFrame] = []

    for _, r in tqdm(master.iterrows(), total=len(master)):
        ticker = r["ticker"]
        code = r["code"]
        name = r["name"]

        raw = download_ohlcv(ticker, p.period)
        if raw.empty or len(raw) < 80:
            time.sleep(p.sleep_sec)
            continue

        df = compute_indicators(raw)

        avg_val = float(df["AVG_Value20"].iloc[-1]) if not math.isnan(float(df["AVG_Value20"].iloc[-1])) else 0.0
        if avg_val < p.min_avg_value20:
            time.sleep(p.sleep_sec)
            continue

        sig = generate_signal(df, p.vol_ratio_th)
        sig_count = int(sig.sum())

        trades_df, stats = backtest_one(df, sig, p)

        summary_rows.append({
            "ticker": ticker,
            "code": code,
            "name": name,
            "signal_count": sig_count,
            "avg_value20": avg_val,
            **stats,
        })

        if not trades_df.empty:
            tdf = trades_df.copy()
            tdf.insert(0, "ticker", ticker)
            tdf.insert(1, "code", code)
            tdf.insert(2, "name", name)
            trades_all.append(tdf)

        time.sleep(p.sleep_sec)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        print("No results (all filtered/failed). Try lowering min_avg_value20 or max_tickers for debug.")
        return

    trades_df_all = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()

    # ランキング作成（回数が少なすぎると運が乗るので最低回数フィルタも用意）
    summary["trades"] = summary["trades"].fillna(0).astype(int)

    summary_all = summary.sort_values(["total_return", "win_rate", "trades"], ascending=False)
    summary_10 = summary[summary["trades"] >= 10].sort_values(["total_return", "win_rate"], ascending=False)
    summary_20 = summary[summary["trades"] >= 20].sort_values(["total_return", "win_rate"], ascending=False)

    summary_all.to_csv("all_summary.csv", index=False, encoding="utf-8-sig")
    summary_10.to_csv("rank_trades_ge_10.csv", index=False, encoding="utf-8-sig")
    summary_20.to_csv("rank_trades_ge_20.csv", index=False, encoding="utf-8-sig")
    trades_df_all.to_csv("all_trades.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print("  all_summary.csv")
    print("  rank_trades_ge_10.csv")
    print("  rank_trades_ge_20.csv")
    print("  all_trades.csv")

    print("\nTop 20 (trades>=10):")
    print(summary_10.head(20).to_string(index=False))

    print("\nTop 20 (trades>=20):")
    print(summary_20.head(20).to_string(index=False))


if __name__ == "__main__":
    run_all(PARAMS)