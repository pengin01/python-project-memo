# pip install pandas numpy yfinance openpyxl xlrd requests tqdm

from __future__ import annotations
import io, re, math, time
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

    # Signal
    vol_ratio_th: float = 2.0

    # Trade
    hold_days: int = 4
    take_profit: Optional[float] = 0.08
    stop_loss: Optional[float] = 0.06

    fee_roundtrip: float = 0.001
    slippage_oneway: float = 0.0005

    # Filters
    min_avg_value20: float = 1_000_000_000  # 10億円/日 くらいからが現実的（小型を外す）

    # Universe exclusions (quick + effective)
    exclude_prefixes: Tuple[str, ...] = ("13", "14", "15", "16", "32")  # ETF/ETN/REIT帯をざっくり除外

    # Portfolio
    max_positions: int = 3  # 同時保有数（3でもOK）
    pick_top_k_per_day: int = 2  # 当日候補が多いときスコア上位から採用（=max_positionsと同じでOK）
    position_sizing: str = "equal"  # equalのみ実装（枠数で均等）

    # Runtime
    sleep_sec: float = 0.08
    max_tickers: Optional[int] = 300  # デバッグ用。例: 300。Noneで全件。


P = Params()

JPX_LIST_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"


# =========================
# JPX銘柄一覧
# =========================
def download_jpx_list_xls() -> bytes:
    html = requests.get(JPX_LIST_URL, timeout=30).text
    m = re.search(r'href="([^"]+\.(?:xls|xlsx))"', html, re.IGNORECASE)
    if not m:
        raise RuntimeError("JPX一覧のExcelリンクが見つかりません（ページ構造変更の可能性）。")
    url = m.group(1)
    if url.startswith("/"):
        url = "https://www.jpx.co.jp" + url
    return requests.get(url, timeout=60).content


def load_master(xls_bytes: bytes, exclude_prefixes: Tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(xls_bytes))

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
    out["code"] = out["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)

    # ETF/ETN/REIT帯を除外（個別株に寄せる）
    out = out[~out["code"].str.startswith(exclude_prefixes)]

    out = out.dropna().drop_duplicates(subset=["code"])
    out["ticker"] = out["code"] + ".T"
    return out[["ticker", "code", "name"]]


# =========================
# 指標
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

    # ブレイクの強さ（どれだけ上に抜けたか）
    df["breakout_pct"] = (df["Close"] / df["HH20_prev"]) - 1.0
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
# yfinance
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
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.dropna()


# =========================
# 1トレードを「現実寄り約定」で前方シミュレーション（TP/SL/ギャップ）
# =========================
def simulate_trade_from_entry(df: pd.DataFrame, entry_i: int, p: Params) -> Dict:
    idx = df.index
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

        # 保守的にSL優先
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
    net_ret = gross_ret - p.fee_roundtrip

    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "hold_days": int(exit_i - entry_i + 1),
        "reason": exit_reason,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
    }


# =========================
# 全銘柄から「候補トレード」を作る（銘柄ごと）
# =========================
def build_trade_candidates(master: pd.DataFrame, p: Params) -> pd.DataFrame:
    candidates: List[Dict] = []

    it = master
    if p.max_tickers is not None:
        it = it.head(p.max_tickers)

    for _, r in tqdm(it.iterrows(), total=len(it)):
        ticker, code, name = r["ticker"], r["code"], r["name"]

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

        # シグナル日 i → エントリーは i+1
        sig_idx = np.where(sig.values)[0]
        for i in sig_idx:
            entry_i = i + 1
            if entry_i >= len(df):
                continue

            # シグナルの“強さ”をスコア化（単純でOK）
            vol_ratio = float(df["vol_ratio"].iloc[i])
            breakout_pct = float(df["breakout_pct"].iloc[i])
            ma_slope = float(df["MA25_slope"].iloc[i]) if not math.isnan(float(df["MA25_slope"].iloc[i])) else 0.0

            trade = simulate_trade_from_entry(df, entry_i, p)

            candidates.append({
                "ticker": ticker,
                "code": code,
                "name": name,
                "signal_date": df.index[i],
                "vol_ratio": vol_ratio,
                "breakout_pct": breakout_pct,
                "ma25_slope": ma_slope,
                "avg_value20": avg_val,
                **trade,
            })

        time.sleep(p.sleep_sec)

    cand = pd.DataFrame(candidates)
    if cand.empty:
        return cand

    # スコア（zっぽく正規化。外れ値に強いようにrankで）
    cand["score"] = (
        cand["vol_ratio"].rank(pct=True) * 0.5 +
        cand["breakout_pct"].rank(pct=True) * 0.3 +
        cand["ma25_slope"].rank(pct=True) * 0.2
    )
    return cand


# =========================
# ポートフォリオシミュレーション
# - 同日候補が多いときは score 上位から最大N枠を埋める
# - 枠が埋まっている間は新規建てない
# - 結果は「クローズ時点で」資産更新（簡易）
# =========================
def portfolio_backtest(cand: pd.DataFrame, p: Params) -> Tuple[pd.DataFrame, Dict]:
    if cand.empty:
        return cand, {"trades": 0}

    # entry_date順 → 同日の中でscore降順
    cand = cand.sort_values(["entry_date", "score"], ascending=[True, False]).reset_index(drop=True)

    equity = 1.0
    equity_curve: List[Tuple[pd.Timestamp, float]] = []

    open_positions: List[Dict] = []
    trades: List[Dict] = []

    # 日付でグループ化して処理
    for entry_date, day_df in cand.groupby("entry_date", sort=True):
        # まず、その日以前に終了しているポジションをクローズ（exit_date <= entry_date）
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= entry_date:
                # クローズ
                equity *= (1.0 + pos["weighted_ret"])
                trades.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        equity_curve.append((entry_date, equity))

        # 空き枠
        slots = p.max_positions - len(open_positions)
        if slots <= 0:
            continue

        # 今日の候補から上位を採用
        picks = day_df.head(min(p.pick_top_k_per_day, slots))

        # 均等配分（枠数で等分）
        weight = 1.0 / p.max_positions

        for _, tr in picks.iterrows():
            open_positions.append({
                "ticker": tr["ticker"],
                "code": tr["code"],
                "name": tr["name"],
                "signal_date": tr["signal_date"],
                "entry_date": tr["entry_date"],
                "exit_date": tr["exit_date"],
                "reason": tr["reason"],
                "score": float(tr["score"]),
                "net_ret": float(tr["net_ret"]),
                "weighted_ret": float(weight * tr["net_ret"]),
                "hold_days": int(tr["hold_days"]),
            })

    # 最後に残ったポジションを全てクローズ（最後の日付で反映）
    if open_positions:
        last_date = pd.to_datetime(cand["entry_date"]).max()
        for pos in open_positions:
            equity *= (1.0 + pos["weighted_ret"])
            trades.append(pos)
        equity_curve.append((last_date, equity))

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).drop_duplicates("date").sort_values("date")

    # DD計算
    curve_df["peak"] = curve_df["equity"].cummax()
    curve_df["dd"] = curve_df["equity"] / curve_df["peak"] - 1.0

    stats = {
        "trades": int(len(trades_df)),
        "total_return": float(curve_df["equity"].iloc[-1] - 1.0) if not curve_df.empty else 0.0,
        "max_drawdown": float(curve_df["dd"].min()) if not curve_df.empty else 0.0,
        "avg_hold_days": float(trades_df["hold_days"].mean()) if not trades_df.empty else float("nan"),
        "win_rate": float((trades_df["net_ret"] > 0).mean()) if not trades_df.empty else float("nan"),
    }
    return trades_df, stats, curve_df


def main():
    xls = download_jpx_list_xls()
    master = load_master(xls, P.exclude_prefixes)

    print("Universe size:", len(master))

    cand = build_trade_candidates(master, P)
    if cand.empty:
        print("No candidates. Try lowering min_avg_value20 or vol_ratio_th.")
        return

    # 保存（候補）
    cand.to_csv("candidates.csv", index=False, encoding="utf-8-sig")

    trades_df, stats, curve_df = portfolio_backtest(cand, P)

    trades_df.to_csv("portfolio_trades.csv", index=False, encoding="utf-8-sig")
    curve_df.to_csv("portfolio_equity_curve.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([stats]).to_csv("portfolio_summary.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print("  candidates.csv")
    print("  portfolio_trades.csv")
    print("  portfolio_equity_curve.csv")
    print("  portfolio_summary.csv")

    print("\n== portfolio stats ==")
    for k, v in stats.items():
        print(k, v)

    print("\nTop trades (latest 10):")
    print(trades_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()