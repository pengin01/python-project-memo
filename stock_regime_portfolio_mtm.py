# pip install pandas numpy yfinance openpyxl xlrd requests tqdm
# python3 stock_regime_portfolio_mtm.py

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
# Params
# =========================
@dataclass
class Params:
    period: str = "1y"

    # ---- Engine 1 (Trend): high-win pullback on individual stocks ----
    tr_ret3_pullback: float = -0.02
    tr_rsi_min: float = 50.0
    tr_hold_days: int = 5
    tr_take_profit: float = 0.025
    tr_stop_loss: float = 0.02

    # ---- Engine 2 (Defensive): mean-reversion on TOPIX ETF 1306.T ----
    dv_ticker: str = "1306.T"
    dv_ret3_pullback: float = -0.02
    dv_rsi_max: float = 40.0
    dv_hold_days: int = 5
    dv_take_profit: float = 0.015
    dv_stop_loss: float = 0.01

    # ---- Costs ----
    slippage_oneway: float = 0.0015
    fee_roundtrip: float = 0.002

    # ---- Universe filters (Trend engine) ----
    min_avg_value20: float = 1_000_000_000  # 10億円/日
    exclude_prefixes: Tuple[str, ...] = ("13", "14", "15", "16", "32")  # ETF/ETN/REIT

    # ---- Market regime ----
    market_ticker: str = "^N225"
    market_ma_days: int = 200

    # ---- Portfolio (Trend engine) ----
    max_positions: int = 3
    pick_top_k_per_day: int = 3

    # ---- Runtime ----
    sleep_sec: float = 0.06
    max_tickers: Optional[int] = None  # debug: 300 etc. None=all


P = Params()
JPX_LIST_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"


# =========================
# Helpers
# =========================
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


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


def market_ok_series(p: Params) -> pd.Series:
    mkt = download_ohlcv(p.market_ticker, p.period)
    if mkt.empty or len(mkt) < p.market_ma_days + 10:
        return pd.Series(dtype=bool)
    close = mkt["Close"].astype(float)
    ma = close.rolling(p.market_ma_days).mean()
    ok = (close > ma).fillna(False)
    ok.name = "market_ok"
    return ok


def market_ok_on(market_ok: pd.Series, day: pd.Timestamp) -> bool:
    if market_ok.empty:
        return True
    if day in market_ok.index:
        return bool(market_ok.loc[day])
    mk = market_ok.loc[:day]
    if mk.empty:
        return False
    return bool(mk.iloc[-1])


# =========================
# JPX Master (for Trend engine universe)
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
    out = out[~out["code"].str.startswith(exclude_prefixes)]
    out = out.dropna().drop_duplicates(subset=["code"])
    out["ticker"] = out["code"] + ".T"
    return out[["ticker", "code", "name"]]


# =========================
# Features & signals
# =========================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    df["MA5"] = close.rolling(5).mean()
    df["MA25"] = close.rolling(25).mean()
    df["ret3"] = close.pct_change(3)
    df["RSI"] = rsi_wilder(close, 14)
    df["Value"] = close * df["Volume"]
    df["AVG_Value20"] = df["Value"].rolling(20).mean()
    df["trend_gap"] = (close / df["MA25"]) - 1.0
    df["ma25_slope"] = df["MA25"].diff(5)
    return df


def signal_trend(df: pd.DataFrame, p: Params) -> pd.Series:
    # Trend engine: uptrend + shallow pullback + RSI>min
    base = (
        (df["Close"] > df["MA25"]) &
        (df["MA5"] > df["MA25"]) &
        (df["ret3"] <= p.tr_ret3_pullback) &
        (df["RSI"] > p.tr_rsi_min)
    )
    return base.fillna(False)


def signal_defensive(df: pd.DataFrame, p: Params) -> pd.Series:
    # Defensive engine: oversold pullback on ETF + RSI<max
    base = (
        (df["ret3"] <= p.dv_ret3_pullback) &
        (df["RSI"] < p.dv_rsi_max)
    )
    return base.fillna(False)


# =========================
# Strict fill simulator (TP/SL + gap + costs)
# =========================
def simulate_trade_strict(
    df: pd.DataFrame,
    entry_i: int,
    tp: float,
    sl: float,
    hold_days: int,
    slippage_oneway: float,
    fee_roundtrip: float,
) -> Dict:
    idx = df.index
    entry_open = float(df["Open"].iloc[entry_i])
    entry_price = entry_open * (1 + slippage_oneway)

    tp_price = entry_price * (1 + tp)
    sl_price = entry_price * (1 - sl)

    exit_i = min(entry_i + hold_days - 1, len(df) - 1)
    exit_reason = "TIME"
    exit_price = float(df["Close"].iloc[exit_i]) * (1 - slippage_oneway)

    for j in range(entry_i, exit_i + 1):
        o = float(df["Open"].iloc[j])
        h = float(df["High"].iloc[j])
        l = float(df["Low"].iloc[j])

        # gap
        if o >= tp_price:
            exit_i = j
            exit_reason = "TP_GAP"
            exit_price = o * (1 - slippage_oneway)
            break
        if o <= sl_price:
            exit_i = j
            exit_reason = "SL_GAP"
            exit_price = o * (1 - slippage_oneway)
            break

        # intraday (conservative: SL first)
        if l <= sl_price:
            exit_i = j
            exit_reason = "SL"
            exit_price = sl_price * (1 - slippage_oneway)
            break
        if h >= tp_price:
            exit_i = j
            exit_reason = "TP"
            exit_price = tp_price * (1 - slippage_oneway)
            break

    gross = exit_price / entry_price - 1.0
    net = gross - fee_roundtrip

    return {
        "entry_date": idx[entry_i],
        "exit_date": idx[exit_i],
        "hold_days": int(exit_i - entry_i + 1),
        "reason": exit_reason,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "net_ret": float(net),
    }


# =========================
# Candidate builders
# =========================
def build_trend_candidates(master: pd.DataFrame, market_ok: pd.Series, p: Params) -> pd.DataFrame:
    rows: List[Dict] = []
    it = master if p.max_tickers is None else master.head(p.max_tickers)

    for _, r in tqdm(it.iterrows(), total=len(it), desc="scan trend universe"):
        ticker, code, name = r["ticker"], r["code"], r["name"]

        raw = download_ohlcv(ticker, p.period)
        if raw.empty or len(raw) < 80:
            time.sleep(p.sleep_sec)
            continue

        df = compute_features(raw)

        avg_val = float(df["AVG_Value20"].iloc[-1]) if not math.isnan(float(df["AVG_Value20"].iloc[-1])) else 0.0
        if avg_val < p.min_avg_value20:
            time.sleep(p.sleep_sec)
            continue

        sig = signal_trend(df, p)
        sig_idx = np.where(sig.values)[0]

        for i in sig_idx:
            entry_i = i + 1
            if entry_i >= len(df):
                continue

            # bounce confirm at entry day (bullish candle)
            if float(df["Close"].iloc[entry_i]) <= float(df["Open"].iloc[entry_i]):
                continue

            entry_date = df.index[entry_i]
            if not market_ok_on(market_ok, entry_date):
                continue  # Trend engine only in trend regime

            tr = simulate_trade_strict(
                df, entry_i,
                tp=p.tr_take_profit, sl=p.tr_stop_loss, hold_days=p.tr_hold_days,
                slippage_oneway=p.slippage_oneway, fee_roundtrip=p.fee_roundtrip
            )

            score = float(df["trend_gap"].iloc[i]) * 0.6 + float(df["ma25_slope"].iloc[i]) * 0.4

            rows.append({
                "engine": "TREND",
                "ticker": ticker,
                "code": code,
                "name": name,
                "signal_date": df.index[i],
                **tr,
                "score": score,
                "avg_value20": avg_val,
            })

        time.sleep(p.sleep_sec)

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    cand["score_rank"] = cand["score"].rank(pct=True)
    return cand


def build_defensive_candidates(market_ok: pd.Series, p: Params) -> pd.DataFrame:
    # Defensive engine trades ONLY when market is NOT ok (<= MA200)
    raw = download_ohlcv(p.dv_ticker, p.period)
    if raw.empty or len(raw) < 80:
        return pd.DataFrame()

    df = compute_features(raw)
    sig = signal_defensive(df, p)
    sig_idx = np.where(sig.values)[0]

    rows: List[Dict] = []
    for i in sig_idx:
        entry_i = i + 1
        if entry_i >= len(df):
            continue

        entry_date = df.index[entry_i]
        if market_ok_on(market_ok, entry_date):
            continue  # defensive only in non-trend regime

        # bounce confirm at entry day (bullish candle)
        if float(df["Close"].iloc[entry_i]) <= float(df["Open"].iloc[entry_i]):
            continue

        tr = simulate_trade_strict(
            df, entry_i,
            tp=p.dv_take_profit, sl=p.dv_stop_loss, hold_days=p.dv_hold_days,
            slippage_oneway=p.slippage_oneway, fee_roundtrip=p.fee_roundtrip
        )

        # For defensive, score can simply prefer more oversold (lower RSI) and bigger pullback
        score = (-float(df["RSI"].iloc[i])) * 0.7 + (-float(df["ret3"].iloc[i])) * 0.3

        rows.append({
            "engine": "DEFENSIVE",
            "ticker": p.dv_ticker,
            "code": "1306",
            "name": "TOPIX ETF",
            "signal_date": df.index[i],
            **tr,
            "score": score,
            "avg_value20": float(df["AVG_Value20"].iloc[i]) if "AVG_Value20" in df.columns else np.nan,
        })

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    cand["score_rank"] = cand["score"].rank(pct=True)
    return cand


# =========================
# MTM Backtester (daily mark-to-market)
# =========================
def build_price_cache(tickers: List[str], period: str) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    for t in tqdm(sorted(set(tickers)), desc="download prices"):
        df = download_ohlcv(t, period)
        if not df.empty:
            cache[t] = df
        time.sleep(0.03)
    return cache


def get_close(cache: Dict[str, pd.DataFrame], ticker: str, day: pd.Timestamp) -> Optional[float]:
    df = cache.get(ticker)
    if df is None or df.empty:
        return None
    if day in df.index:
        return float(df.loc[day, "Close"])
    prev = df.loc[:day]
    if prev.empty:
        return None
    return float(prev["Close"].iloc[-1])


def mtm_regime_portfolio(
    candidates: pd.DataFrame,
    market_days: pd.DatetimeIndex,
    market_ok: pd.Series,
    p: Params
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:

    cand = candidates.sort_values(["entry_date", "score_rank"], ascending=[True, False]).reset_index(drop=True)
    cache = build_price_cache(cand["ticker"].tolist(), p.period)

    by_day = {d: g.copy() for d, g in cand.groupby("entry_date", sort=True)}

    cash = 1.0
    open_pos: List[Dict] = []
    executed: List[Dict] = []
    curve: List[Tuple[pd.Timestamp, float, str, int]] = []  # date,equity,regime,open_count

    # equal-risk slots for TREND engine
    slot_frac_trend = 1.0 / p.max_positions

    for day in market_days:
        regime = "TREND" if market_ok_on(market_ok, day) else "DEFENSIVE"

        # exits
        still = []
        for pos in open_pos:
            if pos["exit_date"] <= day:
                proceeds = pos["shares"] * pos["exit_price"]
                proceeds *= (1.0 - p.fee_roundtrip)  # fee on exit proceeds
                cash += proceeds
                executed.append({k: pos[k] for k in pos if k != "shares"})
            else:
                still.append(pos)
        open_pos = still

        # entries
        todays = by_day.get(day)
        if todays is not None and not todays.empty:
            # only allow candidates matching today's regime engine
            todays = todays[todays["engine"] == regime]
            if not todays.empty:
                if regime == "TREND":
                    slots = p.max_positions - len(open_pos)
                    if slots > 0:
                        picks = todays.head(min(p.pick_top_k_per_day, slots))

                        for _, tr in picks.iterrows():
                            # current equity
                            mtm_positions = 0.0
                            for pos in open_pos:
                                c = get_close(cache, pos["ticker"], day)
                                if c is not None:
                                    mtm_positions += pos["shares"] * c
                            equity_now = cash + mtm_positions

                            alloc = equity_now * slot_frac_trend
                            if alloc <= 0 or cash <= 0:
                                continue
                            alloc = min(alloc, cash)

                            entry_price = float(tr["entry_price"])
                            shares = alloc / entry_price
                            cash -= alloc

                            open_pos.append({
                                "engine": "TREND",
                                "ticker": tr["ticker"],
                                "code": tr["code"],
                                "name": tr["name"],
                                "signal_date": tr["signal_date"],
                                "entry_date": tr["entry_date"],
                                "exit_date": tr["exit_date"],
                                "reason": tr["reason"],
                                "score_rank": float(tr["score_rank"]),
                                "entry_price": float(tr["entry_price"]),
                                "exit_price": float(tr["exit_price"]),
                                "net_ret": float(tr["net_ret"]),
                                "hold_days": int(tr["hold_days"]),
                                "shares": float(shares),
                            })

                else:
                    # DEFENSIVE: 1306 only, keep it simple: at most 1 position
                    if len(open_pos) == 0:
                        tr = todays.head(1).iloc[0]
                        # invest all cash (100%) for defensive leg
                        if cash > 0:
                            entry_price = float(tr["entry_price"])
                            alloc = cash * 0.5     # 50%だけ投入
                            shares = alloc / entry_price
                            cash -= alloc

                            open_pos.append({
                                "engine": "DEFENSIVE",
                                "ticker": tr["ticker"],
                                "code": tr["code"],
                                "name": tr["name"],
                                "signal_date": tr["signal_date"],
                                "entry_date": tr["entry_date"],
                                "exit_date": tr["exit_date"],
                                "reason": tr["reason"],
                                "score_rank": float(tr["score_rank"]),
                                "entry_price": float(tr["entry_price"]),
                                "exit_price": float(tr["exit_price"]),
                                "net_ret": float(tr["net_ret"]),
                                "hold_days": int(tr["hold_days"]),
                                "shares": float(shares),
                            })

        # MTM equity
        mtm = 0.0
        for pos in open_pos:
            c = get_close(cache, pos["ticker"], day)
            if c is not None:
                mtm += pos["shares"] * c
        equity = cash + mtm
        curve.append((day, equity, regime, len(open_pos)))

    trades_df = pd.DataFrame(executed)
    curve_df = pd.DataFrame(curve, columns=["date", "equity", "regime", "open_positions"]).drop_duplicates("date").sort_values("date")

    if not curve_df.empty:
        curve_df["peak"] = curve_df["equity"].cummax()
        curve_df["dd"] = curve_df["equity"] / curve_df["peak"] - 1.0
        max_dd = float(curve_df["dd"].min())
        total_ret = float(curve_df["equity"].iloc[-1] - 1.0)
    else:
        max_dd = 0.0
        total_ret = 0.0

    stats = {
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["net_ret"] > 0).mean()) if not trades_df.empty else float("nan"),
        "avg_hold_days": float(trades_df["hold_days"].mean()) if not trades_df.empty else float("nan"),
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "trend_max_positions": p.max_positions,
        "trend_tp": p.tr_take_profit,
        "trend_sl": p.tr_stop_loss,
        "defensive_ticker": p.dv_ticker,
        "def_tp": p.dv_take_profit,
        "def_sl": p.dv_stop_loss,
        "market_filter": f"{p.market_ticker} Close > MA{p.market_ma_days}",
        "min_avg_value20": p.min_avg_value20,
        "fee_roundtrip": p.fee_roundtrip,
        "slippage_oneway": p.slippage_oneway,
        "equity_curve": "daily MTM (cash + positions at Close), regime-switched",
    }

    return trades_df, curve_df, stats


# =========================
# Main
# =========================
def main():
    # Market regime
    market_ok = market_ok_series(P)

    # Market days for MTM timeline
    mkt = download_ohlcv(P.market_ticker, P.period)
    market_days = mkt.index if not mkt.empty else pd.DatetimeIndex([])

    # Build trend candidates
    xls = download_jpx_list_xls()
    master = load_master(xls, P.exclude_prefixes)
    if P.max_tickers is not None:
        master = master.head(P.max_tickers)

    print("Universe size (trend):", len(master))

    trend_cand = build_trend_candidates(master, market_ok, P)

    # Build defensive candidates
    def_cand = build_defensive_candidates(market_ok, P)

    all_cand = pd.concat([trend_cand, def_cand], ignore_index=True) if not trend_cand.empty or not def_cand.empty else pd.DataFrame()
    if all_cand.empty:
        print("No candidates. Try relaxing parameters.")
        return

    # fallback market days
    if market_days.empty:
        market_days = pd.DatetimeIndex(sorted(pd.to_datetime(all_cand["entry_date"]).unique()))

    # Save candidates
    all_cand.to_csv("regime_candidates.csv", index=False, encoding="utf-8-sig")

    # Run MTM portfolio
    trades_df, curve_df, stats = mtm_regime_portfolio(all_cand, market_days, market_ok, P)

    trades_df.to_csv("regime_trades_mtm.csv", index=False, encoding="utf-8-sig")
    curve_df.to_csv("regime_equity_curve_mtm.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([stats]).to_csv("regime_summary_mtm.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print("  regime_candidates.csv")
    print("  regime_trades_mtm.csv")
    print("  regime_equity_curve_mtm.csv")
    print("  regime_summary_mtm.csv")

    print("\n== regime portfolio stats (MTM) ==")
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
