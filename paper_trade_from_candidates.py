from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd


@dataclass
class PaperParams:
    initial_cash_yen: int = 80000

    # 手動想定：TREND日は2銘柄まで（各50%）
    trend_max_positions_live: int = 1
    trend_lot_size: int = 100

    # 1306は1口扱い
    defensive_lot_size: int = 1

    # A固定：当日完結（entry_date==exit_date）だけ採用
    close_out_same_day_only: bool = True

    candidates_csv: str = "regime_candidates.csv"
    out_today_csv: str = "paper_today_orders.csv"
    out_trades_csv: str = "paper_trades.csv"
    out_summary_csv: str = "paper_summary.csv"


P = PaperParams()


def _today() -> pd.Timestamp:
    return pd.Timestamp(datetime.now().date()).normalize()


def _to_dates(cand: pd.DataFrame) -> pd.DataFrame:
    cand = cand.copy()
    for c in ["signal_date", "entry_date", "exit_date"]:
        if c in cand.columns:
            cand[c] = pd.to_datetime(cand[c]).dt.normalize()
    return cand


def _next_entry_day(cand: pd.DataFrame) -> pd.Timestamp | None:
    today = _today()
    future = cand.loc[cand["entry_date"] >= today, "entry_date"].sort_values().unique()
    if len(future) == 0:
        return None
    return pd.Timestamp(future[0]).normalize()


def _affordable(df: pd.DataFrame, alloc_yen: float, lot_size: int) -> pd.DataFrame:
    df = df.copy()
    df["lot_size"] = int(lot_size)
    df["min_cost"] = df["entry_price"] * df["lot_size"]
    return df[df["min_cost"] <= alloc_yen].copy()


def build_today_orders(cand: pd.DataFrame, p: PaperParams) -> pd.DataFrame:
    day = _next_entry_day(cand)
    if day is None:
        return pd.DataFrame()

    today = cand[cand["entry_date"] == day].copy()
    if today.empty:
        return pd.DataFrame()

    # A固定：当日完結のみ
    if p.close_out_same_day_only:
        today = today[today["exit_date"] == today["entry_date"]].copy()
        if today.empty:
            return pd.DataFrame()

    today = today.sort_values("score_rank", ascending=False)

    orders: List[pd.DataFrame] = []

    # TREND：最大2銘柄、各50%
    tr = today[today["engine"] == "TREND"].copy()
    if not tr.empty:
        alloc = p.initial_cash_yen / p.trend_max_positions_live
        tr = _affordable(tr, alloc, p.trend_lot_size)
        tr = tr.head(p.trend_max_positions_live)
        if not tr.empty:
            tr["alloc_yen"] = alloc
            tr["plan"] = f"TREND up to {p.trend_max_positions_live}"
            orders.append(tr)

    # DEFENSIVE：1306のみ、100%
    dv = today[today["engine"] == "DEFENSIVE"].copy()
    if not dv.empty:
        alloc = p.initial_cash_yen
        dv = _affordable(dv, alloc, p.defensive_lot_size)
        dv = dv.head(1)
        if not dv.empty:
            dv["alloc_yen"] = alloc
            dv["plan"] = "DEFENSIVE 1306"
            orders.append(dv)

    if not orders:
        return pd.DataFrame()

    out = pd.concat(orders, ignore_index=True)
    out["paper_day"] = day
    return out


def calc_qty(row) -> int:
    lot = int(row["lot_size"])
    alloc = float(row["alloc_yen"])
    px = float(row["entry_price"])
    if lot <= 0 or px <= 0:
        return 0
    units = int(alloc // (px * lot))
    return units * lot


def append_trades(orders: pd.DataFrame, p: PaperParams) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame()

    trades = orders.copy()
    trades["qty"] = trades.apply(calc_qty, axis=1)
    trades = trades[trades["qty"] > 0].copy()

    # 円換算
    trades["net_pnl_yen"] = trades["entry_price"] * trades["qty"] * trades["net_ret"]

    keep = [
        "paper_day", "engine", "ticker", "code", "name",
        "signal_date", "entry_date", "exit_date", "hold_days", "reason",
        "score_rank",
        "lot_size", "min_cost", "alloc_yen", "qty",
        "entry_price", "exit_price", "net_ret", "net_pnl_yen"
    ]
    keep = [c for c in keep if c in trades.columns]
    trades = trades[keep].copy()

    try:
        old = pd.read_csv(p.out_trades_csv)
        old = _to_dates(old)
    except FileNotFoundError:
        old = pd.DataFrame()

    merged = pd.concat([old, trades], ignore_index=True)
    # 重複除去
    if not merged.empty and all(c in merged.columns for c in ["paper_day", "ticker", "entry_date"]):
        merged = merged.drop_duplicates(subset=["paper_day", "ticker", "entry_date"], keep="last")

    merged.to_csv(p.out_trades_csv, index=False, encoding="utf-8-sig")
    return merged


def write_summary(trades: pd.DataFrame, p: PaperParams) -> None:
    if trades.empty:
        summary = pd.DataFrame([{
            "trades": 0,
            "win_rate": float("nan"),
            "total_net_pnl_yen": 0.0,
            "avg_net_pnl_yen": float("nan"),
            "initial_cash_yen": p.initial_cash_yen,
        }])
        summary.to_csv(p.out_summary_csv, index=False, encoding="utf-8-sig")
        return

    win_rate = (trades["net_pnl_yen"] > 0).mean()
    total = trades["net_pnl_yen"].sum()
    avg = trades["net_pnl_yen"].mean()

    summary = pd.DataFrame([{
        "trades": int(len(trades)),
        "win_rate": float(win_rate),
        "total_net_pnl_yen": float(total),
        "avg_net_pnl_yen": float(avg),
        "initial_cash_yen": p.initial_cash_yen,
        "trend_max_positions_live": p.trend_max_positions_live,
        "trend_lot_size": p.trend_lot_size,
        "defensive_lot_size": p.defensive_lot_size,
        "close_out_same_day_only": p.close_out_same_day_only,
    }])
    summary.to_csv(p.out_summary_csv, index=False, encoding="utf-8-sig")


def main():
    cand = pd.read_csv(P.candidates_csv)
    cand = _to_dates(cand)

    orders = build_today_orders(cand, P)
    if orders.empty:
        print("== PAPER TODAY ORDERS ==\n(no orders today)")
        return

    orders.to_csv(P.out_today_csv, index=False, encoding="utf-8-sig")
    print("== PAPER TODAY ORDERS ==")
    print(f"paper_day: {orders['paper_day'].iloc[0].date()}")
    print(f"Saved: {P.out_today_csv}")

    show_cols = [
        "engine","ticker","code","name",
        "lot_size","min_cost","alloc_yen",
        "signal_date","entry_date","exit_date",
        "score_rank","entry_price","exit_price","net_ret"
    ]
    show_cols = [c for c in show_cols if c in orders.columns]
    print(orders[show_cols].to_string(index=False))

    trades = append_trades(orders, P)
    write_summary(trades, P)

    print(f"\nSaved: {P.out_trades_csv}, {P.out_summary_csv}")
    print("\n== PAPER SUMMARY ==")
    print(pd.read_csv(P.out_summary_csv).to_string(index=False))


if __name__ == "__main__":
    main()