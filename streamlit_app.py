# streamlit_app.py
# -------------------------------------------------------------------
# Portafolio — Visor Solo Lectura (versión mejorada)
# Lee trades/aliases desde Google Sheets (CSV publicado)
# NO permite modificar datos.
# -------------------------------------------------------------------

import io
from datetime import date, timedelta
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ========================== Config básica ==========================

st.set_page_config(
    page_title="Portafolio Inversiones Lectures",
    page_icon="👀",
    layout="wide",
)

st.title("Portafolio Inversiones Lectures")

# ========================== Helpers CSV / Secrets ==========================


def find_secret(possible_keys: List[str]) -> str:
    """Busca una clave en st.secrets usando varios nombres posibles."""
    for key in possible_keys:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            continue
    raise KeyError(f"No encontré ninguna de las claves en secrets.toml: {possible_keys}")


TRADES_CSV_URL = find_secret(
    ["TRADES_CSV_URL", "trades_csv_url", "TRADES_URL", "trades_url"]
)
ALIAS_CSV_URL = find_secret(
    ["ALIAS_CSV_URL", "alias_csv_url", "ALIAS_URL", "alias_url"]
)


@st.cache_data(ttl=60, show_spinner=False)
def read_csv_url(url: str) -> pd.DataFrame:
    """Descarga un CSV publicado por Google Sheets y lo convierte en DataFrame.

    Incluye un fix para el caso en que cada línea viene entrecomillada.
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text:
        return pd.DataFrame()

    lines = text.splitlines()

    # Caso especial: Google devuelve cada línea entre comillas ("id,portfolio,...")
    if (
        len(lines) > 0
        and lines[0].startswith('"')
        and lines[0].endswith('"')
        and "," in lines[0]
    ):
        cleaned = [ln.strip().strip('"') for ln in lines if ln.strip()]
        buffer = io.StringIO("\n".join(cleaned))
        df = pd.read_csv(buffer)
    else:
        buffer = io.StringIO(text)
        df = pd.read_csv(buffer)

    return df


def read_trades() -> pd.DataFrame:
    """Carga y normaliza el CSV de trades."""
    df = read_csv_url(TRADES_CSV_URL)

    if df.empty:
        return df

    # Segundo safeguard: si se leyó como 1 sola columna con el header dentro
    if df.shape[1] == 1 and isinstance(df.columns[0], str) and "id,portfolio" in df.columns[0]:
        header_line = df.columns[0]
        data_lines = df.iloc[:, 0].dropna().astype(str).tolist()
        all_lines = [header_line] + data_lines
        buffer = io.StringIO("\n".join(all_lines))
        df = pd.read_csv(buffer)

    # Aseguramos columnas básicas
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    required_core = [
        "portfolio",
        "name",
        "ticker",
        "asset_class",
        "trade_date",
        "action",
        "quantity",
        "price",
        "currency",
        "fees",
        "notes",
    ]
    missing = [c for c in required_core if c not in df.columns]
    if missing:
        st.error(
            "El CSV de trades no tiene las columnas esperadas.\n\n"
            f"Faltan: {missing}\n\n"
            "Revisa el encabezado del CSV o la hoja de Google."
        )
        st.write("Columnas encontradas:", list(df.columns))
        st.stop()

    # Tipos
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    for c in ["quantity", "price", "fees"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["portfolio", "name", "ticker", "asset_class", "action", "currency", "notes"]:
        df[c] = df[c].astype(str).str.strip()

    return df


def read_alias() -> pd.DataFrame:
    """Carga tabla de aliases de precios."""
    df = read_csv_url(ALIAS_CSV_URL)
    if df.empty:
        return df

    for c in ["ticker", "price_ticker", "px_currency"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip().str.upper()
    if "multiplier" in df.columns:
        df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1.0)
    return df


# ========================== Precios y FX ==========================


@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(tickers, start_dt: date, end_dt: date, adjusted=True) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    safe_end = end_dt + timedelta(days=1)
    df = yf.download(
        tickers,
        start=start_dt,
        end=safe_end,
        interval="1d",
        auto_adjust=adjusted,
        progress=False,
    )
    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
        df = df["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all").dropna(axis=1, how="all")


def has_crypto_in(tickers: List[str]) -> bool:
    return any(str(t).upper().endswith("-USD") for t in tickers)


def align_calendar_union_ffill(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    idx_all = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    return panel.reindex(idx_all).ffill()


@st.cache_data(ttl=1800, show_spinner=False)
def load_fx_pair(pair: str, start_dt: date, end_dt: date) -> pd.Series:
    df = load_prices([pair], start_dt, end_dt, adjusted=False)
    if pair in df.columns:
        s = df[pair].dropna()
    else:
        s = pd.Series(dtype=float)
    if s.empty:
        idx = pd.date_range(start_dt, end_dt, freq="D")
        return pd.Series(np.nan, index=idx)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()
    return s


def fx_last_rates_from_series(
    usdcop: pd.Series, eurusd: pd.Series
) -> Dict[str, Optional[float]]:
    last_idx = None
    if not usdcop.empty and not eurusd.empty:
        last_idx = min(usdcop.dropna().index.max(), eurusd.dropna().index.max())
    elif not usdcop.empty:
        last_idx = usdcop.dropna().index.max()
    elif not eurusd.empty:
        last_idx = eurusd.dropna().index.max()

    return {
        "USDCOP": float(usdcop.loc[:last_idx].iloc[-1]) if (not usdcop.empty and last_idx in usdcop.index) else None,
        "EURUSD": float(eurusd.loc[:last_idx].iloc[-1]) if (not eurusd.empty and last_idx in eurusd.index) else None,
        "ASOF": last_idx,
    }


def convert_value(
    val: float, from_ccy: str, to_ccy: str, rates: Dict[str, Optional[float]]
) -> Optional[float]:
    if pd.isna(val):
        return None
    from_ccy, to_ccy = (from_ccy or "").upper(), (to_ccy or "").upper()
    if from_ccy == to_ccy:
        return float(val)

    eurusd, usdcop = rates.get("EURUSD"), rates.get("USDCOP")

    if from_ccy == "USD" and to_ccy == "COP" and usdcop:
        return float(val) * usdcop
    if from_ccy == "COP" and to_ccy == "USD" and usdcop:
        return float(val) / usdcop
    if from_ccy == "EUR" and to_ccy == "USD" and eurusd:
        return float(val) * eurusd
    if from_ccy == "USD" and to_ccy == "EUR" and eurusd:
        return float(val) / eurusd
    if from_ccy == "EUR" and to_ccy == "COP" and eurusd and usdcop:
        return float(val) * eurusd * usdcop
    if from_ccy == "COP" and to_ccy == "EUR" and eurusd and usdcop:
        return float(val) / (eurusd * usdcop)
    return None


# ========================== Posiciones (FIFO) ==========================


def build_positions_fifo(trades: pd.DataFrame) -> pd.DataFrame:
    """Construye posiciones abiertas usando un esquema FIFO sencillo."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "portfolio",
                "ticker",
                "name",
                "asset_class",
                "currency",
                "qty",
                "avg_cost",
                "invested",
            ]
        )

    positions = []
    trades_sorted = trades.sort_values(
        ["portfolio", "ticker", "trade_date", "id"], kind="stable"
    )
    for (pf, tkr), grp in trades_sorted.groupby(["portfolio", "ticker"], dropna=False):
        name = grp["name"].dropna().iloc[0] if not grp["name"].dropna().empty else tkr
        asset_class = (
            grp["asset_class"].dropna().iloc[0]
            if not grp["asset_class"].dropna().empty
            else ""
        )
        currency = (
            grp["currency"].dropna().iloc[-1]
            if not grp["currency"].dropna().empty
            else "USD"
        )

        lots: List[List[float]] = []
        for _, r in grp.iterrows():
            action = str(r.get("action", "")).upper()
            qty = float(r.get("quantity", 0) or 0)
            price = float(r.get("price", 0) or 0)
            if action == "BUY":
                lots.append([qty, price])
            elif action == "SELL":
                remaining = qty
                while remaining > 1e-12 and lots:
                    lot_qty, lot_cost = lots[0]
                    if lot_qty <= remaining + 1e-12:
                        remaining -= lot_qty
                        lots.pop(0)
                    else:
                        lots[0] = [lot_qty - remaining, lot_cost]
                        remaining = 0.0

        total_qty = sum(q for q, _ in lots)
        if total_qty > 1e-12:
            total_invested = sum(q * c for q, c in lots)
            avg_cost = total_invested / total_qty
            positions.append(
                {
                    "portfolio": pf,
                    "ticker": tkr,
                    "name": name,
                    "asset_class": asset_class,
                    "currency": currency,
                    "qty": total_qty,
                    "avg_cost": avg_cost,
                    "invested": total_invested,
                }
            )

    return pd.DataFrame(positions)


def compute_position_values(
    positions: pd.DataFrame,
    price_panel: pd.DataFrame,
    asof,
    alias_df: pd.DataFrame,
    rates: Dict[str, Optional[float]],
) -> pd.DataFrame:
    if positions.empty:
        return positions

    alias_map = {}
    for _, r in alias_df.iterrows():
        tk = str(r.get("ticker", "")).upper()
        if not tk:
            continue
        alias_map[tk] = {
            "price_ticker": str(r.get("price_ticker", "")).upper() or tk,
            "multiplier": float(r.get("multiplier", 1.0) or 1.0),
            "px_currency": (str(r.get("px_currency", "")) or "").upper() or None,
        }

    rows = []
    for _, r in positions.iterrows():
        orig_tk = r["ticker"]
        qty = float(r["qty"])
        avg_cost = float(r["avg_cost"])
        pos_ccy = (r["currency"] or "USD").upper()

        a = alias_map.get(orig_tk.upper(), None)
        px_tk = a["price_ticker"] if a else orig_tk
        mult = a["multiplier"] if a else 1.0
        px_ccy = a["px_currency"] if a else None

        last_price_raw = np.nan
        if px_tk in price_panel.columns:
            s = price_panel[px_tk].dropna()
            s = s[s.index <= asof]
            if not s.empty:
                last_price_raw = float(s.iloc[-1]) * float(mult)

        last_price_conv = last_price_raw
        if pd.notna(last_price_raw):
            src_ccy = px_ccy or pos_ccy
            if src_ccy != pos_ccy:
                conv = convert_value(last_price_raw, src_ccy, pos_ccy, rates)
                last_price_conv = conv if conv is not None else np.nan

        market_value = qty * last_price_conv if pd.notna(last_price_conv) else np.nan
        pnl_abs = (
            (last_price_conv - avg_cost) * qty if pd.notna(last_price_conv) else np.nan
        )
        pnl_pct = (
            ((last_price_conv / avg_cost) - 1) * 100
            if (pd.notna(last_price_conv) and avg_cost > 0)
            else np.nan
        )

        row_out = dict(r)
        row_out.update(
            {
                "last_price": last_price_conv,
                "market_value": market_value,
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
            }
        )
        rows.append(row_out)

    return pd.DataFrame(rows)


# ========================== Sidebar ==========================

with st.sidebar:
    st.header("Preferencias")
    use_adjusted = st.toggle(
        "Usar precios ajustados",
        value=True,
        help="Incluye dividendos/splits; recomendado.",
    )
    st.caption("FX: EURUSD=X y USDCOP=X (últimas tasas disponibles).")
    if st.button("Refrescar datos"):
        st.cache_data.clear()
        st.rerun()

# ========================== Carga base (trades + alias) ==========================

trades = read_trades()
alias_df = read_alias()

if trades.empty:
    st.warning("No hay operaciones en 'trades'. Verifica el CSV publicado.")
    st.stop()

# ========================== Precios y FX ==========================

first_trade_date = trades["trade_date"].min()
if pd.isna(first_trade_date):
    st.error("No se pudieron interpretar las fechas de 'trade_date' en el CSV.")
    st.stop()

start_hist = first_trade_date - timedelta(days=7)
end_hist = date.today()

unique_tickers = sorted(set(trades["ticker"])) if not trades.empty else []
alias_needed = (
    sorted(set(alias_df["price_ticker"].dropna().unique().tolist()))
    if not alias_df.empty and "price_ticker" in alias_df.columns
    else []
)
universe = sorted(set(unique_tickers + alias_needed))

price_raw = (
    load_prices(universe, start_hist, end_hist, adjusted=use_adjusted)
    if universe
    else pd.DataFrame()
)
price_panel = (
    align_calendar_union_ffill(price_raw)
    if (not price_raw.empty and has_crypto_in(universe))
    else price_raw.copy()
)

usdcop_series, eurusd_series = (pd.Series(dtype=float), pd.Series(dtype=float))
if not price_panel.empty:
    idx = price_panel.index
    start_dt = idx.min().date()
    end_dt = idx.max().date()
    usdcop_series = load_fx_pair("USDCOP=X", start_dt, end_dt).reindex(idx).ffill()
    eurusd_series = load_fx_pair("EURUSD=X", start_dt, end_dt).reindex(idx).ffill()

rates_last = fx_last_rates_from_series(usdcop_series, eurusd_series)
asof = (
    pd.to_datetime(price_panel.index.max())
    if not price_panel.empty
    else pd.Timestamp(end_hist)
)

# ========================== Tabs ==========================

tab_positions, tab_portfolio, tab_prices = st.tabs(
    ["📋 Posiciones", "📦 Composición", "📈 Evolución de precios"]
)

# ----- TAB Posiciones -----
with tab_positions:
    positions = build_positions_fifo(trades)
    if positions.empty:
        st.info("No hay posiciones abiertas (quizá todo está vendido).")
    else:
        pos_with_px = compute_position_values(
            positions, price_panel, asof, alias_df, rates_last
        )

        # Valor de la posición en USD y COP
        def _val_usd(row):
            mv, cur = row["market_value"], row["currency"]
            v = (
                convert_value(mv, cur, "USD", rates_last)
                if (pd.notna(mv) and cur)
                else None
            )
            return np.nan if v is None else float(v)

        def _val_cop(row):
            mv, cur = row["market_value"], row["currency"]
            v = (
                convert_value(mv, cur, "COP", rates_last)
                if (pd.notna(mv) and cur)
                else None
            )
            return np.nan if v is None else float(v)

        pos_with_px["valor_pos_usd"] = pos_with_px.apply(_val_usd, axis=1)
        pos_with_px["valor_pos_cop"] = pos_with_px.apply(_val_cop, axis=1)

        # Inversión inicial en USD y P&L absoluto en USD por posición
        def _inv_usd(row):
            inv, cur = row["invested"], row["currency"]
            v = (
                convert_value(inv, cur, "USD", rates_last)
                if (pd.notna(inv) and cur)
                else None
            )
            return np.nan if v is None else float(v)

        pos_with_px["invested_usd"] = pos_with_px.apply(_inv_usd, axis=1)
        pos_with_px["pnl_usd"] = (
            pos_with_px["valor_pos_usd"] - pos_with_px["invested_usd"]
        )

        total_usd = float(np.nansum(pos_with_px["valor_pos_usd"].values))
        total_inv_usd = float(np.nansum(pos_with_px["invested_usd"].values))
        total_pnl_usd = total_usd - total_inv_usd

        c1, _ = st.columns(2)
        with c1:
            st.metric(
                "Valor actual del portafolio (USD)",
                f"${total_usd:,.2f}",
                delta=f"${total_pnl_usd:,.2f}",
            )

        # Tabla de posiciones sin el nombre del portafolio
        df_show = (
            pos_with_px[
                [
                    "name",
                    "ticker",
                    "asset_class",
                    "currency",
                    "qty",
                    "avg_cost",
                    "last_price",
                    "valor_pos_usd",
                    "pnl_usd",
                    "pnl_pct",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "name": "Nombre",
                    "ticker": "Ticker",
                    "asset_class": "Clase",
                    "currency": "Moneda",
                    "qty": "Cantidad",
                    "avg_cost": "Costo prom.",
                    "last_price": "Precio actual",
                    "valor_pos_usd": "Valor posición (USD)",
                    "pnl_usd": "P&L (USD)",
                    "pnl_pct": "Rentab. (%)",
                }
            )
        )

        def color_pnl(val):
            try:
                if pd.isna(val):
                    return ""
                if val > 0:
                    return "color: green;"
                if val < 0:
                    return "color: red;"
            except Exception:
                return ""
            return ""

        styled = (
            df_show.style.format(
                {
                    "Cantidad": "{:.4f}",
                    "Costo prom.": "{:.4f}",
                    "Precio actual": "{:.4f}",
                    "Valor posición (USD)": "${:,.2f}",
                    "P&L (USD)": "${:,.2f}",
                    "Rentab. (%)": "{:.2f}%",
                }
            )
            .applymap(color_pnl, subset=["P&L (USD)", "Rentab. (%)"])
        )

        st.dataframe(styled, use_container_width=True)

# ----- TAB Composición -----
with tab_portfolio:
    positions = build_positions_fifo(trades)
    if positions.empty:
        st.info("No hay posiciones abiertas.")
    else:
        pos_with_px = compute_position_values(
            positions, price_panel, asof, alias_df, rates_last
        )
        parts = []
        for _, r in pos_with_px.iterrows():
            mv, cur, name = r["market_value"], r["currency"], r["name"]
            if pd.isna(mv) or not cur:
                continue
            v_usd = convert_value(mv, cur, "USD", rates_last)
            if v_usd is not None:
                parts.append((name, v_usd))

        st.subheader("Composición del portafolio (USD)")
        parts = [(n, v) for (n, v) in parts if pd.notna(v) and v > 0]
        if parts:
            names, values = zip(*parts)
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=list(names),
                        values=list(values),
                        hole=0.4,
                    )
                ]
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay valores válidos para graficar.")

# ----- TAB Evolución de precios -----
with tab_prices:
    st.subheader("Evolución de precios")

    if price_panel.empty:
        st.info("No hay precios disponibles para graficar.")
    else:
        available_tickers = list(price_panel.columns)
        default_sel = (
            available_tickers[: min(5, len(available_tickers))]
            if available_tickers
            else []
        )

        sel = st.multiselect(
            "Selecciona uno o varios activos",
            options=available_tickers,
            default=default_sel,
        )

        if not sel:
            st.info("Selecciona al menos un activo para ver la serie de precios.")
        else:
            fig = go.Figure()
            for t in sel:
                fig.add_trace(
                    go.Scatter(
                        x=price_panel.index,
                        y=price_panel[t],
                        mode="lines",
                        name=t,
                    )
                )
            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Precio",
                margin=dict(l=0, r=0, t=30, b=0),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)
