# streamlit_app.py
# ------------------------------------------------------------
# Portafolio Inversiones Lectures — VISOR SOLO LECTURA
# - Lee trades y aliases desde Google Sheets (URLs en st.secrets)
# - UI similar al tracker: hero, KPIs, tablas y gráficos
# - Solo lectura: no hay edición de datos
# ------------------------------------------------------------
from datetime import date, timedelta, datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ========================== Config & Theme ==========================
st.set_page_config(
    page_title="Portafolio Inversiones Lectures",
    page_icon="👀",
    layout="wide",
)


def theme_opt(key: str, default):
    try:
        val = st.get_option(key)
        return val if val else default
    except Exception:
        return default


THEME_BASE = theme_opt("theme.base", "light")
CLR_PRIMARY = theme_opt("theme.primaryColor", "#2563eb")
CLR_BG = (
    theme_opt("theme.backgroundColor", "#ffffff")
    if THEME_BASE == "light"
    else theme_opt("theme.backgroundColor", "#0e1117")
)
CLR_BG_2 = (
    theme_opt("theme.secondaryBackgroundColor", "#f6f8fa")
    if THEME_BASE == "light"
    else theme_opt("theme.secondaryBackgroundColor", "#1b1f24")
)
CLR_TEXT = (
    theme_opt("theme.textColor", "#0f172a")
    if THEME_BASE == "light"
    else theme_opt("theme.textColor", "#f3f4f6")
)
PLOTLY_TEMPLATE = "plotly_white" if THEME_BASE == "light" else "plotly_dark"

# ========================== Look & Feel (CSS) ==========================
st.markdown(
    f"""
<style>
:root {{
  --clr-primary: {CLR_PRIMARY};
  --clr-bg: {CLR_BG};
  --clr-bg2: {CLR_BG_2};
  --clr-text: {CLR_TEXT};
  --clr-green: #10b981;
  --clr-red: #ef4444;
  --shadow: 0 8px 30px rgba(0,0,0,.06);
  --radius: 16px;
}}
.main .block-container {{ max-width: 1220px; padding-top: 0rem; }}

/* HERO */
.hero {{
  margin: 0 0 18px 0;
  padding: 28px 28px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(37,99,235,0.02));
  border: 1px solid rgba(125,125,125,0.18);
}}
.hero h1 {{ margin: 0; font-size: 2.0rem; letter-spacing: -0.02em; }}
.hero .sub {{ opacity: .9; margin-top: 6px; }}
.hero .chip {{
  display:inline-flex; align-items:center; gap:.45rem;
  background: var(--clr-bg2); border:1px solid rgba(125,125,125,.2);
  padding: .35rem .6rem; border-radius: 999px; font-size:.82rem;
}}

/* Sticky KPIs */
.kpi-bar {{
  position: sticky; top: 0; z-index: 50;
  backdrop-filter: blur(6px);
  background: color-mix(in srgb, var(--clr-bg) 92%, transparent);
  padding: 8px 0 12px 0; margin-bottom: 10px;
  border-bottom: 1px solid rgba(125,125,125,.12);
}}
.kpi-card {{
  background: var(--clr-bg2); border:1px solid rgba(125,125,125,.15);
  border-radius: var(--radius); padding: 12px 14px; box-shadow: var(--shadow);
}}
.kpi-label {{ font-size:.85rem; color: rgba(127,127,127,.9); }}
.kpi-value {{ font-size:1.6rem; font-weight:700; margin-top:4px; }}

/* Cards generales */
.card {{
  background: var(--clr-bg2); border:1px solid rgba(125,125,125,.15);
  border-radius: var(--radius); padding: 16px; box-shadow: var(--shadow);
}}

/* Encabezados y separadores */
.section-title {{ margin: 8px 0 8px; }}
hr.sep {{ border:none; border-top: 1px solid rgba(125,125,125,.15); margin: 8px 0 18px; }}

/* Badges */
.badge {{
  display:inline-block; padding:.2rem .55rem; border-radius:999px;
  font-size:.75rem; font-weight:600; border:1px solid rgba(125,125,125,.25);
  background: rgba(125,125,125,.08);
}}
.badge-Equity {{ background: rgba(37,99,235,.12); color:#2563eb; border-color:rgba(37,99,235,.25); }}
.badge-ETF {{ background: rgba(99,102,241,.12); color:#6366f1; border-color:rgba(99,102,241,.25); }}
.badge-Crypto {{ background: rgba(234,179,8,.12); color:#eab308; border-color:rgba(234,179,8,.25); }}
.badge-Index {{ background: rgba(34,197,94,.12); color:#22c55e; border-color:rgba(34,197,94,.25); }}
.badge-Other {{ background: rgba(148,163,184,.18); color:#94a3b8; border-color:rgba(148,163,184,.3); }}

/* DataFrame tweaks */
.stDataFrame thead th {{ font-weight: 700 !important; }}
</style>
""",
    unsafe_allow_html=True,
)

# ========================== Helpers de secrets & CSV ==========================
def find_secret(possible_keys: List[str]) -> str:
    """
    Busca la primera clave existente en st.secrets entre varias alternativas.
    Lanza KeyError si ninguna existe.
    """
    for key in possible_keys:
        try:
            return st.secrets[key]  # type: ignore[index]
        except Exception:
            continue
    raise KeyError(f"No encontré ninguna de las claves en secrets.toml: {possible_keys}")


@st.cache_data(ttl=300, show_spinner=True)
def load_trades() -> pd.DataFrame:
    """
    Carga el CSV de trades desde la URL en secrets y normaliza:
    - nombres de columnas en minúscula
    - intenta detectar columna de fecha aunque tenga otro nombre
    - garantiza que exista 'trade_date'
    """
    url = find_secret(["TRADES_CSV_URL", "trades_csv_url", "TRADES_URL", "trades_url"])
    df = pd.read_csv(url)

    # normalizar nombres de columnas
    df.columns = [str(c).strip().lower() for c in df.columns]

    # intentar detectar la columna de fecha si no se llama exactamente 'trade_date'
    if "trade_date" not in df.columns:
        possible_date_cols = {
            "trade_date",
            "trade date",
            "fecha",
            "fecha_operacion",
            "fecha operación",
            "date",
        }
        detected = None
        for c in df.columns:
            if c.strip().lower() in possible_date_cols:
                detected = c
                break
        if detected is not None and detected != "trade_date":
            df = df.rename(columns={detected: "trade_date"})

    # si aun así no hay 'trade_date', mostramos mensaje y devolvemos DF vacío estándar
    if "trade_date" not in df.columns:
        st.error(
            "No encontré una columna de fecha de operación en el CSV de *trades*.\n\n"
            "Asegúrate de que el encabezado incluya una columna llamada `trade_date` "
            "o alguna de estas variantes: `trade date`, `fecha`, `fecha_operacion`, `date`."
        )
        return pd.DataFrame(
            columns=[
                "id",
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
        )

    # mapeo y tipos
    rename_map = {
        "portfolio": "portfolio",
        "name": "name",
        "ticker": "ticker",
        "asset_class": "asset_class",
        "trade_date": "trade_date",
        "action": "action",
        "quantity": "quantity",
        "price": "price",
        "currency": "currency",
        "fees": "fees",
        "notes": "notes",
    }
    df = df.rename(columns=rename_map)

    # crear id si no existe
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # tipos
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

    for c in ["quantity", "price", "fees"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in [
        "portfolio",
        "name",
        "ticker",
        "asset_class",
        "action",
        "currency",
        "notes",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_alias() -> pd.DataFrame:
    try:
        url = find_secret(
            ["ALIAS_CSV_URL", "alias_csv_url", "PRICES_ALIAS_URL", "alias_url"]
        )
    except KeyError:
        # si no hay alias configurado, devolvemos DF vacío
        return pd.DataFrame(
            columns=["ticker", "price_ticker", "multiplier", "px_currency"]
        )

    df = pd.read_csv(url)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "ticker": "ticker",
        "price_ticker": "price_ticker",
        "multiplier": "multiplier",
        "px_currency": "px_currency",
    }
    df = df.rename(columns=rename_map)

    for c in ["ticker", "price_ticker", "px_currency"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip().str.upper()
    if "multiplier" in df.columns:
        df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1.0)
    else:
        df["multiplier"] = 1.0
    return df


# ========================== Precios y FX ==========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_prices(tickers, start_dt: date, end_dt: date, adjusted=True) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    safe_end = end_dt + timedelta(days=1)  # yfinance usa end exclusivo
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
        "USDCOP": (
            float(usdcop.loc[:last_idx].iloc[-1])
            if (not usdcop.empty and last_idx in usdcop.index)
            else None
        ),
        "EURUSD": (
            float(eurusd.loc[:last_idx].iloc[-1])
            if (not eurusd.empty and last_idx in eurusd.index)
            else None
        ),
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
        name = (
            grp["name"].dropna().iloc[0]
            if "name" in grp.columns and not grp["name"].dropna().empty
            else tkr
        )
        asset_class = (
            grp["asset_class"].dropna().iloc[0]
            if "asset_class" in grp.columns and not grp["asset_class"].dropna().empty
            else ""
        )
        currency = (
            grp["currency"].dropna().iloc[-1]
            if "currency" in grp.columns and not grp["currency"].dropna().empty
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
    asof: pd.Timestamp,
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

        rows.append(
            {
                **r,
                "last_price": (
                    round(last_price_conv, 6) if pd.notna(last_price_conv) else np.nan
                ),
                "market_value": market_value,
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
            }
        )
    return pd.DataFrame(rows)


# ===== Helpers para la pestaña de precios =====
def first_buy_date_for_ticker(trades: pd.DataFrame, ticker: str) -> Optional[date]:
    sub = trades[(trades["ticker"] == ticker) & (trades["action"].str.upper() == "BUY")]
    if sub.empty:
        return None
    return sub["trade_date"].min()


def get_price_series_for_ticker(
    ticker: str,
    alias_df: pd.DataFrame,
    panel: pd.DataFrame,
    start_d: date,
    end_d: date,
) -> Optional[pd.Series]:
    alias_df = alias_df.copy()
    alias_df["ticker"] = alias_df["ticker"].str.upper()
    row = alias_df[alias_df["ticker"] == ticker.upper()]
    px_tk, mult = ticker, 1.0
    if not row.empty:
        px_tk = row["price_ticker"].iloc[0] or ticker
        mult = float(row["multiplier"].iloc[0] or 1.0)
    if px_tk not in panel.columns:
        return None
    s = panel[px_tk].dropna() * mult
    s = s[(s.index.date >= start_d) & (s.index.date <= end_d)]
    if s.empty:
        return None
    return s


def _nearest_index(
    idx: pd.DatetimeIndex, target: pd.Timestamp, max_days: int = 4
) -> Optional[pd.Timestamp]:
    if idx.empty:
        return None
    pos = idx.get_indexer([pd.Timestamp(target)], method="nearest")[0]
    ts = idx[pos]
    return ts if abs((ts - pd.Timestamp(target)).days) <= max_days else None


# ========================== Sidebar ==========================
with st.sidebar:
    st.header("⚙️ Preferencias")
    use_adjusted = st.toggle(
        "Usar precios ajustados",
        value=True,
        help="Incluye dividendos/splits; recomendado.",
    )
    st.caption("FX: EURUSD=X y USDCOP=X (últimas tasas disponibles).")

    if st.button("🔄 Refrescar datos"):
        st.cache_data.clear()
        st.experimental_rerun()

# ========================== HERO ==========================
st.markdown(
    f"""
<div class="hero">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:16px;">
    <div>
      <h1>Portafolio Inversiones Lectures</h1>
      <div class="sub">
        Visor profesional sincronizado con las operaciones del fondo. 
        Posiciones, composición y evolución histórica sin permitir cambios en los datos.
      </div>
    </div>
    <div class="chip">👁 Modo: <b>Lectura</b></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ========================== Carga de datos base ==========================
trades = load_trades()
alias_df = load_alias()

if trades.empty:
    start_hist = date.today() - timedelta(days=365)
else:
    first_trade_date = pd.to_datetime(trades["trade_date"]).min().date()
    start_hist = first_trade_date - timedelta(days=7)
end_hist = date.today()

unique_tickers = sorted(set(trades["ticker"])) if not trades.empty else []
alias_needed = (
    sorted(set(alias_df["price_ticker"].dropna().unique().tolist()))
    if not alias_df.empty
    else []
)
universe = sorted(set(unique_tickers + alias_needed))

price_raw = (
    load_prices(universe, start_hist, end_hist, adjusted=use_adjusted)
    if universe
    else pd.DataFrame()
)

# cripto → calendario diario con ffill
def has_crypto_in(tks):
    return any(str(t).upper().endswith("-USD") for t in tks)


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

# ========================== Barra de KPIs (sticky) ==========================
with st.container():
    st.markdown('<div class="kpi-bar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-label">Datos</div>
          <div class="kpi-value">Fin de día</div>
          <div class="kpi-label">Ajustados: <b>{'Sí' if use_adjusted else 'No'}</b></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        asof_txt = (
            asof.date().isoformat()
            if isinstance(asof, pd.Timestamp)
            else str(date.today())
        )
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-label">Último cierre disponible</div>
          <div class="kpi-value">{asof_txt}</div>
          <div class="kpi-label">Fuente: Yahoo Finance</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        fx_txt = []
        if rates_last.get("USDCOP"):
            fx_txt.append(f"USD→COP: <b>{rates_last['USDCOP']:,.0f}</b>")
        if rates_last.get("EURUSD"):
            fx_txt.append(f"EUR→USD: <b>{rates_last['EURUSD']:.4f}</b>")
        st.markdown(
            f"""
        <div class="kpi-card">
          <div class="kpi-label">FX (último)</div>
          <div class="kpi-value">{' • '.join(fx_txt) if fx_txt else '—'}</div>
          <div class="kpi-label">Actualizado al último precio disponible</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ========================== Tabs ==========================
tab_positions, tab_portfolio, tab_prices = st.tabs(
    ["📋 Posiciones", "📦 Portafolio", "📈 Evolución"]
)

# ========================== TAB — Posiciones ==========================
with tab_positions:
    st.markdown(
        '<h3 class="section-title">Posiciones abiertas (FIFO)</h3><hr class="sep">',
        unsafe_allow_html=True,
    )
    if trades.empty:
        st.info("No hay operaciones cargadas en la hoja de cálculo.")
    else:
        positions = build_positions_fifo(trades)
        if positions.empty:
            st.info("No hay posiciones abiertas (quizá todo está vendido).")
        else:
            pos_with_px = compute_position_values(
                positions, price_panel, asof, alias_df, rates_last
            )

            # Valor de posición en USD/COP por fila
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

            total_usd = float(np.nansum(pos_with_px["valor_pos_usd"].values))
            total_cop = float(np.nansum(pos_with_px["valor_pos_cop"].values))

            # Inversión total en USD para PnL
            def _inv_usd(row):
                inv, cur = row["invested"], row["currency"]
                v = (
                    convert_value(inv, cur, "USD", rates_last)
                    if (pd.notna(inv) and cur)
                    else None
                )
                return np.nan if v is None else float(v)

            pos_with_px["invested_usd"] = pos_with_px.apply(_inv_usd, axis=1)
            total_invested_usd = float(np.nansum(pos_with_px["invested_usd"].values))
            total_pnl_usd = total_usd - total_invested_usd
            total_pnl_pct = (
                (total_pnl_usd / total_invested_usd * 100)
                if total_invested_usd > 0
                else 0.0
            )
            pnl_color = "var(--clr-green)" if total_pnl_usd >= 0 else "var(--clr-red)"

            k1, k2 = st.columns(2)
            with k1:
                st.markdown(
                    f"""
                <div class="kpi-card">
                  <div class="kpi-label">Valor actual del portafolio (USD)</div>
                  <div class="kpi-value">${total_usd:,.2f}</div>
                  <div class="kpi-label">Equivalente COP: ${total_cop:,.0f}</div>
                </div>""",
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""
                <div class="kpi-card">
                  <div class="kpi-label">Resultado acumulado (USD)</div>
                  <div class="kpi-value" style="color:{pnl_color};">
                    ${total_pnl_usd:,.2f}
                  </div>
                  <div class="kpi-label">Rentabilidad total: {total_pnl_pct:,.2f}%</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            # Tabla pro con badges y colores (sin columna de portafolio)
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
                        "pnl_abs",
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
                        "pnl_abs": "PnL (USD)",
                        "pnl_pct": "Rentab. (%)",
                    }
                )
            )

            def badge(cls):
                c = (
                    f"badge-{cls}"
                    if cls in {"Equity", "ETF", "Crypto", "Index", "Other"}
                    else "badge-Other"
                )
                return f'<span class="badge {c}">{cls}</span>'

            if not df_show.empty:
                df_show["Clase"] = df_show["Clase"].apply(badge)
            df_show["Costo prom."] = df_show["Costo prom."].round(4)
            df_show["Precio actual"] = df_show["Precio actual"].round(4)
            df_show["Valor posición (USD)"] = df_show["Valor posición (USD)"].round(2)
            df_show["PnL (USD)"] = df_show["PnL (USD)"].round(2)
            df_show["Rentab. (%)"] = df_show["Rentab. (%)"].round(2)

            formatter = {
                "Costo prom.": "{:.4f}",
                "Precio actual": "{:.4f}",
                "Valor posición (USD)": "${:,.2f}",
                "PnL (USD)": "${:,.2f}",
                "Rentab. (%)": "{:.2f}%",
            }
            styler = (
                df_show.style.format(formatter, na_rep="—")
                .hide(axis="index")
                .apply(
                    lambda s: [
                        (
                            "color: var(--clr-green); font-weight:700;"
                            if (pd.notna(v) and v > 0)
                            else (
                                "color: var(--clr-red); font-weight:700;"
                                if (pd.notna(v) and v < 0)
                                else ""
                            )
                        )
                        for v in s
                    ],
                    subset=["Rentab. (%)", "PnL (USD)"],
                )
                .applymap(lambda _: "white-space: nowrap;", subset=["Clase"])
            )
            st.write(styler.to_html(escape=False), unsafe_allow_html=True)

# ========================== TAB — Portafolio ==========================
with tab_portfolio:
    st.markdown(
        '<h3 class="section-title">Resumen y composición</h3><hr class="sep">',
        unsafe_allow_html=True,
    )
    if trades.empty:
        st.info("Agrega operaciones en la hoja de cálculo para ver el resumen.")
    else:
        positions = build_positions_fifo(trades)
        if positions.empty:
            st.info("No hay posiciones abiertas.")
        else:
            pos_with_px = compute_position_values(
                positions, price_panel, asof, alias_df, rates_last
            )
            total_usd = total_cop = 0.0
            parts = []
            left, right = st.columns([0.55, 0.45])
            with left:
                currency_for_pie = st.selectbox(
                    "Moneda del gráfico", ["USD", "COP"], index=0
                )
            for _, r in pos_with_px.iterrows():
                mv, cur, name = r["market_value"], r["currency"], r["name"]
                if pd.isna(mv) or not cur:
                    continue
                v_usd = convert_value(mv, cur, "USD", rates_last)
                v_cop = convert_value(mv, cur, "COP", rates_last)
                if v_usd is not None:
                    total_usd += v_usd
                if v_cop is not None:
                    total_cop += v_cop
                v_base = convert_value(mv, cur, currency_for_pie, rates_last)
                if v_base is not None:
                    parts.append((name, v_base))

            with left:
                st.markdown(
                    f"""
                <div class="card">
                  <b>Valor total (USD):</b> ${total_usd:,.2f} &nbsp;&nbsp;|&nbsp;&nbsp;
                  <b>Valor total (COP):</b> ${total_cop:,.2f}<br/>
                  <span style="color: rgba(127,127,127,.9)">Precios {'ajustados' if use_adjusted else 'sin ajustar'} • Último cierre: {asof.date()}</span>
                </div>""",
                    unsafe_allow_html=True,
                )

            with right:
                st.empty()  # espaciador

            st.subheader(f"Composición del portafolio (en {currency_for_pie})")
            parts = [(n, v) for (n, v) in parts if pd.notna(v) and v > 0]
            if parts:
                names, values = zip(*parts)
                fig = go.Figure(
                    data=[go.Pie(labels=list(names), values=list(values), hole=0.35)]
                )
                fig.update_traces(
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>%{percent} (%{value:,.0f})<extra></extra>",
                )
                fig.update_layout(
                    template=PLOTLY_TEMPLATE,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=430,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay valores válidos para graficar.")

# ========================== TAB — Evolución de precios ==========================
with tab_prices:
    st.markdown(
        '<h3 class="section-title">Precios desde la primera compra (con marcadores)</h3><hr class="sep">',
        unsafe_allow_html=True,
    )
    if trades.empty:
        st.info("Agrega operaciones en la hoja de cálculo para visualizar precios.")
    else:
        positions_now = build_positions_fifo(trades)
        if positions_now.empty:
            st.info("No hay posiciones abiertas para graficar.")
        else:
            tickers_available = positions_now["ticker"].tolist()
            names_map = {r["ticker"]: r["name"] for _, r in positions_now.iterrows()}
            sel = st.multiselect(
                "Selecciona activos",
                options=tickers_available,
                default=tickers_available,
                format_func=lambda tk: f"{names_map.get(tk, tk)} ({tk})",
                max_selections=15,
            )
            normalize = st.toggle(
                "Normalizar a 100 (facilita comparar tendencias)", value=True
            )
            if not sel:
                st.info("Selecciona al menos un activo.")
            else:
                fig = go.Figure()
                any_series = False
                for tk in sel:
                    start_buy = first_buy_date_for_ticker(trades, tk)
                    if not start_buy:
                        continue
                    s = get_price_series_for_ticker(
                        tk, alias_df, price_panel, start_buy, asof.date()
                    )
                    if s is None:
                        continue
                    y_line = (s / s.iloc[0] * 100.0) if normalize else s
                    fig.add_trace(
                        go.Scatter(
                            x=s.index,
                            y=y_line,
                            mode="lines",
                            name=f"{names_map.get(tk, tk)}",
                        )
                    )
                    trades_tk = trades[
                        (trades["ticker"] == tk) & (trades["trade_date"] >= start_buy)
                    ]
                    if not trades_tk.empty:
                        xs_buy, ys_buy, hover_buy = [], [], []
                        xs_sell, ys_sell, hover_sell = [], [], []
                        for _, tr in trades_tk.iterrows():
                            tdate = pd.Timestamp(tr["trade_date"])
                            ts = _nearest_index(s.index, tdate, max_days=4)
                            if ts is None:
                                continue
                            y_val = s.loc[ts]
                            if normalize:
                                y_val = y_val / s.iloc[0] * 100.0
                            htxt = f"{names_map.get(tk, tk)}<br>{tr['action'].upper()} — {tr['quantity']} @ {tr['price']} {tr['currency']}<br>Fecha: {tr['trade_date']}"
                            if str(tr["action"]).upper() == "BUY":
                                xs_buy.append(ts)
                                ys_buy.append(y_val)
                                hover_buy.append(htxt)
                            elif str(tr["action"]).upper() == "SELL":
                                xs_sell.append(ts)
                                ys_sell.append(y_val)
                                hover_sell.append(htxt)
                        if xs_buy:
                            fig.add_trace(
                                go.Scatter(
                                    x=xs_buy,
                                    y=ys_buy,
                                    mode="markers",
                                    marker=dict(
                                        symbol="triangle-up",
                                        size=11,
                                        line=dict(width=0.5),
                                    ),
                                    marker_color="#16a34a",
                                    name=f"Compras {names_map.get(tk, tk)}",
                                    showlegend=False,
                                    hovertemplate="%{text}<extra></extra>",
                                    text=hover_buy,
                                )
                            )
                        if xs_sell:
                            fig.add_trace(
                                go.Scatter(
                                    x=xs_sell,
                                    y=ys_sell,
                                    mode="markers",
                                    marker=dict(
                                        symbol="triangle-down",
                                        size=11,
                                        line=dict(width=0.5),
                                    ),
                                    marker_color="#dc2626",
                                    name=f"Ventas {names_map.get(tk, tk)}",
                                    showlegend=False,
                                    hovertemplate="%{text}<extra></extra>",
                                    text=hover_sell,
                                )
                            )
                    any_series = True
                if any_series:
                    fig.update_layout(
                        template=PLOTLY_TEMPLATE,
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=520,
                        legend_title="Activos (click para ocultar/mostrar)",
                    )
                    fig.update_yaxes(
                        title="Índice (100=inicio)" if normalize else "Precio"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay series disponibles para las fechas seleccionadas.")
