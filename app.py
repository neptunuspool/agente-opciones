import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date

st.set_page_config(page_title="Agente Opciones", page_icon="📈", layout="centered")
st.title("📈 Agente de Opciones")
st.caption("Modo simulado · No es asesoría financiera")

EVENTOS = {
    "2026-11-06": "Reporte de empleo",
    "2026-11-10": "Inflación (CPI)",
    "2026-12-09": "Decisión de la Fed",
}

def f(d):
    return datetime.strptime(d, "%Y-%m-%d")

@st.cache_data(ttl=300)
def traer(simbolo):
    t = yf.Ticker(simbolo)
    h = t.history(period="3mo")
    venc = min(t.options, key=lambda d: abs((f(d) - datetime.now()).days - 35))
    ch = t.option_chain(venc)
    vix = yf.Ticker("^VIX").history(period="1y")["Close"]
    return h, venc, ch.calls, ch.puts, vix

simbolo = st.text_input("Símbolo (ejemplo: SPY, AAPL, TSLA)", "SPY").upper().strip()

if "historial" not in st.session_state:
    st.session_state.historial = []

if st.button("🔍 Analizar ahora", use_container_width=True, type="primary"):
    try:
        h, venc, calls, puts, vix = traer(simbolo)
        precio = h["Close"].iloc[-1]
        dias = max((f(venc) - datetime.now()).days, 1)

        atm = calls.loc[(calls.strike - precio).abs().idxmin()]
        iv = atm.impliedVolatility
        mov = precio * iv * np.sqrt(dias / 365)
        mid = (atm.bid + atm.ask) / 2
        spread = (atm.ask - atm.bid) / mid * 100 if mid > 0 else 0
        hv = np.log(h["Close"]).diff().tail(30).std() * np.sqrt(252)
        rank = (vix.iloc[-1] - vix.min()) / (vix.max() - vix.min()) * 100
        pc_oi = puts.openInterest.sum() / calls.openInterest.sum()
        skew_p = puts.loc[(puts.strike - precio * 0.95).abs().idxmin()].impliedVolatility
        skew_c = calls.loc[(calls.strike - precio * 1.05).abs().idxmin()].impliedVolatility

        cerca = [(n, (f(d).date() - date.today()).days)
                 for d, n in EVENTOS.items() if 0 <= (f(d).date() - date.today()).days <= 2]

        st.subheader(f"{simbolo} · ${precio:.2f}")
        st.caption(f"Vencimiento analizado: {venc} ({dias} días)")

        c1, c2 = st.columns(2)
        c1.metric("Movimiento esperado", f"±${mov:.0f}", f"{mov/precio*100:.1f}%", delta_color="off")
        c2.metric("Rango esperado", f"${precio-mov:.0f} - ${precio+mov:.0f}")
        c3, c4 = st.columns(2)
        c3.metric("Volatilidad implícita", f"{iv*100:.1f}%")
        c4.metric("Volatilidad real (30d)", f"{hv*100:.1f}%")
        c5, c6 = st.columns(2)
        c5.metric("VIX rank (1 año)", f"{rank:.0f}/100")
        c6.metric("Put/Call (int. abierto)", f"{pc_oi:.2f}")

        st.subheader("🧠 Lectura del agente")
        if cerca:
            st.warning("⏳ ESPERAR: evento cercano → " + ", ".join(f"{n} (en {d} días)" for n, d in cerca))
        elif iv > hv * 1.25:
            st.warning("Las opciones cobran bastante más de lo que el precio realmente se mueve. Comprar sale caro. Mejor esperar.")
        elif rank < 30 and iv <= hv * 1.1:
            st.success("Condiciones razonables para comprar. Revisa manualmente antes de simular una operación.")
        else:
            st.info("Sin ventaja clara. No operar también es una decisión válida.")

        st.write(f"- **Sesgo:** {'defensivo (más puts, suele ser cobertura de fondos)' if pc_oi > 1.2 else 'optimista' if pc_oi < 0.8 else 'equilibrado'}")
        st.write(f"- **Protección a la baja:** IV de puts −5% = {skew_p*100:.1f}% vs calls +5% = {skew_c*100:.1f}%")
        st.write(f"- **Liquidez:** {'buena' if spread < 2 else 'regular, cuidado al entrar'} (spread {spread:.1f}%)")

        st.line_chart(h["Close"])

        st.session_state.historial.append({
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "simbolo": simbolo,
            "precio": round(precio, 2), "iv_atm": round(iv, 3), "vol_hist": round(hv, 3),
            "vix_rank": round(rank), "pc_oi": round(pc_oi, 2), "mov_esperado": round(mov, 1),
        })
    except Exception as e:
        st.error(f"No pude traer los datos de {simbolo}. Intenta de nuevo en un minuto. ({e})")

if st.session_state.historial:
    st.divider()
    st.subheader("📒 Análisis de esta sesión")
    df = pd.DataFrame(st.session_state.historial)
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Descargar registro (CSV)", df.to_csv(index=False), "registro.csv", use_container_width=True)
