import math
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date

st.set_page_config(page_title="Spreads de crédito", page_icon="🛡️", layout="centered")
st.title("🛡️ Spreads de crédito")
st.caption("Modo simulado · No es asesoría financiera · Riesgo máximo definido")

EVENTOS = {
    "2026-11-06": "Reporte de empleo",
    "2026-11-10": "Inflación (CPI)",
    "2026-12-09": "Decisión de la Fed",
}
TASA = 0.04
COLS = ["id", "abierto", "simbolo", "tipo", "vence", "corto", "largo", "credito",
        "contratos", "max_perdida", "estado", "cierre", "resultado"]

if "diario" not in st.session_state:
    st.session_state.diario = []


def e(s):
    return s.replace("$", "\\$")


def m(x):
    return f"-${abs(x):.0f}" if x < 0 else f"${x:.0f}"


def f(d):
    return datetime.strptime(d, "%Y-%m-%d")


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


ncdf_v = np.vectorize(ncdf)


def prob_arriba(precio, nivel, iv, T):
    d2 = (math.log(precio / nivel) + (TASA - iv ** 2 / 2) * T) / (iv * math.sqrt(T))
    return ncdf(d2)


def eventos_cercanos(dias=2):
    out = []
    for d, n in EVENTOS.items():
        faltan = (f(d).date() - date.today()).days
        if 0 <= faltan <= dias:
            out.append(f"{n} (en {faltan} días)")
    return out


@st.cache_data(ttl=300, show_spinner=False)
def datos(sim):
    t = yf.Ticker(sim)
    return t.history(period="6mo")["Close"], list(t.options)


@st.cache_data(ttl=300, show_spinner=False)
def cadena(sim, venc):
    ch = yf.Ticker(sim).option_chain(venc)
    return ch.calls, ch.puts


def disenar(sim, es_put, delta_obj, ancho, capital, riesgo_pct):
    h, vencs = datos(sim)
    precio = h.iloc[-1]
    cand = [(v, (f(v) - datetime.now()).days) for v in vencs]
    cand = [(v, d) for v, d in cand if 25 <= d <= 50]
    if not cand:
        return None, "No hay vencimientos entre 25 y 50 días."
    venc, dias = min(cand, key=lambda x: abs(x[1] - 38))
    T = dias / 365
    calls, puts = cadena(sim, venc)
    base = (puts if es_put else calls).copy()

    tab = base[(base.bid > 0) & (base.ask > 0) & (base.impliedVolatility > 0.01)]
    tab = tab[tab.strike < precio] if es_put else tab[tab.strike > precio]
    if tab.empty:
        return None, "No encontré opciones válidas fuera del dinero."
    iv_ = tab.impliedVolatility.values
    K = tab.strike.values
    d1 = (np.log(precio / K) + (TASA + iv_ ** 2 / 2) * T) / (iv_ * np.sqrt(T))
    tab = tab.copy()
    tab["delta"] = (1 - ncdf_v(d1)) if es_put else ncdf_v(d1)
    corto = tab.iloc[(tab.delta - delta_obj).abs().argsort()[:1]].iloc[0]

    objetivo = corto.strike - ancho if es_put else corto.strike + ancho
    pool = base[base.ask > 0]
    pool = pool[pool.strike < corto.strike] if es_put else pool[pool.strike > corto.strike]
    if pool.empty:
        return None, "No encontré el strike de protección."
    largo = pool.iloc[(pool.strike - objetivo).abs().argsort()[:1]].iloc[0]

    ancho_real = abs(corto.strike - largo.strike)
    cred_nat = corto.bid - largo.ask
    cred_mid = (corto.bid + corto.ask) / 2 - (largo.bid + largo.ask) / 2
    if cred_nat <= 0:
        return None, "Con precios reales, este spread no da crédito positivo. Prueba otro ancho o delta."

    max_perd = (ancho_real - cred_nat) * 100
    max_gan = cred_nat * 100
    be = corto.strike - cred_nat if es_put else corto.strike + cred_nat
    iv_c = corto.impliedVolatility
    p_ar = prob_arriba(precio, be, iv_c, T)
    p = p_ar if es_put else 1 - p_ar
    hv = np.log(h).diff().tail(30).std() * math.sqrt(252)
    atm = base[(base.impliedVolatility > 0.01)]
    atm_iv = atm.iloc[(atm.strike - precio).abs().argsort()[:1]].impliedVolatility.iloc[0]
    ratio = atm_iv / hv if hv > 0 else 9
    ev = p * max_gan - (1 - p) * max_perd
    presupuesto = capital * riesgo_pct / 100
    n = int(presupuesto // max_perd) if max_perd > 0 else 0
    sma50 = h.tail(50).mean()

    checks = []
    ev_c = eventos_cercanos()
    checks.append((not ev_c, "Sin eventos macro en 2 días" if not ev_c else "Evento cercano: " + ", ".join(ev_c)))
    checks.append((ratio >= 1.2, f"Prima vale la pena (IV/vol real = {ratio:.2f}, mínimo 1.20)"))
    tend_ok = precio > sma50 if es_put else precio < sma50
    checks.append((tend_ok, "Tendencia a favor" if tend_ok else "Tendencia en contra del spread"))
    checks.append((ev > 0, f"Ganancia esperada positiva ({m(ev)} por contrato)" if ev > 0
                   else f"Ganancia esperada NEGATIVA ({m(ev)} por contrato)"))
    checks.append((corto.openInterest >= 300, f"Liquidez del strike vendido (interés abierto {int(corto.openInterest)})"))
    checks.append((n >= 1, f"Cabe en tu límite de riesgo ({n} contrato(s))" if n >= 1
                   else f"Un contrato arriesga {m(max_perd)} y tu límite es {m(presupuesto)}"))
    checks.append((p >= 0.75, f"Probabilidad de éxito {p*100:.0f}% (mínimo 75%)"))
    ok = all(c[0] for c in checks)

    return dict(sim=sim, es_put=es_put, precio=precio, venc=venc, dias=dias,
                corto=corto.strike, largo=largo.strike, cred_nat=cred_nat, cred_mid=cred_mid,
                max_perd=max_perd, max_gan=max_gan, be=be, p=p, ev=ev, n=n,
                delta=float(corto.delta), checks=checks, ok=ok), None


def costo_cierre(r):
    try:
        calls, puts = cadena(r["simbolo"], r["vence"])
        tab = puts if "PUT" in r["tipo"] else calls
        c = tab.loc[(tab.strike - float(r["corto"])).abs().idxmin()]
        l = tab.loc[(tab.strike - float(r["largo"])).abs().idxmin()]
        return max(float(c.ask) - float(l.bid), 0.0)
    except Exception:
        return None


tab1, tab2 = st.tabs(["🛠️ Diseñar spread", "📒 Diario y desempeño"])

with tab1:
    st.write("Cobras dinero por apostar a que el precio **no** llegará a cierto nivel, "
             "con una protección que limita tu pérdida.")
    sim = st.text_input("Símbolo", "SPY").upper().strip()
    tipo = st.radio("Tipo", ["PUT spread (apuesto a que NO baja de cierto nivel)",
                             "CALL spread (apuesto a que NO sube de cierto nivel)"])
    es_put = tipo.startswith("PUT")
    capital = st.number_input("Capital simulado (USD)", 500, 1000000, 3000, step=500)
    riesgo_pct = st.slider("Riesgo máximo por operación (% del capital)", 1.0, 5.0, 2.0, 0.5)
    delta_obj = st.slider("Delta del strike vendido (más bajo = más probable ganar, menos crédito)", 0.10, 0.35, 0.20, 0.01)
    ancho = st.number_input("Ancho del spread (USD entre strikes)", 1.0, 50.0, 5.0, step=1.0)

    if st.button("🛠️ Diseñar spread", use_container_width=True, type="primary"):
        try:
            res, err = disenar(sim, es_put, delta_obj, ancho, capital, riesgo_pct)
            if err:
                st.session_state.pop("sp", None)
                st.error(err)
            else:
                st.session_state.sp = res
        except Exception as ex:
            st.session_state.pop("sp", None)
            st.error(f"No pude analizar {sim}. Intenta en un minuto. ({ex})")

    r = st.session_state.get("sp")
    if r:
        nombre = "PUT" if r["es_put"] else "CALL"
        st.subheader(e(f"{r['sim']} · ${r['precio']:.2f} · vence {r['venc']} ({r['dias']} días)"))
        if r["ok"]:
            st.success("✅ ABRIR (simulado): pasa las 7 reglas")
        else:
            st.warning("⏳ ESPERAR: no pasa todas las reglas")
        for ok, txt in r["checks"]:
            st.write(e(("✅ " if ok else "❌ ") + txt))

        a, b = st.columns(2)
        a.metric("Crédito (conservador)", m(r["cred_nat"] * 100))
        b.metric("Pérdida máxima", m(r["max_perd"]))
        c, d = st.columns(2)
        c.metric("Punto de equilibrio", f"${r['be']:.2f}")
        d.metric("Probabilidad de éxito", f"{r['p']*100:.0f}%")
        st.caption(e(f"Ganancia máxima {m(r['max_gan'])} por contrato. Por cada $1 que ganas, arriesgas "
                     f"${r['max_perd']/max(r['max_gan'],1):.1f}."))

        if r["n"] >= 1:
            st.info(e(f"**Orden para Robinhood ({r['n']} contrato(s))**\n\n"
                      f"Vender 1 {nombre} {r['corto']:g} · Comprar 1 {nombre} {r['largo']:g} · vence {r['venc']}\n\n"
                      f"Precio límite sugerido (crédito): **${r['cred_mid']:.2f}** por acción. "
                      f"En el peor caso cobrarías ${r['cred_nat']:.2f}."))
        else:
            st.info(e(f"Con tu límite actual no cabe ni un contrato (arriesga {m(r['max_perd'])}). "
                      "Reduce el ancho del spread o sube el capital o el % de riesgo."))
        st.caption("Cierra al 50% de la ganancia, si la pérdida llega a 2 veces el crédito, "
                   "o cuando falten 21 días para el vencimiento.")

        forzar = False
        if not r["ok"]:
            forzar = st.checkbox("Registrar de todos modos (no recomendado)")
        if r["n"] >= 1 and (r["ok"] or forzar):
            if st.button("📝 Registrar operación simulada", use_container_width=True):
                st.session_state.diario.append(dict(
                    id=datetime.now().strftime("%m%d%H%M%S"),
                    abierto=datetime.now().strftime("%Y-%m-%d %H:%M"), simbolo=r["sim"],
                    tipo=f"{nombre} SPREAD", vence=r["venc"], corto=r["corto"], largo=r["largo"],
                    credito=round(r["cred_nat"], 2), contratos=r["n"],
                    max_perdida=round(r["max_perd"] * r["n"], 2), estado="ABIERTO",
                    cierre="", resultado=""))
                st.success("Registrada. Revisa la pestaña Diario y descarga el CSV para no perderla.")

with tab2:
    arch = st.file_uploader("Cargar diario guardado (CSV)", type="csv")
    if arch is not None:
        clave = f"{arch.name}-{arch.size}"
        if st.session_state.get("cargado") != clave:
            dfc = pd.read_csv(arch, dtype={"id": str}).fillna("")
            st.session_state.diario = dfc.to_dict("records")
            st.session_state.cargado = clave

    diario = st.session_state.diario
    if not diario:
        st.info("Aún no hay operaciones. Diseña un spread y regístralo.")
    else:
        df = pd.DataFrame(diario, columns=COLS)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Descargar diario (CSV)", df.to_csv(index=False), "diario.csv",
                           use_container_width=True)

        abiertas = [x for x in diario if x["estado"] == "ABIERTO"]
        if abiertas:
            st.subheader("📊 Operaciones abiertas")
            filas = []
            for x in abiertas:
                costo = costo_cierre(x)
                dias = (f(x["vence"]) - datetime.now()).days
                if costo is None:
                    filas.append(dict(id=x["id"], simbolo=x["simbolo"], costo_cerrar="sin datos",
                                      pnl_actual="-", accion="Revisar manualmente (¿ya venció?)"))
                    continue
                cred = float(x["credito"])
                pnl = (cred - costo) * 100 * int(x["contratos"])
                if cred - costo >= 0.5 * cred:
                    acc = "CERRAR: 50% de la ganancia"
                elif costo >= 3 * cred:
                    acc = "CERRAR: pérdida de 2x el crédito"
                elif dias <= 21:
                    acc = "CERRAR: quedan 21 días o menos"
                else:
                    acc = "MANTENER"
                filas.append(dict(id=x["id"], simbolo=x["simbolo"], costo_cerrar=round(costo, 2),
                                  pnl_actual=round(pnl, 0), dias=dias, accion=acc))
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

            st.subheader("✅ Cerrar una operación")
            idc = st.selectbox("Operación", [x["id"] for x in abiertas])
            px = st.number_input("Precio de cierre por acción (0 si venció sin valor)", 0.0, 1000.0, 0.0, 0.01)
            if st.button("Cerrar operación", use_container_width=True):
                for x in diario:
                    if x["id"] == idc:
                        x["estado"] = "CERRADO"
                        x["cierre"] = px
                        x["resultado"] = round((float(x["credito"]) - px) * 100 * int(x["contratos"]), 2)
                st.rerun()

        cerr = pd.to_numeric(df[df.estado == "CERRADO"]["resultado"], errors="coerce").dropna().tolist()
        if cerr:
            st.subheader("📈 Desempeño")
            gan = [x for x in cerr if x > 0]
            per = [x for x in cerr if x <= 0]
            peor_racha = racha = 0
            for x in cerr:
                racha = racha + 1 if x < 0 else 0
                peor_racha = max(peor_racha, racha)
            k1, k2 = st.columns(2)
            k1.metric("Operaciones cerradas", len(cerr))
            k2.metric("Acierto", f"{len(gan)/len(cerr)*100:.0f}%")
            k3, k4 = st.columns(2)
            k3.metric("Ganancia promedio", m(np.mean(gan)) if gan else "-")
            k4.metric("Pérdida promedio", m(np.mean(per)) if per else "-")
            k5, k6 = st.columns(2)
            k5.metric("Resultado por operación", m(np.mean(cerr)))
            k6.metric("Resultado total", m(sum(cerr)))
            st.write(e(f"Peor pérdida: **{m(min(cerr))}** · Peor racha de pérdidas: **{peor_racha}**"))
            if len(cerr) < 50:
                st.warning(f"Con {len(cerr)} operaciones todavía no hay evidencia suficiente. Objetivo: 50 o más.")
