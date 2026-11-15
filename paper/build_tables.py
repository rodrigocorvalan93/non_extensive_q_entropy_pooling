"""
paper/build_tables.py
=====================
Genera las tablas y los números del texto del paper, en inglés y en español,
a partir de los archivos de results/. Así ninguna cifra del paper se copia a
mano: si cambia un resultado, cambian las tablas y el texto.

Entradas (las produce reproduce_paper.py):
    results/verificaciones.json, results/barrido_q.csv
    results/stress_dureza.csv, results/stress_ranking_e_info.csv,
    results/stress_otros.csv
    results/merval_pesos_simple.csv, results/merval_diagnosticos_simple.json
    results/oos_metricas_buyhold.csv, results/oos_metricas_rebalanceo_diario.csv,
    results/oos_bootstrap_sharpe.csv, results/oos_turnover.csv
    results/merval_robustez.csv

Salidas:
    paper/tables/numbers_{en,es}.tex    macros \\n... con cada cifra del texto
    paper/tables/tab_{octave,stress,weights,oos,robust}_{en,es}.tex

El script también comprueba las afirmaciones cualitativas del texto (por
ejemplo, que q = 2 coincide con el QP a menos de 1e-15) y falla si alguna
deja de ser cierta.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "tables"

LANGS = ("en", "es")


# ═══════════════════════════════════════════════════════════════
# Formato
# ═══════════════════════════════════════════════════════════════

def f(x, d, lang, sign=False):
    """Número con d decimales; coma decimal en español; signo opcional."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0.0:      # evita "-0.00"
        s = s[1:]
    if lang == "es":
        s = s.replace(".", "{,}")
    return s


def fm(x, d, lang, sign=False):
    """Como f(), pero en modo matemático (para negativos en tablas)."""
    return f"${f(x, d, lang, sign)}$"


def sci(x, lang, d=1):
    """Notación científica a\\times10^{b} (sin los signos de dólar)."""
    b = int(np.floor(np.log10(abs(x))))
    a = x / 10 ** b
    a_s = f"{a:.{d}f}"
    if lang == "es":
        a_s = a_s.replace(".", "{,}")
    return f"{a_s}\\times 10^{{{b}}}"


def entero(n, lang):
    """Entero con separador de miles a partir de cinco cifras."""
    s = f"{int(n):,}"
    if len(str(int(n))) <= 4:
        return str(int(n))
    return s if lang == "en" else s.replace(",", ".")


# ═══════════════════════════════════════════════════════════════
# Carga
# ═══════════════════════════════════════════════════════════════

def cargar():
    d = {}
    d["ver"] = json.loads((RES / "verificaciones.json").read_text(encoding="utf-8"))
    d["stress"] = pd.read_csv(RES / "stress_dureza.csv")
    d["rank"] = pd.read_csv(RES / "stress_ranking_e_info.csv").iloc[0]
    d["otros"] = pd.read_csv(RES / "stress_otros.csv")
    d["loc"] = pd.read_csv(RES / "stress_equivalencia_local.csv")
    d["sboot"] = pd.read_csv(RES / "stress_bootstrap.csv").set_index("k")
    d["contrib"] = pd.read_csv(RES / "oos_contribuciones.csv", index_col=0)
    d["w"] = pd.read_csv(RES / "merval_pesos_simple.csv", index_col=0)
    d["diag"] = json.loads((RES / "merval_diagnosticos_simple.json").read_text(encoding="utf-8"))
    d["oos"] = pd.read_csv(RES / "oos_metricas_buyhold.csv", index_col=0)
    d["oos_d"] = pd.read_csv(RES / "oos_metricas_rebalanceo_diario.csv", index_col=0)
    d["boot"] = pd.read_csv(RES / "oos_bootstrap_sharpe.csv").set_index(["a", "b"])
    d["turn"] = pd.read_csv(RES / "oos_turnover.csv", index_col=0).iloc[:, 0]
    d["bsens"] = pd.read_csv(RES / "oos_bootstrap_sensibilidad.csv")
    d["dia"] = pd.read_csv(RES / "oos_dia_extremo.csv", index_col=0)
    d["rob"] = pd.read_csv(RES / "merval_robustez.csv").set_index("variante")
    return d


# ═══════════════════════════════════════════════════════════════
# Comprobaciones de las afirmaciones cualitativas del texto
# ═══════════════════════════════════════════════════════════════

def comprobar(d):
    v, vr = d["ver"]["base_completa"], d["ver"]["base_reducida"]
    assert abs(v["q_optimo"] - 2.0) < 1e-9, "el mínimo del barrido no está en q = 2"
    assert all(abs(q - 2.0) < 1e-9 for q in d["ver"]["views_aleatorios_q_optimo"].values()), \
        "algún view aleatorio no tiene el mínimo en q = 2"
    assert v["max_dif_python_octave"] < 1e-5
    if vr["q2_vs_qp_max_abs"] is not None:
        assert vr["q2_vs_qp_max_abs"] < 1e-15, "q = 2 y el QP difieren en más de 1e-15"
    assert vr["q_max_N1"] == 1.0 and vr["q_max_N2"] == 2.0
    assert max(vr["renyi_primal_vs_tsallis_max_abs"].values()) < 1e-8
    lim = vr["limite_q1_max_abs"]
    assert lim["1.1"] > lim["1.01"] > lim["1.001"]

    s = d["stress"].set_index("k")
    assert (s["ess_T"] >= s["ess_S"] - 1e-12).all() and (s["ens_S"] >= s["ens_T"] - 1e-12).all()
    assert (s["cvar_T"] >= s["cvar_S"] - 1e-9).all()
    loc = d["loc"]["dif_sobre_k2"]
    assert loc.max() / loc.min() < 1.15, "la diferencia entre posteriors no escala con k²"
    cola = d["otros"][d["otros"]["view"] == "cola"]
    assert cola["max_dif_posteriors_xJ"].max() < 1e-8 and cola["max_dif_vs_jeffrey_xJ"].max() < 1e-8
    vol = d["otros"][d["otros"]["view"] == "volatilidad"]
    assert (vol["ceros_T"] == 0).all()

    w = d["w"]
    assert np.abs(w["MV-eq"] - w["Benchmark"]).sum() < 0.01, "MV-eq no reproduce el benchmark"
    design = ["Benchmark", "BL", "EP-eq", "qEP-eq", "MV-hist", "BL-hist", "EP", "qEP"]
    assert w[design].max().max() < 0.30 - 1e-6, "el tope de 30% está activo en alguna cartera"
    dg = d["diag"]
    assert dg["historico_shannon"]["desigualdades_activas"] == 1
    assert dg["codificacion_relativos"] == "inequality" and dg["retornos"] == "simple"
    assert np.isnan(d["rob"].loc["igualdades mezcladas", "sharpe_EP-eq"]), \
        "la codificación con igualdades ahora tiene solución con el prior de equilibrio"
    rob = d["rob"]
    assert abs(rob.loc["views invertidos", "sharpe_BL"] - rob.loc["base", "sharpe_BL"]) < 0.005, \
        "BL cambia al invertir los views"
    # Afirmaciones de la Sección 6 agregadas en la revisión
    assert abs(w["MinVar"].max() - 0.30) < 1e-6, "la cartera de mínima varianza ya no está en el tope"
    dif_eq = rob["sharpe_EP-eq"] - rob["sharpe_Benchmark"]
    assert dif_eq.min() > -0.005, "EP con prior de equilibrio queda por debajo del benchmark en alguna variante"
    assert dif_eq.idxmax() == "confianza plena", "el máximo de EP-eq − benchmark no es la variante de confianza plena"
    assert rob.loc["confianza plena", "l1_EPeq_vs_bench"] > rob.loc["confianza plena", "l1_BL_vs_bench"], \
        "con c = 1 EP ya no mueve la cartera más que BL (prior de equilibrio)"
    assert rob.loc["views invertidos", "l1_EPeq_vs_bench"] < 0.01, \
        "con los views invertidos y prior de equilibrio la cartera de EP ya no es el benchmark"
    orden = ["Benchmark", "BL", "EP-eq", "qEP-eq", "MV-hist", "BL-hist", "EP", "qEP"]
    r0 = rob.loc["base", [f"sharpe_{c}" for c in orden]].rank().values
    r30 = rob.loc["tasa libre 30%", [f"sharpe_{c}" for c in orden]].rank().values
    assert (r0 == r30).all(), "la tasa libre de 30% cambia el orden de las carteras del diseño"
    dia = d["dia"]
    assert dia.loc["MV-hist", "sharpe_sin_dia"] > dia.loc["Benchmark", "sharpe_sin_dia"], \
        "sin la rueda de mayor movimiento cambia el orden de los dos priors"
    assert dg["precios_faltantes"] == {"VALO": 3}, "cambió la cantidad de precios interpolados"
    cero = pd.Series(dg["dias_retorno_cero_pct"])
    assert cero[["VALO", "BYMA", "ALUA"]].min() > cero[["YPFD", "GGAL", "BBAR", "BMA", "SUPV"]].max(), \
        "los tres papeles de la cartera sin views ya no tienen más ruedas sin cambio que YPFD y los bancos"
    b = d["boot"]
    desc = b.loc[("MV-hist", "Benchmark"), "diff"] + b.loc[("EP", "MV-hist"), "diff"] - b.loc[("BL", "Benchmark"), "diff"]
    assert abs(desc - b.loc[("EP", "BL"), "diff"]) < 1e-6, "la descomposición de EP − BL no cierra"

    sb = d["sboot"]
    assert (sb.loc[[20, 35, 50], "sd_media_bp_T"] < sb.loc[[20, 35, 50], "sd_media_bp_S"]).all()
    assert (sb["n_S"] == sb["replicas"]).all() and (sb["n_T"] == sb["replicas"]).all()


# ═══════════════════════════════════════════════════════════════
# Números del texto
# ═══════════════════════════════════════════════════════════════

def numeros(d, lang):
    v, vr = d["ver"]["base_completa"], d["ver"]["base_reducida"]
    s = d["stress"].set_index("k")
    rk, dg, w = d["rank"], d["diag"], d["w"]
    oos, boot, rob = d["oos"], d["boot"], d["rob"]
    N = {}

    # --- Sección 4.3
    N["MeucciN"] = str(v["N"])
    N["ShannonErr"] = f(v["error_shannon"], 6, lang)
    N["QtwoErr"] = f(float(v["errores_tabla"]["2.0"]), 6, lang)
    N["ImprovPct"] = f(v["mejora_pct_q2_vs_shannon"], 2, lang)
    N["OctaveDiff"] = f"${sci(v['max_dif_python_octave'], lang, 0)}$"
    N["LimA"] = sci(vr["limite_q1_max_abs"]["1.1"], lang)
    N["LimB"] = sci(vr["limite_q1_max_abs"]["1.01"], lang)
    N["LimC"] = sci(vr["limite_q1_max_abs"]["1.001"], lang)
    N["ShannonShift"] = sci(vr["desplazamiento_shannon_max_abs"], lang)
    # El texto dice "menos de un segundo": los tiempos dependen de la máquina
    # y de su carga, así que no se imprimen; solo se comprueba la cota.
    assert d["ver"]["resuelve_en_menos_de_1s_J100k"], "el texto dice que cada problema se resuelve en menos de un segundo"

    # --- Sección 4.4
    J = int(rk["J"])
    N["J"] = entero(J, lang)
    N["N"] = str(int(rk["N"]))
    N["SEday"] = f(rk["se_diaria_pct"], 3, lang)
    N["SEann"] = f(rk["se_diaria_pct"] * 252, 0, lang)
    N["CVaRprior"] = f(rk["cvar_prior_pct"], 2, lang)
    for k, nom in ((5, "Five"), (20, "Twenty"), (50, "Fifty")):
        r = s.loc[k]
        N[f"Ex{nom}"] = entero(round(r["exceso_anual_pp"]), lang)
        N[f"KishS{nom}"] = f(r["ess_S"], 3, lang)
        N[f"KishT{nom}"] = f(r["ess_T"], 3, lang)
        N[f"EnsS{nom}"] = f(r["ens_S"], 3, lang)
        N[f"EnsT{nom}"] = f(r["ens_T"], 3, lang)
        N[f"MaxS{nom}"] = f(r["maxw_S"], 1 if r["maxw_S"] < 100 else 0, lang)
        N[f"MaxT{nom}"] = f(r["maxw_T"], 1, lang)
        N[f"Zeros{nom}"] = str(int(r["ceros_T"]))
        N[f"CVaRS{nom}"] = f(r["cvar_S"], 2, lang)
        N[f"CVaRT{nom}"] = f(r["cvar_T"], 2, lang)
        N[f"Dw{nom}"] = f(r["dw1"], 2, lang)
    N["KishSFiftyPct"] = f(s.loc[50, "ess_S"] * 100, 0, lang)
    N["KishTFiftyPct"] = f(s.loc[50, "ess_T"] * 100, 0, lang)
    N["ZerosFiftyPct"] = f(s.loc[50, "ceros_T"] / J * 100, 0, lang)
    N["RankEss"] = f(rk["ess_S"], 3, lang)
    N["RankDw"] = f(rk["dw1"], 3, lang)
    vol = d["otros"][d["otros"]["view"] == "volatilidad"].set_index("parametro")
    N["VolKishSTwo"], N["VolKishTTwo"] = f(vol.loc[2.0, "ess_S"], 3, lang), f(vol.loc[2.0, "ess_T"], 3, lang)
    N["VolKishSThree"], N["VolKishTThree"] = f(vol.loc[3.0, "ess_S"], 3, lang), f(vol.loc[3.0, "ess_T"], 3, lang)

    # --- Sección 6: priors, views y carteras
    mh = pd.Series(dg["media_anual_pct"]["historico"])
    pi = pd.Series(dg["media_anual_pct"]["equilibrio"])
    N["MuHistMin"], N["MuHistMax"] = f(mh.min(), 0, lang), f(mh.max(), 0, lang)
    bajos, bancos = ["ALUA", "VALO", "BYMA"], ["GGAL", "BBAR", "BMA", "SUPV"]
    N["PiLowMin"], N["PiLowMax"] = f(pi[bajos].min(), 0, lang), f(pi[bajos].max(), 0, lang)
    N["PiBankMin"], N["PiBankMax"] = f(pi[bancos].min(), 0, lang), f(pi[bancos].max(), 0, lang)
    N["Ineq"] = str(dg["historico_shannon"]["desigualdades"])
    N["ActiveEq"] = str(dg["equilibrio_shannon"]["desigualdades_activas"])
    N["EssHist"] = f(dg["historico_shannon"]["ess_kish_sobre_J"] * 100, 2, lang)
    N["EssEq"] = f(dg["equilibrio_shannon"]["ess_kish_sobre_J"] * 100, 2, lang)
    N["ShiftEq"] = f(max(dg["equilibrio_shannon"]["max_cambio_media_pp_anual"],
                         dg["historico_shannon"]["max_cambio_media_pp_anual"]), 0, lang)
    l1 = lambda a, b: float(np.abs(w[a] - w[b]).sum())
    N["LonePrior"] = f(l1("MV-hist", "Benchmark"), 2, lang)
    N["LoneEPhist"] = f(l1("EP", "MV-hist"), 2, lang)
    N["LoneBLhist"] = f(l1("BL-hist", "MV-hist"), 2, lang)
    N["LoneEPeq"] = f(l1("EP-eq", "Benchmark"), 2, lang)
    N["LoneBLeq"] = f(l1("BL", "Benchmark"), 2, lang)
    N["LoneQhist"] = f(l1("EP", "qEP"), 4, lang)
    N["LoneQeq"] = f(l1("EP-eq", "qEP-eq"), 3, lang)
    N["LoneEPeqFull"] = f(rob.loc["confianza plena", "l1_EPeq_vs_bench"], 2, lang)
    # Proxy de iliquidez: ruedas con retorno cero (% de la muestra de estimación)
    cero = pd.Series(dg["dias_retorno_cero_pct"])
    iliq, liq = ["VALO", "BYMA", "ALUA"], ["YPFD", "GGAL", "BBAR", "BMA", "SUPV"]
    N["ZeroIlliqMin"], N["ZeroIlliqMax"] = f(cero[iliq].min(), 1, lang), f(cero[iliq].max(), 1, lang)
    N["ZeroLiqMin"], N["ZeroLiqMax"] = f(cero[liq].min(), 1, lang), f(cero[liq].max(), 1, lang)
    N["WIlliq"] = f(w.loc[iliq, "MV-hist"].sum() * 100, 0, lang)

    # --- Sección 6: fuera de muestra
    N["Days"] = str(int(oos["Días en el período"].iloc[0]))
    sh, ret = oos["Sharpe Ratio"], oos["Retorno acumulado (%)"]
    for col, nom in (("Benchmark", "Bench"), ("BL", "BL"), ("EP-eq", "EPeq"), ("MV-hist", "MVhist"),
                     ("BL-hist", "BLhist"), ("EP", "EP"), ("1/N", "EW"), ("MinVar", "MinVar")):
        N[f"Sh{nom}"] = f(sh[col], 2, lang)
        N[f"Ret{nom}"] = f(ret[col], 1, lang)
    N["ShBenchDaily"] = f(d["oos_d"].loc["Benchmark", "Sharpe Ratio"], 2, lang)
    for (a, b), nom in ((("EP", "MV-hist"), "EP"), (("BL", "Benchmark"), "BL"),
                        (("EP-eq", "Benchmark"), "EPeq"), (("MV-hist", "Benchmark"), "Prior")):
        r = boot.loc[(a, b)]
        N[f"Boot{nom}diff"] = f(r["diff"], 2, lang, sign=True)
        N[f"Boot{nom}lo"] = f(r["ci_lo"], 2, lang, sign=True)
        N[f"Boot{nom}hi"] = f(r["ci_hi"], 2, lang, sign=True)
    r = boot.loc[("EP", "BL")]              # la comparación habitual: EP histórico contra BL de equilibrio
    N["BootUsualdiff"] = f(r["diff"], 2, lang, sign=True)
    N["BootUsuallo"] = f(r["ci_lo"], 2, lang, sign=True)
    N["BootUsualhi"] = f(r["ci_hi"], 2, lang, sign=True)
    lo = d["bsens"]["ci_lo"]                # extremo inferior según el bloque y la semilla
    N["BootPriorLoMin"] = f(np.floor(lo.min() * 100) / 100, 2, lang, sign=True)
    N["BootPriorLoMax"] = f(np.ceil(lo.max() * 100) / 100, 2, lang, sign=True)
    dia = d["dia"]                          # rueda de mayor movimiento del benchmark
    fecha = pd.Timestamp(dia.loc["Benchmark", "fecha"])
    meses_en = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"]
    meses_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                "septiembre", "octubre", "noviembre", "diciembre"]
    N["ExtDay"] = (f"{fecha.day}~{meses_en[fecha.month - 1]}~{fecha.year}" if lang == "en"
                   else f"{fecha.day} de {meses_es[fecha.month - 1]} de {fecha.year}")
    N["ExtBench"] = f(dia.loc["Benchmark", "retorno_dia_pct"], 1, lang)
    N["ExtMVhist"] = f(dia.loc["MV-hist", "retorno_dia_pct"], 1, lang)
    N["ShBenchEx"] = f(dia.loc["Benchmark", "sharpe_sin_dia"], 2, lang)
    N["ShMVhistEx"] = f(dia.loc["MV-hist", "sharpe_sin_dia"], 2, lang)
    N["TurnEP"] = f(d["turn"]["EP"] * 100, 0, lang) + r"\%"
    N["CostEP"] = f(d["turn"]["EP"] * 0.5, 1, lang)

    # --- Variabilidad muestral (bootstrap) y contribuciones
    sb = d["sboot"]
    N["BootReps"] = str(int(sb["replicas"].iloc[0]))
    for k, nom in ((20, "Twenty"), (50, "Fifty")):
        N[f"BootSdS{nom}"] = f(sb.loc[k, "sd_media_bp_S"], 1, lang)
        N[f"BootSdT{nom}"] = f(sb.loc[k, "sd_media_bp_T"], 1, lang)
    c = d["contrib"]
    dif_c = c["MV-hist"] - c["Benchmark"]
    N["ContribTop"] = f(dif_c[["VALO", "BYMA"]].sum(), 1, lang)
    N["ContribTotal"] = f(dif_c.sum(), 1, lang)
    win = rob.loc[["ventana 5 anios", "ventana 3 anios"], "sharpe_MV-hist"]
    N["ShWinNVmin"], N["ShWinNVmax"] = f(win.min(), 2, lang), f(win.max(), 2, lang)
    N["ShPlaceboEP"] = f(rob.loc["views invertidos", "sharpe_EP"], 2, lang)
    N["ShPlaceboEPeq"] = f(rob.loc["views invertidos", "sharpe_EP-eq"], 2, lang)
    N["ShBenchRf"] = f(rob.loc["tasa libre 30%", "sharpe_Benchmark"], 2, lang)
    N["ShMVhistRf"] = f(rob.loc["tasa libre 30%", "sharpe_MV-hist"], 2, lang)

    # --- Robustez
    dif = (rob["sharpe_EP"] - rob["sharpe_MV-hist"])
    N["RobMin"] = f(dif.min(), 2, lang, sign=True)
    N["RobMax"] = f(dif.max(), 2, lang, sign=True)
    qmax = rob["l1_EP_vs_qEP"].max()
    N["RobQmax"] = f(np.ceil(qmax * 1000) / 1000, 3, lang)
    dif_eq = rob["sharpe_EP-eq"] - rob["sharpe_Benchmark"]      # prior de equilibrio
    N["RobEqMax"] = f(dif_eq.max(), 2, lang, sign=True)
    N["RobQmaxEq"] = f(np.ceil(rob["l1_EPeq_vs_qEPeq"].max() * 100) / 100, 2, lang)
    return N


def escribir_numeros(N, lang):
    lineas = ["% Generado por paper/build_tables.py. No editar a mano."]
    for k in sorted(N):
        assert k.isalpha(), k
        lineas.append(f"\\newcommand{{\\n{k}}}{{{N[k]}}}")
    (OUT / f"numbers_{lang}.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# Tablas
# ═══════════════════════════════════════════════════════════════

def tab_octave(d, lang):
    v = d["ver"]["base_completa"]
    qs = ["1.5", "1.9", "2.0", "2.1", "2.5"]
    cab = " & Shannon & " + " & ".join(f"$q={f(float(q), 1, lang)}$" for q in qs) + r" \\"
    fila = lambda nom, src: nom + " & " + " & ".join(
        f(float(src[k]), 6, lang) for k in ["Shannon"] + qs) + r" \\"
    return "\n".join([r"\begin{tabular}{lcccccc}", r"\toprule", cab, r"\midrule",
                      fila("Octave", v["octave"]), fila("Python", v["errores_tabla"]),
                      r"\bottomrule", r"\end{tabular}"]) + "\n"


def tab_stress(d, lang):
    s, rk = d["stress"], d["rank"]
    tx = {"en": ("Zeros", "ranking"), "es": ("Ceros", "ranking")}[lang]
    L = [r"\begin{tabular}{rrccccrccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{$N_2/J$} & \multicolumn{2}{c}{$N_1/J$} & & "
         r"\multicolumn{2}{c}{CVaR} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}",
         rf"$k$ & $\Delta\mu$ & EP & $q=2$ & EP & $q=2$ & {tx[0]} & EP & $q=2$ & "
         r"$\norm{\Delta w}_1$ \\", r"\midrule"]
    for _, r in s.iterrows():
        L.append(" & ".join([
            str(int(r["k"])), entero(round(r["exceso_anual_pp"]), lang),
            f(r["ess_S"], 3, lang), f(r["ess_T"], 3, lang),
            f(r["ens_S"], 3, lang), f(r["ens_T"], 3, lang),
            str(int(r["ceros_T"])),
            fm(r["cvar_S"], 2, lang), fm(r["cvar_T"], 2, lang),
            f(r["dw1"], 2, lang)]) + r" \\")
    L += [r"\midrule",
          " & ".join([tx[1], "--", f(rk["ess_S"], 3, lang), f(rk["ess_T"], 3, lang),
                      f(rk["ens_S"], 3, lang), f(rk["ens_T"], 3, lang),
                      str(int(rk["ceros_T"])), "--", "--", f(rk["dw1"], 2, lang)]) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tab_weights(d, lang):
    w = d["w"]
    tx = {"en": ("Equilibrium prior", "Historical prior", "No views", r"$\norm{w - w_{\text{no views}}}_1$"),
          "es": ("Prior de equilibrio", "Prior hist\\'orico", "Sin views", r"$\norm{w - w_{\text{sin views}}}_1$")}[lang]
    cols = ["Benchmark", "BL", "EP-eq", "qEP-eq", "MV-hist", "BL-hist", "EP", "qEP"]
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         rf" & \multicolumn{{4}}{{c}}{{{tx[0]}}} & \multicolumn{{4}}{{c}}{{{tx[1]}}} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         rf" & {tx[2]} & BL & EP & $q=2$ & {tx[2]} & BL & EP & $q=2$ \\", r"\midrule"]
    for t in w.index:
        L.append(t + " & " + " & ".join(f(w.loc[t, c] * 100, 2, lang) for c in cols) + r" \\")
    dist = lambda c, ctrl: f(float(np.abs(w[c] - w[ctrl]).sum()), 2, lang)
    L += [r"\midrule",
          tx[3] + " & -- & " + " & ".join(dist(c, "Benchmark") for c in cols[1:4])
          + " & -- & " + " & ".join(dist(c, "MV-hist") for c in cols[5:]) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tab_oos(d, lang):
    oos, boot = d["oos"], d["boot"]
    tx = {"en": dict(eq="Equilibrium prior", hi="Historical prior", ref="Reference portfolios",
                     nv_eq="No views (benchmark)", nv="No views", ew="Equally weighted",
                     mv="Minimum variance",
                     cab=r" & Return & Vol. & Sharpe & Max.\ DD & TE & Beta & $\Delta$ Sharpe \\",
                     sub=r" & (\%) & (\%) & & (\%) & (\%) & & [95\% interval] \\"),
          "es": dict(eq="Prior de equilibrio", hi="Prior hist\\'orico", ref="Carteras de referencia",
                     nv_eq="Sin views (benchmark)", nv="Sin views", ew="Equiponderada",
                     mv="M\\'inima varianza",
                     cab=r" & Retorno & Vol. & Sharpe & Ca\'ida m\'ax. & TE & Beta & $\Delta$ Sharpe \\",
                     sub=r" & (\%) & (\%) & & (\%) & (\%) & & [intervalo de 95\%] \\")}[lang]

    def fila(nombre, col, ctrl=None):
        r = oos.loc[col]
        te = r.get("Tracking Error anualizado (%)", np.nan)
        beta = r.get("Beta", np.nan)
        celdas = [nombre, f(r["Retorno acumulado (%)"], 1, lang),
                  f(r["Volatilidad anualizada (%)"], 1, lang), f(r["Sharpe Ratio"], 2, lang),
                  fm(r["Max Drawdown (%)"], 1, lang),
                  "--" if pd.isna(te) else f(te, 1, lang),
                  "--" if pd.isna(beta) else f(beta, 2, lang)]
        if ctrl is None:
            celdas.append("--")
        else:
            b = boot.loc[(col, ctrl)]
            celdas.append(f"${f(b['diff'], 2, lang, sign=True)}$ "
                          f"$[{f(b['ci_lo'], 2, lang, sign=True)}{';' if lang == 'es' else ','}\\, "
                          f"{f(b['ci_hi'], 2, lang, sign=True)}]$")
        return " & ".join(celdas) + r" \\"

    grupo = lambda t: rf"\multicolumn{{8}}{{l}}{{\emph{{{t}}}}} \\"
    L = [r"\begin{tabular}{lrrrrrrl}", r"\toprule", tx["cab"], tx["sub"], r"\midrule",
         grupo(tx["eq"]),
         fila(r"\quad " + tx["nv_eq"], "Benchmark"),
         fila(r"\quad BL", "BL", "Benchmark"),
         fila(r"\quad EP", "EP-eq", "Benchmark"),
         fila(r"\quad EP, $q=2$", "qEP-eq", "Benchmark"),
         r"\addlinespace", grupo(tx["hi"]),
         fila(r"\quad " + tx["nv"], "MV-hist"),
         fila(r"\quad BL", "BL-hist", "MV-hist"),
         fila(r"\quad EP", "EP", "MV-hist"),
         fila(r"\quad EP, $q=2$", "qEP", "MV-hist"),
         r"\addlinespace", grupo(tx["ref"]),
         fila(r"\quad " + tx["ew"], "1/N"),
         fila(r"\quad " + tx["mv"], "MinVar"),
         r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


VARIANTES = {  # nombre en el CSV → (inglés, español)
    "base": ("Base case", "Caso base"),
    "igualdades mezcladas": ("Blended equalities", "Igualdades mezcladas"),
    "confianza plena": ("Full confidence, $c=1$", "Confianza plena, $c=1$"),
    "q = 1,5": ("$q = 1.5$", "$q = 1{,}5$"),
    "q = 3": ("$q = 3$", "$q = 3$"),
    "tope 20%": (r"Cap 20\%", r"Tope de 20\%"),
    "tope 40%": (r"Cap 40\%", r"Tope de 40\%"),
    "sin tope": ("No cap", "Sin tope"),
    "delta = 1": (r"$\delta = 1$", r"$\delta = 1$"),
    "delta = 5": (r"$\delta = 5$", r"$\delta = 5$"),
    "retornos log": ("Log returns", "Retornos logar\\'itmicos"),
    "ventana 5 anios": ("Estimation sample: last 5 years", "Muestra de estimaci\\'on: \\'ultimos 5 a\\~nos"),
    "ventana 3 anios": ("Estimation sample: last 3 years", "Muestra de estimaci\\'on: \\'ultimos 3 a\\~nos"),
    "views invertidos": ("Reversed views (placebo)", "Views invertidos (placebo)"),
    "rebalanceo diario": ("Daily rebalancing", "Rebalanceo diario"),
    "tasa libre 30%": (r"Risk-free rate of 30\%", r"Tasa libre de riesgo de 30\%"),
}


def tab_robust(d, lang):
    rob = d["rob"]
    i = 0 if lang == "en" else 1
    tx = {"en": ("Equilibrium", "Historical prior", "No views", r"$\norm{\Delta w}_1$",
                 "views", "$q$"),
          "es": ("Equilibrio", "Prior hist\\'orico", "Sin views", r"$\norm{\Delta w}_1$",
                 "views", "$q$")}[lang]
    L = [r"\begin{tabular}{lcccccccc}", r"\toprule",
         rf" & \multicolumn{{2}}{{c}}{{{tx[0]}}} & \multicolumn{{4}}{{c}}{{{tx[1]}}} & "
         rf"\multicolumn{{2}}{{c}}{{{tx[3]}}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
         rf" & BL & EP & {tx[2]} & BL & EP & $q$ & {tx[4]} & {tx[5]} \\", r"\midrule"]
    cel = lambda x, dd: "--" if pd.isna(x) else f(x, dd, lang)
    for clave, nombres in VARIANTES.items():
        r = rob.loc[clave]
        L.append(" & ".join([nombres[i], cel(r["sharpe_BL"], 2), cel(r["sharpe_EP-eq"], 2),
                             cel(r["sharpe_MV-hist"], 2), cel(r["sharpe_BL-hist"], 2),
                             cel(r["sharpe_EP"], 2), cel(r["sharpe_qEP"], 2),
                             cel(r["l1_EP_vs_MVhist"], 2), cel(r["l1_EP_vs_qEP"], 4)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def tab_boot(d, lang):
    sb = d["sboot"]
    tx = {"en": ("SD of the posterior mean (bp)", "CV of the posterior volatility"),
          "es": ("Desv\\'io de la media posterior (pb)", "CV de la volatilidad posterior")}[lang]
    L = [r"\begin{tabular}{rcccc}", r"\toprule",
         rf" & \multicolumn{{2}}{{c}}{{{tx[0]}}} & \multicolumn{{2}}{{c}}{{{tx[1]}}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"$k$ & EP & $q=2$ & EP & $q=2$ \\", r"\midrule"]
    for k, r in sb.iterrows():
        L.append(" & ".join([str(int(k)), f(r["sd_media_bp_S"], 1, lang), f(r["sd_media_bp_T"], 1, lang),
                             f(r["cv_vol_S"], 3, lang), f(r["cv_vol_T"], 3, lang)]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = cargar()
    comprobar(d)
    for lang in LANGS:
        escribir_numeros(numeros(d, lang), lang)
        for nombre, fn in (("octave", tab_octave), ("stress", tab_stress), ("weights", tab_weights),
                           ("oos", tab_oos), ("robust", tab_robust), ("boot", tab_boot)):
            (OUT / f"tab_{nombre}_{lang}.tex").write_text(fn(d, lang), encoding="utf-8")
    print(f"Tablas y números escritos en {OUT}")


if __name__ == "__main__":
    main()
