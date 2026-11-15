"""
reproduce_paper.py
==================
Regenera, desde un clon limpio del repositorio, todos los números, tablas y
figuras del paper.

    python reproduce_paper.py            # todo
    python reproduce_paper.py --skip-tests

Pasos (cada uno es un script que también se puede correr por separado):

  1. s_main_verificaciones.py     Sección 4.3 y Apéndice B: barrido de q sobre
                                  la base de Meucci, comparación con Octave,
                                  QP independiente, límite q→1, Rényi, tiempos.
                                  → results/verificaciones.json, Figura 1
  2. s_main_qep_vs_ep.py          Sección 4.4: experimento de stress.
                                  → results/stress_*.csv, Figura 2
  3. s_main_merval.py             Sección 6: carteras (diseño prior × método).
                                  → results/merval_pesos_*.csv
  4. portfolio_evolution.py       Sección 6: evaluación fuera de muestra.
                                  → results/oos_*.csv, Figura 3
  5. s_main_merval_robustez.py    Apéndice C: robustez.
                                  → results/merval_robustez.csv
  6. paper/build_tables.py        Tablas y números del texto, en inglés y en
                                  español. → paper/tables/*.tex
  7. pytest tests                 Propiedades del solver.

Para compilar los PDF después:  cd paper && pdflatex paper_en.tex (dos veces).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

ACCIONES = Path("data") / "dataacciones"
PRECIOS = [ACCIONES / "input_mkt_px.xlsx", ACCIONES / "input_mkt_w.xlsx"]
PRECIOS_OOS = PRECIOS + [ACCIONES / "input_current_mkt_px.xlsx"]

# (nombre, comando, archivos de datos de mercado que el paso necesita)
PASOS = [
    ("Verificaciones numéricas (Sección 4.3 y Apéndice B)", [sys.executable, "s_main_verificaciones.py"], []),
    ("Experimento de stress (Sección 4.4)", [sys.executable, "s_main_qep_vs_ep.py"], PRECIOS),
    ("Carteras del Merval (Sección 6)", [sys.executable, "s_main_merval.py"], PRECIOS),
    ("Evaluación fuera de muestra (Sección 6)", [sys.executable, "portfolio_evolution.py"], PRECIOS_OOS),
    ("Robustez (Apéndice C)", [sys.executable, "s_main_merval_robustez.py"], PRECIOS_OOS),
    ("Tablas y números del paper", [sys.executable, str(Path("paper") / "build_tables.py")], []),
]
TESTS = ("Tests del solver", [sys.executable, "-m", "pytest", "tests", "-q"], [])


def main() -> int:
    pasos = list(PASOS)
    if "--skip-tests" not in sys.argv:
        pasos.append(TESTS)
    (HERE / "results").mkdir(exist_ok=True)
    salteados = 0
    for i, (nombre, cmd, datos) in enumerate(pasos, 1):
        print(f"\n[{i}/{len(pasos)}] {nombre}")
        faltan = [str(f) for f in datos if not (HERE / f).exists()]
        if faltan:
            # Sin los datos de mercado el paso no se puede rehacer: se conserva
            # su salida, que está en results/, y las tablas se arman con ella.
            salteados += 1
            print(f"      se saltea: falta {', '.join(faltan)} (ver data/README.md); "
                  "se usa la salida guardada en results/")
            continue
        t0 = time.time()
        log = HERE / "results" / f"log_{i:02d}.txt"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        estado = "ok" if proc.returncode == 0 else f"FALLÓ (código {proc.returncode})"
        print(f"      {estado} en {time.time() - t0:.0f} s — salida completa en {log.relative_to(HERE)}")
        if proc.returncode != 0:
            print(log.read_text(encoding="utf-8")[-2000:])
            return proc.returncode
    print("\nListo. Resultados en results/, figuras en paper/figs/, tablas en paper/tables/.")
    if salteados:
        print(f"{salteados} paso(s) salteado(s) por falta de datos de mercado: sus resultados son los guardados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
