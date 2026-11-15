# Non-Extensive q-Entropy Pooling

**Asignación cuantitativa de portafolios basada en entropía no extensiva: integración del *view* del portfolio manager en mercados no gaussianos**

*Non-Extensive Entropy Pooling: Integrating the Portfolio Manager's Views into Non-Gaussian Markets*

Código y resultados del documento de trabajo de Rodrigo Corvalán Salguero:

- inglés: [`paper/paper_en.pdf`](paper/paper_en.pdf)
- español: [`paper/paper_es.pdf`](paper/paper_es.pdf)

> **In short (English).** Entropy Pooling (Meucci, 2008) with the Kullback–Leibler divergence replaced by the Tsallis relative entropy of order *q* (the Cressie–Read power-divergence family). The repository contains the solver, the scripts that produce every table and figure of the paper, and a test suite. `python reproduce_paper.py` regenerates everything.

---

## Qué hace

Entropy Pooling (EP) actualiza un prior de escenarios con los views del portfolio manager: busca la distribución más cercana al prior entre las que cumplen los views. El EP clásico mide esa cercanía con la divergencia de Kullback–Leibler. Este repositorio la reemplaza por la entropía relativa de Tsallis de orden *q* y estudia qué controla *q*.

| Modelo | Divergencia | Forma del posterior |
|---|---|---|
| Black–Litterman | (bayesiano paramétrico) | normal |
| EP clásico (Shannon) | Kullback–Leibler, *q* → 1 | inclinación exponencial del prior; conserva todos los escenarios |
| q-EP | Tsallis de orden *q* > 1; *q* = 2 es la χ² de Pearson | potencia truncada de una función lineal; puede dejar escenarios en cero |

### Resultados del paper

1. **Cada *q* maximiza su propio número efectivo de escenarios** (con prior uniforme). El EP clásico maximiza el número basado en la entropía (Meucci, 2012); *q* = 2 maximiza el tamaño muestral efectivo de Kish y es la proyección euclídea del prior sobre los views. Por eso una búsqueda de *q* puntuada con la distancia euclídea devuelve *q* = 2: es una verificación del código, no un hallazgo.
2. **Para la mayoría de los views *q* no importa.** Todos los miembros de la familia coinciden a primer orden en el tamaño de los views, y dan exactamente el mismo posterior para views sobre la probabilidad de un evento (views de cola).
3. **Importa para views muy exigentes sobre momentos** (pruebas de stress). Ahí hay una disyuntiva: *q* = 2 reparte el peso de manera más pareja (más muestra efectiva de Kish, momentos posteriores más estables entre remuestras), pero lo logra dejando en cero los escenarios que contradicen el view, lo que achica la cola adversa. Shannon conserva todos los escenarios y concentra el peso en unos pocos.
4. **En la aplicación al Merval** (20 acciones, views basados en la opinión de un PM, fuera de muestra del 22/9/2025 al 20/3/2026) EP y q-EP dan la misma cartera. La brecha entre EP y Black–Litterman en sus configuraciones habituales es la brecha entre sus **priors** (medias históricas contra equilibrio), no un efecto de los views ni de la regla de actualización: la cartera de media-varianza del prior histórico *sin ningún view* rinde lo mismo que EP. En una ventana de seis meses ninguna de las diferencias se distingue con claridad de cero.

La matemática es clásica (divergencias de potencia de Cressie y Read, verosimilitud empírica generalizada, calibración de encuestas, tamaño muestral efectivo). El antecedente más cercano en finanzas es Dey y Juneja (2012). El aporte es aplicado: llevar la familia al problema de probabilidades de escenarios de EP con views de desigualdad, caracterizar qué optimiza cada miembro, medir cuándo importa y dejar una implementación abierta.

---

## Reproducir el paper

```bash
git clone https://github.com/rodrigocorvalan93/non_extensive_q_entropy_pooling.git
cd non_extensive_q_entropy_pooling
pip install -r requirements.txt
python reproduce_paper.py          # unos 2 minutos
cd paper && pdflatex paper_en.tex && pdflatex paper_en.tex
```

`reproduce_paper.py` corre los pasos en orden y deja la salida de cada uno en `results/`. Las tablas y los números del texto no se copian a mano: `paper/build_tables.py` los genera desde `results/` y los `.tex` los incluyen. Si faltan los datos de mercado, los pasos que los necesitan se saltean y las tablas se arman con la salida guardada en `results/`.

| Resultado del paper | Script | Salida |
|---|---|---|
| Tabla 1, Figura 1 y comprobaciones de la Sección 4.3 y del Apéndice B | `s_main_verificaciones.py` | `results/verificaciones.json`, `results/barrido_q.csv`, `paper/figs/optimal_q_zoom_*.png` |
| Tablas 2 y 3, Figura 2 (experimento de stress) y comprobación de la equivalencia local que cita el Apéndice B | `s_main_qep_vs_ep.py` | `results/stress_*.csv`, `paper/figs/qep_vs_ep_stress_*.png` |
| Tabla 5 (carteras, diseño prior × método) | `s_main_merval.py` | `results/merval_pesos_*.csv`, `results/merval_diagnosticos_*.json` |
| Tabla 6, Figura 3 (fuera de muestra) | `portfolio_evolution.py` | `results/oos_*.csv`, `paper/figs/oos_evolution_*.png` |
| Tabla 7 (robustez) | `s_main_merval_robustez.py` | `results/merval_robustez.csv` |
| Tablas y números del texto, en inglés y en español | `paper/build_tables.py` | `paper/tables/*.tex` |
| Propiedades del solver | `python -m pytest tests -q` | 34 pruebas |

La numeración de las tablas es la de `paper_en.pdf`. Los resultados se generaron con Python 3.11 y las versiones que figuran en `requirements.txt`.

---

## Estructura

```
├── entropy_pooling_v2.py         Solver de EP generalizado (Shannon / Tsallis / Rényi)
├── views_config.py               Especificación de views para BL y para EP
├── models.py                     Black-Litterman, EP y q-EP + optimización de media-varianza
├── covariance_estimation.py      Covarianza para paneles desbalanceados (por pares + Higham)
├── portfolio_evolution.py        Evaluación fuera de muestra y métricas
│
├── s_main_verificaciones.py      Comprobaciones numéricas que cita el paper
├── s_main_qep_vs_ep.py           Experimento de stress: EP contra q-EP
├── s_main_merval.py              Aplicación al Merval: diseño prior × método
├── s_main_merval_robustez.py     Robustez de la aplicación
├── s_main_bonds.py               Extensión a renta fija (Δy → retorno de precio por duration)
├── s_main_optimal_q.py           Barrido de q original (con gráficos y views aleatorios)
├── s_main_custom_entropy.py, s_main_v2_all.py, datos_chicos_test.py   Demos
├── reproduce_paper.py            Corre todo
│
├── data/dataacciones/            Insumos de la aplicación a acciones
├── data/databonos/               Insumos de la extensión a bonos
├── results/                      Salida de cada paso
├── paper/                        Paper (EN/ES), figuras, tablas generadas
├── tests/                        Pruebas del solver
├── ReturnsDistributionShort.mat  Base de Meucci reducida (1.000 escenarios)
└── octave-matlab-versions/       Código de Meucci (2008) y su extensión en Octave
                                  (incluye ReturnsDistribution.mat, 100.000 escenarios)
```

---

## Uso del solver

```python
import numpy as np
from entropy_pooling_v2 import entropy_prog

rng = np.random.default_rng(0)
X = 0.02 * rng.standard_t(df=4, size=(5000, 3))    # escenarios (J × N)
J = X.shape[0]
p = np.full(J, 1.0 / J)                            # prior uniforme

A, b = -X[:, [0]].T, np.array([-0.002])            # view: E[R_0] >= 0,2% diario
Aeq, beq = np.ones((1, J)), np.array([1.0])        # las probabilidades suman 1

p_shannon, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="S")
p_q2, _, _ = entropy_prog(p, A, b, Aeq, beq, entropy_family="T", q=2.0)

print(p_shannon @ X[:, 0], p_q2 @ X[:, 0])         # 0.002 en los dos casos
print(1 / (J * np.sum(p_q2**2)))                   # muestra efectiva de Kish (fracción de J)
```

El solver valida la solución con las condiciones KKT. Si los views son incompatibles entre sí, falla con un mensaje explícito en lugar de devolver un posterior inválido. Los casos que usa el paper son Shannon y Tsallis con *q* > 1. Con *q* < 1 solo se admiten views de igualdad, en problemas chicos.

### Views

```python
from views_config import ViewSpec

views = [
    ViewSpec.ranking(["YPFD", "PAMP", "TGSU2", "CEPU"], confidence=0.7),
    ViewSpec.relative_ineq("YPFD", "GGAL", bound=0.0, direction="geq", confidence=1.0),
    ViewSpec.absolute("YPFD", expected_return=0.0015, confidence=0.6),
    ViewSpec.volatility("YPFD", target_vol=0.03, confidence=0.4),
    ViewSpec.tail("GGAL", threshold=-0.08, max_prob=0.05, confidence=0.5),
]
```

| Tipo | Descripción | Modelos |
|---|---|---|
| `absolute`, `relative` | Retorno esperado o spread (igualdad) | BL y EP |
| `ranking` | Ordenamiento de retornos esperados | EP (desigualdades); BL (spreads ≈ 0, sin dirección) |
| `absolute_ineq`, `relative_ineq` | Desigualdades sobre retornos esperados | EP |
| `volatility`, `volatility_ineq` | Volatilidad objetivo o cota | EP |
| `tail`, `tail_upper` | Probabilidad máxima en una cola | EP |

---

## Datos y código de terceros

Ver [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) y [`data/README.md`](data/README.md).

- El código de `octave-matlab-versions/` y la base `ReturnsDistribution.mat` provienen del paquete que acompaña a Meucci (2008), distribuido bajo licencia BSD de 2 cláusulas. `EntropyProg.m`, `ViewRanking.m` y `S_MAIN.m` están modificados para la familia generalizada, y `entropy_pooling_v2.py` es un port a Python de ese código, extendido.
- Los precios de las acciones, las ponderaciones del benchmark y los rendimientos de los bonos provienen de Bloomberg y están sujetos a los términos de su licencia.

## Uso de herramientas de IA

El marco, sus implementaciones en Octave y en Python, el barrido de *q* y la aplicación al Merval son trabajo del autor, y el historial de commits los fecha en marzo de 2026. Para el paper el autor usó un asistente de IA generativa para revisar el modelo, el código y la literatura, derivar las proposiciones, implementar el experimento de stress y los controles de robustez, y redactar el borrador. La declaración, con el nombre de la herramienta, está debajo del resumen del paper.

## Historial

Este repositorio se volvió a publicar en noviembre de 2026. Los commits de marzo y de agosto conservan sus fechas, autores y mensajes originales, sin las planillas de datos ni los archivos compilados que traían; el trabajo de noviembre está consolidado en un solo commit.

## Aviso

Trabajo de investigación académica. No es asesoramiento ni recomendación de inversión. Las carteras ilustran un método, y los resultados fuera de muestra corresponden a una sola ventana.

## Cómo citar

Corvalán Salguero, R. (2026). *Non-Extensive Entropy Pooling: Integrating the Portfolio Manager's Views into Non-Gaussian Markets*. Documento de trabajo. Ver [`CITATION.cff`](CITATION.cff).

## Autor

**Rodrigo Corvalán Salguero** — Universidad Torcuato Di Tella y Universidad de Buenos Aires. El marco se desarrolló originalmente como tesis de la Maestría en Finanzas de la UTDT (2026).

## Licencia

Todavía no se definió una licencia para el código propio de este repositorio; hasta entonces, todos los derechos reservados. El material de terceros conserva su licencia original (ver `THIRD_PARTY_NOTICES.md`).
