# Non-Extensive q-Entropy Pooling

**Asignación Cuantitativa de Portafolios basada en entropía no extensiva: Integración del *view* del Portfolio Manager en mercados no gaussianos**

*Non-extensive Entropy-Based Quantitative Portfolio Allocation: Integrating the Portfolio Manager's View into Non-Gaussian Markets*

---

## Resumen

Este repositorio contiene el código desarrollado para la tesis de maestría en Finanzas (Universidad Torcuato Di Tella, 2026), que extiende el framework de *Entropy Pooling* de Meucci (2008) incorporando las entropías generalizadas de Tsallis (1988) y Rényi (1961).

El aporte principal es la demostración de que el parámetro **q = 2** (entropía de colisión) minimiza consistentemente la distancia del posterior al prior, y la implementación de un framework modular en Python que compara tres modelos de asignación:

| Modelo | Entropía | Supuestos |
|---|---|---|
| Black-Litterman | — (bayesiano paramétrico) | Normalidad, 2 primeros momentos |
| Entropy Pooling (Shannon) | KL-divergencia (q → 1) | Distribución completa, no paramétrico |
| **q-Tsallis-EP** | **q-divergencia de Tsallis (q = 2)** | **Distribución completa, colas pesadas** |

### Resultados principales

- **q óptimo = 2.0** demostrado con la base de datos de Meucci (10⁵ escenarios) y verificado con múltiples semillas random.
- **Backtest out-of-sample** (Merval, sep 2025 – mar 2026): EP y q-Tsallis-EP logran Sharpe 3.24 vs 2.06 del benchmark, con alfa anualizado de +48%.
- Validación cruzada Python vs GNU Octave con discrepancias < 10⁻⁵.

### Extensión a renta fija (bonos)

El framework se extiende a carteras reales de bonos (`s_main_bonds.py`): el factor invariante es el **cambio de yield (Δy)** y el puente hacia retornos de precio es el mapeo por duration/convexity:

```
ΔP/P ≈ -D_mod · Δy + ½ · C · (Δy)²
```

con tres modos de duration (`constant`, `rolling`, `rolling_convex`). Los views del Portfolio Manager se expresan sobre Δy (recordar: Δy negativo = suba de precio). Para paneles desbalanceados (papeles con poca historia) la covarianza se estima de forma robusta con `covariance_estimation.py` (estimación pairwise + corrección PSD de Higham).

---

## Estructura del repositorio

```
├── entropy_pooling_v2.py         Motor de EP generalizado (Shannon/Tsallis/Rényi)
├── views_config.py               Sistema de especificación de views (incluye views de desigualdad)
├── models.py                     Modelos: BL, EP-Shannon, q-Tsallis-EP (+ restricción vs benchmark)
├── covariance_estimation.py      Covarianza robusta p/ paneles desbalanceados (pairwise + Higham)
├── portfolio_evolution.py        Backtest out-of-sample y métricas (+ alfa/beta OLS)
│
├── s_main_merval.py              Aplicación empírica: Merval Panel Líder
├── s_main_bonds.py               Aplicación empírica: bonos (yields → duration mapping)
├── s_main_optimal_q.py           Determinación del q óptimo
├── s_main_custom_entropy.py      Demo: EP con entropía seleccionable
├── s_main_v2_all.py              Demo: ranking + toy-sample
├── datos_chicos_test.py          Test: EP vs Newton-KKT
│
├── input_mkt_px.xlsx             Serie histórica activa (precios o yields, según dataset)
├── input_mkt_w.xlsx              Weights del portfolio benchmark
├── input_current_mkt_px.xlsx     Precios del período out-of-sample
├── input_bond_stats.xlsx         Duration y convexity de los bonos (hojas "duration"/"convexity")
├── bbg_input_data_merval.xlsx    Datos fuente Bloomberg (acciones)
├── bbg_input_data_globales_yields.xlsx  Datos fuente Bloomberg (yields de globales)
├── views_bonds_pm.docx           Views del PM para la cartera de bonos
│
├── data/
│   ├── dataacciones/             Juego de inputs para acciones (Merval)
│   └── databonos/                Juego de inputs para bonos
│
├── ReturnsDistributionShort.mat  Base de Meucci reducida (1k escenarios)
│
└── octave-matlab-versions/       Código original en GNU Octave/MATLAB
                                  (incluye ReturnsDistribution.mat, 100k escenarios)
```

> **Datasets intercambiables:** los scripts leen los `input_*.xlsx` de la raíz. Las carpetas `data/dataacciones/` y `data/databonos/` guardan los dos juegos de inputs; copiá el juego que corresponda a la raíz según lo que quieras correr. **El estado actual de la raíz tiene activo el dataset de bonos** (`input_mkt_px.xlsx` contiene yields).

> **Nota:** `ReturnsDistribution.mat` (100k escenarios) se quitó de la raíz para aligerar el repo; sigue disponible en `octave-matlab-versions/`. `s_main_optimal_q.py` usa automáticamente la versión Short si no lo encuentra; para el barrido completo, copialo de vuelta a la raíz.

---

## Instalación

```bash
git clone https://github.com/rodrigocorvalan93/non_extensive_q_entropy_pooling.git
cd non_extensive_q_entropy_pooling
pip install numpy scipy pandas matplotlib openpyxl
```

**Requisitos:** Python ≥ 3.9, NumPy, SciPy, Pandas, Matplotlib, OpenPyXL.

---

## Uso rápido

### 1. Determinación del q óptimo

```bash
python s_main_optimal_q.py
```

Barre q de 1.1 a 3.0, genera gráficos y la validación cruzada vs Octave.

### 2. Aplicación empírica (Merval)

```bash
python s_main_merval.py
```

Compara BL vs EP-Shannon vs q-Tsallis-EP con los 20 activos del panel líder. Genera gráficos comparativos y exporta resultados a Excel.

> Requiere el dataset de acciones en la raíz (`data/dataacciones/` → raíz).

### 3. Portfolio óptimo de bonos

```bash
python s_main_bonds.py
```

Aplica el pipeline completo a una cartera real de bonos usando yields como factor invariante (`input_mkt_px.xlsx` con yields en decimal), el mapeo Δy → ΔP/P vía modified duration/convexity (`input_bond_stats.xlsx`) y views del PM expresados sobre Δy. Configurable con `DURATION_MODE` (`constant` / `rolling` / `rolling_convex`). Genera `model_comparison_bonds.png` y `resultados_bonds.xlsx`.

> Requiere el dataset de bonos en la raíz (`data/databonos/` → raíz), que es el estado actual del repo.

### 4. Backtest out-of-sample

```bash
python portfolio_evolution.py
```

Evalúa la performance de cada cartera en el período sep 2025 – mar 2026. Calcula Sharpe, Sortino, Max Drawdown, Tracking Error, Alfa y Beta (regresión OLS contra el benchmark).

### 5. Definir views personalizados

```python
from views_config import ViewSpec

views = [
    ViewSpec.ranking(["YPFD", "PAMP", "TGSU2", "CEPU"], confidence=0.7),
    ViewSpec.absolute("YPFD", expected_return=0.0015, confidence=0.6),
    ViewSpec.relative("YPFD", "GGAL", spread=0.0005, confidence=0.5),
    ViewSpec.volatility("YPFD", target_vol=0.03, confidence=0.4),
    ViewSpec.tail("GGAL", threshold=-0.08, max_prob=0.05, confidence=0.5),
]
```

---

## Tipos de views soportados

| Tipo | Descripción | Compatible con |
|---|---|---|
| `absolute` | Retorno esperado de un activo | BL + EP |
| `relative` | Spread entre dos activos | BL + EP |
| `ranking` | Ordenamiento de retornos | BL + EP |
| `volatility` | Volatilidad target | EP solamente |
| `tail` | Probabilidad máxima en la cola inferior | EP solamente |
| `absolute_ineq` | Desigualdad absoluta: E[R_i] ≥ o ≤ cota | EP solamente |
| `relative_ineq` | Desigualdad relativa: E[R_long − R_short] ≥ o ≤ cota | EP solamente |
| `volatility_ineq` | Desigualdad de volatilidad: σ(X_i) ≤ o ≥ cota | EP solamente |
| `tail_upper` | Probabilidad máxima en la cola superior | EP solamente |

Además, el optimizador de `models.py` acepta una **restricción de desvío activo vs benchmark**: `|w_i − w_bench_i| ≤ max_active_deviation` (parámetros `w_benchmark` y `max_active_deviation`).

---

## Referencias

- Black, F. & Litterman, R. (1992). *Global portfolio optimization*. Financial Analysts Journal.
- He, G. & Litterman, R. (2002). *The intuition behind Black-Litterman model portfolios*. Goldman Sachs.
- Idzorek, T. (2007). *A step-by-step guide to the Black-Litterman model*. Ibbotson Associates.
- Meucci, A. (2008). *Fully Flexible Views: Theory and Practice*. Risk Magazine.
- Rényi, A. (1961). *On measures of entropy and information*. Berkeley Symposium.
- Tsallis, C. (1988). *Possible generalization of Boltzmann-Gibbs statistics*. J. Stat. Physics.
- Tsallis, C. (2009). *Introduction to Nonextensive Statistical Mechanics*. Springer.

---

## Autor

**Rodrigo Corvalán Salguero**
Maestría en Finanzas — Universidad Torcuato Di Tella (2026)

---

## Licencia

Este código fue desarrollado con fines académicos. El uso del dataset de Meucci está sujeto a los términos de [MATLAB Central File Exchange](https://la.mathworks.com/matlabcentral/fileexchange/21307-fully-flexible-views-and-stress-testing).
