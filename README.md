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

---

## Estructura del repositorio

```
├── entropy_pooling_v2.py         Motor de EP generalizado (Shannon/Tsallis/Rényi)
├── views_config.py               Sistema de especificación de views
├── models.py                     Modelos: BL, EP-Shannon, q-Tsallis-EP
├── portfolio_evolution.py        Backtest out-of-sample y métricas
│
├── s_main_merval.py              Aplicación empírica: Merval Panel Líder
├── s_main_optimal_q.py           Determinación del q óptimo
├── s_main_custom_entropy.py      Demo: EP con entropía seleccionable
├── s_main_v2_all.py              Demo: ranking + toy-sample
├── datos_chicos_test.py          Test: EP vs Newton-KKT
│
├── input_mkt_px.xlsx             Precios históricos (panel líder, 2017–2025)
├── input_mkt_w.xlsx              Weights del Merval al 22/sep/2025
├── input_current_mkt_px.xlsx     Precios OOS (sep 2025 – mar 2026)
├── ReturnsDistribution.mat       Base de datos de Meucci (100k escenarios)
├── ReturnsDistributionShort.mat  Versión reducida (1k escenarios)
│
└── octave-matlab-versions/       Código original en GNU Octave/MATLAB
```

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

### 3. Backtest out-of-sample

```bash
python portfolio_evolution.py
```

Evalúa la performance de cada cartera en el período sep 2025 – mar 2026. Calcula Sharpe, Sortino, Max Drawdown, Tracking Error, Alfa y Beta.

### 4. Definir views personalizados

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
| `tail` | Probabilidad máxima en la cola | EP solamente |

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
