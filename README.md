"""
DOCUMENTACIÓN DEL CÓDIGO — Estructura y descripción de módulos
==============================================================

Estructura del repositorio:

    ├── entropy_pooling_v2.py        Motor de cálculo (EP generalizado)
    ├── views_config.py              Sistema de especificación de views
    ├── models.py                    Modelos de asignación (BL, EP, q-EP)
    ├── s_main_merval.py             Aplicación empírica: Merval
    ├── s_main_optimal_q.py          Determinación del q óptimo
    ├── s_main_custom_entropy.py     Demo: EP con familia de entropía elegible
    ├── s_main_v2_all.py             Demo: ranking + toy-sample
    ├── datos_chicos_test.py         Test: EP vs Newton-KKT (validación)
    ├── ReturnsDistribution.mat      Base de datos de Meucci (100k escenarios)
    ├── ReturnsDistributionShort.mat  Versión reducida (1k escenarios)
    ├── input_mkt_px.xlsx            Precios panel líder Merval
    └── input_mkt_w.xlsx             Weights del Merval al 22/sep/2025
"""

# =================================================================
# TEXTO PARA LA TESIS (descripción de cada módulo)
# =================================================================

DESCRIPCION_MODULOS = """

1. entropy_pooling_v2.py — Motor de Entropy Pooling generalizado

Este módulo constituye el núcleo computacional del framework. Implementa
la resolución numérica del problema de Entropy Pooling para cuatro
familias de entropía: Shannon (S), Tsallis (T), Rényi (R) y General (G).
La función principal, entropy_prog(), recibe como inputs el vector de
probabilidades del prior p ∈ R^J, las matrices de constraints de
desigualdad (A, b) e igualdad (Aeq, beq), la familia de entropía y el
parámetro q, y devuelve el posterior p̃ ∈ R^J que minimiza la
q-divergencia relativa sujeto a las restricciones, junto con la función
Lagrangiana evaluada en el óptimo y el vector de multiplicadores. La
resolución se realiza en el espacio dual mediante L-BFGS-B (para
q ≥ 1) o Newton-Raphson multivariado (para q < 1). Para las familias
Rényi y General, donde el posterior no admite despeje explícito, se
emplea un método de punto fijo con iteración de Mann y arranque desde
la solución de Tsallis. El módulo incluye además las funciones
efficient_frontier() para el cálculo de la frontera eficiente de
media-varianza con probabilidades ponderadas, implied_exp_rets() para
la optimización inversa de retornos implícitos, y funciones de
visualización (plot_frontier(), plot_results()). Se proveen aliases
con nombres en estilo MATLAB (EntropyProg, EfficientFrontier, etc.)
para facilitar la correspondencia con el código original de Meucci (2008).


2. views_config.py — Sistema de especificación de views

Este módulo provee una interfaz para la definición de views del
inversor, traduciendo especificaciones intuitivas a las estructuras
matemáticas requeridas por cada modelo. Se soportan cinco tipos de
views: absoluto (retorno esperado de un activo), relativo (spread entre
dos activos), ranking (ordenamiento de retornos esperados), volatilidad
(desvío estándar target) y cola (probabilidad máxima debajo de un
umbral). Cada view se crea mediante constructores estáticos de la clase
ViewSpec, especificando los tickers involucrados, los parámetros del
view y el nivel de confianza subjetiva (entre 0 y 1). La función
build_views() toma una lista de ViewSpec y genera simultáneamente las
estructuras para Black-Litterman (matrices P, Q, Ω según Idzorek
(2007)) y para Entropy Pooling (matrices A, b, Aeq, beq). Los views de
tipo absoluto, relativo y ranking son compatibles con ambos modelos;
los de volatilidad y cola requieren la distribución completa de
escenarios y solo son compatibles con EP y q-Tsallis-EP. La confianza
se incorpora en BL mediante la calibración de la diagonal de Ω
(fórmula de Idzorek: ω_k = ((1−c)/c) · τ · P_k Σ P_k^T), y en EP
mediante la mezcla ponderada del target con la expectativa del prior.


3. models.py — Modelos de asignación de cartera

Este módulo encapsula los tres modelos comparados en el trabajo. La
función run_black_litterman() implementa el modelo de Black-Litterman
(1992) según la formulación de He & Litterman (2002): calcula los
retornos implícitos de equilibrio Π = δΣw_mkt mediante optimización
inversa, obtiene la distribución posterior
μ_BL = (τΣ⁻¹ + P^TΩ⁻¹P)⁻¹(τΣ⁻¹Π + P^TΩ⁻¹Q), y determina la
cartera óptima mediante optimización de media-varianza con el vector
μ_BL resultante. La función run_entropy_pooling() implementa el EP
clásico de Meucci (2008) minimizando la divergencia de Kullback-Leibler
(Shannon) para obtener el posterior p̃, del cual se extraen los momentos
y se optimiza la cartera. La función run_q_tsallis_ep() extiende el EP
utilizando la q-divergencia de Tsallis con q parametrizable (por
defecto q = 2, el valor óptimo demostrado en este trabajo). Los tres
modelos reciben como inputs los retornos históricos, la covarianza, los
weights del benchmark y los views traducidos, y devuelven un objeto
ModelResult que contiene los pesos óptimos, los retornos esperados
posteriores, la covarianza posterior, la frontera eficiente y medidas
de riesgo (retorno anualizado, volatilidad, Sharpe ratio, VaR y CVaR
al 95% y 99%). Todos los modelos admiten un parámetro max_weight que
impone un límite máximo de concentración por activo.


4. s_main_merval.py — Aplicación empírica con el Merval

Script principal que ejecuta la comparación empírica de los tres modelos
sobre el panel líder del índice Merval. Carga los precios históricos
(input_mkt_px.xlsx) y los weights de mercado (input_mkt_w.xlsx), calcula
retornos (logarítmicos, simples o delta para yields de bonos), define
los views del portfolio manager (ranking de energéticas y underweight en
bancos), y ejecuta secuencialmente Black-Litterman, EP-Shannon y
q-Tsallis-EP. Los resultados se presentan en tablas comparativas
impresas en consola, gráficos PNG de cuatro paneles (pesos, fronteras,
retornos, métricas de riesgo), y archivos Excel con tres hojas (pesos
óptimos, retornos esperados, medidas de riesgo) listos para incluir en
el documento.


5. s_main_optimal_q.py — Determinación del q óptimo

Script que implementa la metodología para determinar el valor óptimo del
parámetro q de la entropía de Tsallis. Incluye la función
views_generator() (traducción de views_generator.m) que genera views
con tres métodos (original Meucci, sigma-escalado y randomizado), la
función sweep_q() que barre un rango de valores de q calculando el
error relativo ‖p̃−p‖/‖p‖ para cada uno, y la función
sweep_q_random_seeds() que repite el barrido con múltiples semillas
para verificar robustez. El script genera tablas de resultados, la
validación cruzada contra los valores de referencia de GNU Octave, y
tres gráficos (barrido completo, zoom del óptimo, robustez con
semillas). Los resultados demuestran que q = 2 (entropía de colisión)
minimiza consistentemente la distancia del posterior al prior.


6. s_main_custom_entropy.py — Demostración de EP con entropía elegible

Script de demostración que permite seleccionar la familia de entropía
(Shannon, Tsallis, Rényi) y el parámetro q desde el encabezado del
archivo, utilizando la base de datos de Meucci (ReturnsDistribution.mat).
Calcula la frontera eficiente del prior, aplica un view de ranking, y
genera la frontera posterior con la familia de entropía elegida.


7. s_main_v2_all.py — Demostración combinada (ranking + toy-sample)

Script que combina dos bloques: (1) el ejemplo de ranking con la base de
datos de Meucci, equivalente a S_MAIN.m, y (2) la validación con datos
reducidos (toy-sample), equivalente a DatosChicos.m. Permite activar o
desactivar cada bloque independientemente.


8. datos_chicos_test.py — Validación numérica (EP vs Newton-KKT)

Script de validación que compara los resultados de entropy_prog() contra
una solución obtenida resolviendo las condiciones KKT del Lagrangiano
mediante Newton-Raphson multivariado con Jacobiano aproximado por
diferencias finitas. Se utiliza un problema de dimensión reducida
(J = 4 escenarios) con constraints de igualdad y desigualdad, y se
verifica la coincidencia para las tres familias: Shannon, Tsallis
(q = 1.5) y Rényi (q = 1.5).
"""

# =================================================================
# TABLA DE FUNCIONES PRINCIPALES
# =================================================================

TABLA_FUNCIONES = """

┌─────────────────────────────────────────────────────────────────────┐
│ MÓDULO: entropy_pooling_v2.py                                       │
├─────────────────────┬───────────────────────┬───────────────────────┤
│ Función             │ Inputs                │ Outputs               │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ entropy_prog()      │ p, A, b, Aeq, beq,    │ p̃ (posterior),        │
│                     │ entropy_family, q,     │ Lx (Lagrangiana),     │
│                     │ g, dg                  │ lv (multiplicadores)  │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ efficient_frontier()│ X, p, options          │ e, s, w, Exps, Covs   │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ implied_exp_rets()  │ S (cov), w (pesos)     │ M (retornos implíc.)  │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ plot_results()      │ e, s, w, M, Lower,     │ figura matplotlib     │
│                     │ Upper                  │                       │
└─────────────────────┴───────────────────────┴───────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ MÓDULO: views_config.py                                             │
├─────────────────────┬───────────────────────┬───────────────────────┤
│ ViewSpec.absolute() │ ticker, expected_ret,  │ ViewSpec              │
│                     │ confidence             │                       │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ ViewSpec.relative() │ ticker_long,           │ ViewSpec              │
│                     │ ticker_short, spread,  │                       │
│                     │ confidence             │                       │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ ViewSpec.ranking()  │ tickers_ordered,       │ ViewSpec              │
│                     │ confidence             │                       │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ ViewSpec.volatility()│ ticker, target_vol,   │ ViewSpec              │
│                     │ confidence             │                       │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ ViewSpec.tail()     │ ticker, threshold,     │ ViewSpec              │
│                     │ max_prob, confidence   │                       │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ build_views()       │ views, tickers, Σ,     │ BLViews, EPViews      │
│                     │ X, p, τ                │                       │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ build_bl_views()    │ views, tickers, Σ, τ   │ BLViews (P, Q, Ω)     │
├─────────────────────┼───────────────────────┼───────────────────────┤
│ build_ep_views()    │ views, tickers, X, p   │ EPViews (A, b,        │
│                     │                        │  Aeq, beq)            │
└─────────────────────┴───────────────────────┴───────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ MÓDULO: models.py                                                   │
├──────────────────────────┬──────────────────┬───────────────────────┤
│ run_black_litterman()    │ Σ, w_mkt,        │ ModelResult           │
│                          │ bl_views,        │ (w_optimal,           │
│                          │ tickers, δ, τ,   │  mu_posterior,        │
│                          │ X, p, max_weight │  Sigma_posterior,     │
│                          │                  │  frontier, risk)      │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ run_entropy_pooling()    │ X, p, ep_views,  │ ModelResult           │
│                          │ tickers, w_mkt,  │                       │
│                          │ δ, confidence,   │                       │
│                          │ max_weight       │                       │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ run_q_tsallis_ep()       │ X, p, ep_views,  │ ModelResult           │
│                          │ tickers, w_mkt,  │                       │
│                          │ δ, q, confidence,│                       │
│                          │ max_weight       │                       │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ print_model_comparison() │ results, tickers,│ tabla en consola      │
│                          │ w_mkt            │                       │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ plot_model_comparison()  │ results, tickers,│ figura PNG            │
│                          │ w_mkt, save_path │                       │
└──────────────────────────┴──────────────────┴───────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ MÓDULO: s_main_optimal_q.py                                         │
├──────────────────────────┬──────────────────┬───────────────────────┤
│ views_generator()        │ X, Lower, Upper, │ V (vector de views)   │
│                          │ ViewMethod, seed │                       │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ sweep_q()                │ X, p, Lower,     │ q_values, errors,     │
│                          │ Upper, ViewMethod│ err_clasico, q_opt    │
│                          │ q_values,        │                       │
│                          │ confidence       │                       │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ sweep_q_random_seeds()   │ X, p, Lower,     │ dict con all_errors,  │
│                          │ Upper, seeds,    │ all_shannon,          │
│                          │ q_values         │ all_q_opt por seed    │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ plot_optimal_q()         │ q_values, errors,│ figura PNG            │
│                          │ err_clasico,     │                       │
│                          │ q_opt            │                       │
├──────────────────────────┼──────────────────┼───────────────────────┤
│ plot_random_seeds()      │ results (dict)   │ figura PNG            │
└──────────────────────────┴──────────────────┴───────────────────────┘
"""
