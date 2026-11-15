# Datos

## `dataacciones/` (aplicación a acciones, Secciones 4.4 y 6 del paper)

| Archivo | Contenido |
|---|---|
| `input_mkt_px.xlsx` | Precios de cierre diarios, ajustados por eventos corporativos, de las 20 acciones del panel líder: ALUA, BBAR, BMA, BYMA, CEPU, COME, CRES, EDN, GGAL, LOMA, METR, PAMP, SUPV, TECO2, TGNO4, TGSU2, TRAN, TXAR, VALO, YPFD. Del 23/11/2017 al 22/9/2025, en pesos. |
| `input_mkt_w.xlsx` | Ponderaciones del benchmark al 22/9/2025 (columnas `ticker`, `weight`): cantidad de cada acción en la cartera del índice Merval por su precio de cierre, normalizadas para sumar 1. |
| `input_current_mkt_px.xlsx` | Los mismos precios extendidos hasta el 20/3/2026 (período fuera de muestra). |

## `databonos/` (extensión a renta fija, Sección 7)

| Archivo | Contenido |
|---|---|
| `input_mkt_px.xlsx` | Rendimientos diarios (en porcentaje) de GD29, GD30, GD35, GD38, GD41, GD46 y de los Treasuries a 2 y 10 años, del 30/9/2021 al 26/3/2026. |
| `input_bond_stats.xlsx` | Duration y convexidad diarias de los mismos instrumentos (hojas `duration` y `convexity`), hasta el 10/4/2026. |
| `input_mkt_w.xlsx` | Ponderaciones de la cartera de referencia (los seis Globales por partes iguales y 2,5% en cada Treasury). |
| `input_current_mkt_px.xlsx` | Rendimientos hasta el 10/4/2026. |

Los dos archivos de rendimientos provienen de descargas distintas y no coinciden en la
convención de rendimiento de los Globales. `s_main_bonds.py` usa `input_mkt_px.xlsx`. El paper
describe la extensión a renta fija como método y no informa resultados; antes de informarlos hay
que unificar las dos series.

## Fuente y condiciones de uso

Los datos provienen de Bloomberg y están sujetos a los términos de su licencia. Para reconstruir
los archivos con otra fuente alcanza con respetar el formato: primera columna con la fecha y una
columna por instrumento, con el ticker como encabezado.

Los archivos de `results/` contienen la salida de cada paso (ponderaciones de las carteras,
métricas, valores diarios de las carteras), de modo que las tablas del paper se pueden regenerar
con `python paper/build_tables.py` sin volver a correr los pasos que usan datos de mercado. Si las
planillas de `dataacciones/` no están, `python reproduce_paper.py` saltea esos pasos y usa la
salida guardada.
