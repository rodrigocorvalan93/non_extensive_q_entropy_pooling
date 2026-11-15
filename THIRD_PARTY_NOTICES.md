# Código y datos de terceros / Third-party code and data

## Attilio Meucci, "Fully Flexible Views: Theory and Practice" (código MATLAB y datos)

Fuente: MATLAB Central File Exchange, archivo 21307, "Fully Flexible Views and Stress-testing"
(https://www.mathworks.com/matlabcentral/fileexchange/21307-fully-flexible-views-and-stress-testing);
también www.symmys.com. Referencia: A. Meucci (2008), "Fully Flexible Views: Theory and Practice",
*Risk* 21(10), 97-102.

El paquete pide citar al autor y la fuente: "Attilio Meucci - www.symmys.com".

Archivos sin modificar
- `octave-matlab-versions/ImpliedExpRets.m`
- `octave-matlab-versions/PlotFrontier.m`
- `octave-matlab-versions/RobustEfficientFrontier.m`
- `octave-matlab-versions/ReturnsDistribution.mat`

Archivos modificados por R. Corvalán Salguero (2026)
- `octave-matlab-versions/EfficientFrontier.m`, `PlotResults.m`: solo sintaxis de Octave.
- `octave-matlab-versions/ViewRanking.m`, `S_MAIN.m`: parámetros para el generador de views y la familia de entropía.
- `octave-matlab-versions/EntropyProg.m`: extendido del caso de Shannon a las familias de Tsallis, de Rényi y general.

Archivos derivados
- `ReturnsDistributionShort.mat`: los primeros 1.000 escenarios de `ReturnsDistribution.mat`.
- `entropy_pooling_v2.py`: port a Python de `EntropyProg.m`, `ViewRanking.m`, `EfficientFrontier.m` y
  funciones auxiliares, con la extensión a la familia generalizada y un solver dual reescrito.

Licencia del paquete original (BSD de 2 cláusulas):

```
Copyright (c) 2008, Attilio Meucci
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

Nota: el texto de la licencia se tomó de una copia pública del paquete. Conviene cotejarlo con el
`license.txt` de la descarga oficial de File Exchange.

## Datos de mercado

Los precios de las acciones, las ponderaciones del índice y los rendimientos, duration y convexidad
de los bonos provienen de Bloomberg y están sujetos a los términos de su licencia. Ver `data/README.md`.
