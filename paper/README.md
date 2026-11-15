# Paper

- `paper_en.tex` / `paper_en.pdf`: versión en inglés (la que se sube a SSRN).
- `paper_es.tex` / `paper_es.pdf`: versión en español, con el mismo contenido.
- `figs/`: figuras, generadas por los scripts de la raíz.
- `tables/`: tablas y números del texto, generados por `build_tables.py` desde `../results/`.
  No se editan a mano: si cambia un resultado, se vuelve a correr el script.

## Compilar

```bash
python ../reproduce_paper.py      # opcional: regenera results/, figs/ y tables/
pdflatex paper_en.tex && pdflatex paper_en.tex
pdflatex paper_es.tex && pdflatex paper_es.tex
```

Hace falta una instalación de LaTeX con `natbib`, `booktabs`, `microtype`, `lmodern` y `babel`
(español). En Overleaf hay que subir también las carpetas `figs/` y `tables/`.

## Antes de subir a SSRN

1. SSRN acepta solo trabajos en inglés: se sube `paper_en.pdf`. La versión en español queda en este
   repositorio.
2. El PDF tiene que mostrar el título, el autor y sus afiliaciones. SSRN pide que la declaración
   de uso de IA vaya con el resumen y se vea en el PDF: está debajo del resumen, y esa misma
   oración se pega al final del campo *Abstract* del formulario. Conviene leer antes la política
   vigente (<https://www.elsevier.support/ssrn/answer/AI>) y las guías de envío
   (<https://www.elsevier.support/ssrn/answer/get-started>).
3. Revisar que cada frase de la sección *Declarations* sea cierta al momento de subir: uso de IA,
   financiamiento, conflictos de interés y disponibilidad de datos.
4. Palabras clave: cargarlas una por una. Los códigos JEL son opcionales.
5. Elegir con cuidado la clasificación (redes y eJournals): después no se puede cambiar con el
   formulario de revisión.
6. Las revisiones se suben con *Revise* en *My Papers*; el DOI no cambia.
