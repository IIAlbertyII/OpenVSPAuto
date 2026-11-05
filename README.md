# Reconstrucción paramétrica de aeronaves en OpenVSP

Un solo programa (`aircraft_recon.py`) que construye y comprueba aviones en OpenVSP 3.53.1, a partir de tres vistas dibujadas o de archivos JSON. Dependencias: `numpy`, `opencv-python` y `scipy`, con OpenVSP y su binding de Python.

- **Guía de uso, paso a paso:** [Guia_aircraft_recon.pdf](Guia_aircraft_recon.pdf)
- **Caso de validación:** Airbus A300-600 (lámina "2.2 General Airplane Dimensions", redibujada a mano en tres vistas)

> La IA interpreta; el software determinista mide y ejecuta.

## 1. Instalación

Necesitas Windows, Git y Python 3.13: el binding de OpenVSP está compilado para esa versión.

1. Descarga el programa:
   ```
   git clone https://github.com/IIAlbertyII/OpenVSPAuto.git
   cd OpenVSPAuto
   ```
2. Descarga OpenVSP para Windows desde [openvsp.org/download.php](https://openvsp.org/download.php) y descomprímelo dentro de `OpenVSPAuto`, en una carpeta cuyo nombre empiece por `OpenVSP`. El programa se ha probado con la 3.53.1. Si lo dejas en otro sitio, indica la carpeta en la variable `VSP_PYTHON`.
3. Crea el entorno e instala las dependencias:
   ```
   python -m venv C:\venvs\vsp313
   C:\venvs\vsp313\Scripts\activate
   pip install numpy opencv-python scipy
   ```
4. Comprueba que todo funciona:
   ```
   python aircraft_recon.py selftest      # debe terminar en SELFTEST OK
   python aircraft_recon.py check-env     # debe decir OpenVSP OK y la versión
   ```

Para archivos temporales, usa rutas sin acentos ni "ñ". El repositorio contiene el programa, este README y la guía; OpenVSP, los modelos y la presentación no se suben. Los comandos y las rutas de este documento parten de la carpeta del programa.

## 2. Dónde está cada cosa

Estructura de la carpeta de trabajo del autor; `aircraft/` y `project/` se crean al usar el programa.

```
aircraft_recon.py        el programa
README.md                este documento
Guia_aircraft_recon.pdf  guía de uso
presentacion/            presentación del A300, capturas y datos de análisis
OpenVSP-3.53.1-win64/    OpenVSP
project/                 proyecto A300 original (versión actual v088); carpeta antigua, no se mueve
aircraft/
  current.txt            proyecto seleccionado
  a300_test/             A300 de prueba
  a340/                  A340-300; sus archivos de partida están en fuentes/
  a340_ala_a350/         A340 con ala de A350; datos y comparación en fuentes/
  far25/                 modelos y exportaciones FAR25
  Numbus/                modelos Nimbus 200, 226 y 252
    iteraciones_mom226/  ala Middle-of-the-Market para el Nimbus226 (tabla de abajo)
version control/         versiones del programa, copias, README antiguos y registro de movimientos
descargas/               fotos y vídeos, sin cambios
```

`version control/movimientos_organizacion.json` guarda de dónde viene cada archivo movido, con su suma de comprobación.

### Iteraciones del ala del Nimbus226

Todas parten de `aircraft/Numbus/Nimbus226.vsp3` y no modifican el fuselaje, el tren ni los motores. Cada carpeta contiene el `.vsp3`, sus JSON y sus comparaciones.

| Carpeta | Área alar | Cambio principal |
|---|---|---|
| `v1_ala_inicial` | 215 m² | Primera ala para 226 pasajeros |
| `v2_cola_y_mandos` | 215 m² | Estabilizador de 63 m² con flecha inversa; mandos alejados del tren |
| `v3_perfiles_y_calado` | 215 m² | Perfiles NASA SC(2), twist y calado para crucero |
| `v4_ala_235m2` | 235 m² | Más área y flecha, menos espesor interior; flap hasta y = 2.4 m |
| `v5_winglet_suave` | 235 m² | **La más reciente.** Winglet con transición en arco |

`perfiles_sc2/` guarda las coordenadas descargadas de UIUC. `scripts/` guarda los programas de diseño y verificación: documentan cómo se obtuvo cada versión, pero usan las rutas de cuando se crearon.

**Pendiente en v4 y v5:** el tren y los motores se ajustarán después. El muñón del tren no toca el ala y queda dentro de la zona del flap interior. Los resultados aerodinámicos (VSPAERO sin fuselaje, Mdd con la regla de Korn) son estimaciones conceptuales.

## 3. Proyectos

Cada avión vive en `aircraft/NOMBRE/`, con sus parámetros, máscaras, versiones, `fuselage.json`, `wing.json` y `fairings.json`.

| Comando | Para qué |
|---|---|
| `project-new NOMBRE [--params F] [--fuselage F] [--wing F]` | Crea y selecciona un proyecto |
| `project-list`, `project-use NOMBRE` | Lista o selecciona |
| `project-copy ORIGEN DESTINO [--full] [--use]` | Copia, sin versiones salvo con `--full` |
| `project-adopt NOMBRE` | Mueve la carpeta antigua `project/` a `aircraft/NOMBRE` |
| `--project NOMBRE comando` | Usa otro proyecto solo para ese comando; va antes del comando |

Sin proyecto seleccionado se usa `project/`. También existe la variable `AIRCRAFT_PROJECT`. Cada comando imprime `[project: ...]` al empezar.

Estructura de un proyecto:

```
params.json     parámetros con evidencia (measured / calculated / inferred / unknown), confianza y fixed
views.json      calibración de cada vista (px/m, bbox, longitud)
fuselage.json   estaciones del fuselaje
wing.json       ala multi-sección
fairings.json   carenados de las guías de flaps
state.json      versión actual y escala de teselación
processed/      máscaras: *_mask.png (activa), *_mask_full.png, *_mask_core.png (sin motores), *_mask_extras.png (con motores)
versions/ iterations/ renders/
```

## 4. Flujo de trabajo

### Desde tres vistas

```
project-new MI_AVION
calibrate --view top   --image planta.png  --length 53.85 --keep-parts
calibrate --view side  --image perfil.png  --length 54.08 --keep-parts
calibrate --view front --image frontal.png --length 44.84 --keep-parts
erase-mask --view top        # quita motores y tren; repetir por vista
restore-mask --view top      # vuelve a poner solo los motores; crea core y extras
set fuselage.max_width 5.64 --force
set fuselage.max_height 5.64 --force
auto --step --masks
compare
report-metrics
```

`wizard` guía estos pasos de forma interactiva. Las longitudes son las del A300; marca como `measured` y `"fixed": true` las cotas fiables de `params.json`, y el optimizador no las tocará. Si la escala está mal, `rescale --view V --length L` no rehace la máscara.

`auto` tiene tres fases:

| Fase | Qué hace | Máscaras |
|---|---|---|
| `airframe` | Fuselaje por estaciones, ala, estabilizadores, diedros y grosores | core |
| `extras` | Góndolas, pilones, carenados y refinado | extras |
| `finish` | Tren, superficies de mando, ventanas, malla fina | — |

Opciones: `--phase`, `--passes`, `--iters`, `--step` (para tras cada paso), `--masks`, `--refit`, `--no-finish`, `--no-guard` (quita la reversión automática si una etapa empeora más de un 2 %), `--pchip-simplify`.

### Desde archivos JSON, sin imágenes

```
project-new a340b --params aircraft\a340\fuentes\a340_params.json --fuselage aircraft\a340\fuentes\a340_fuselage.json --wing aircraft\a340\fuentes\a340_wing.json
build
wing-check
```

### Revisar y deshacer

`compare` escribe overlays en `renders/` (rojo: solo dibujo; azul: solo modelo). `report-metrics` da IoU, chamfer, Hausdorff-95 y errores de área y extensión. `history --last N`, `diff A B`, `undo` y `checkout N` manejan las versiones. Cada cambio guarda `versions/vNNN.vsp3` y `.json`.

Cambios manuales: `set`, `delta` y `patch ARCHIVO.json`. Formato de parche:

```json
[{"operation":"modify","parameter":"wing.x","delta":-0.3,"reason":"ala 300 mm demasiado adelantada"}]
```

Los parámetros `measured` o `fixed` se rechazan salvo con `--force`.

## 5. Ala multi-sección (`wing.json`)

Metros y grados; `y` es la distancia al eje en una semiala.

| Campo | Significado |
|---|---|
| `stations[]` | `y`, `chord`, `x_le` (absoluto), `z` (opcional), `thickness` (t/c) o `t_rel`, `twist`, `winglet: true` (no cuenta para la envergadura en planta) |
| `root_perpendicular` | `true` (por defecto): del eje al costado del fuselaje mantiene cuerda y borde de ataque |
| `z_from_params` | `true`: `z = wing.z + (y − y_encastre)·tan(wing.dihedral)` |
| `twist_mode` | `absolute` (por defecto) o `incremental`. OpenVSP no encadena los twists de sección |
| `u_mode` | `span` (por defecto) o `section`: conversión de posición en envergadura a U del mando. En los modelos de este proyecto se usa `section` |
| `x_offset`, `z_offset` | Desplazan toda el ala |
| `controls[]` | `name`, `y0`, `y1` (m), `chord_frac`, `le`, `enabled` |

Comandos: `wing-fit` (mide el ala de la máscara de planta), `wing-stations [--init | --load F]`, `wing-check` (compara lo esperado con lo leído de OpenVSP), `wing-export F.json [--fuselage] [--segments] [--no-controls]`. `AIRCRAFT_WING_SEGMENTS=1` fuerza paneles separados.

Detalle técnico: cada sección necesita su propio grupo de drivers (`SetDriverGroup`) y un `Update()` tras fijarla; sin ello OpenVSP ignora `Span` o guarda un área obsoleta y reescala el ala al reabrirla. El programa fija primero el perfil y luego los parámetros, y lee cada sección de vuelta.

## 6. Todos los comandos

| Grupo | Comandos |
|---|---|
| Diagnóstico | `check-env`, `selftest`, `selfcheck`, `timing`, `inspect`, `probe`, `probe-geom --type T`, `probe-subsurf`, `check-var-comp` |
| Proyecto | `init`, `migrate`, `wizard` y los de la sección 3 |
| Máscaras | `calibrate`, `rescale`, `erase-mask`, `restore-mask`, `masks --use core\|extras`, `landmarks` |
| Fuselaje | `fuselage-fit`, `fuselage-test`, `fuselage-simplify`, `simplify`, `smooth` |
| Góndolas y carenados | `nacelle-fit`, `fairings-fit [--clear]` |
| Optimización | `auto`, `optimize --free ...`, `front-fit` |
| Construir y medir | `build`, `compare`, `report-metrics` |
| Cambios y versiones | `set`, `delta`, `patch`, `history`, `diff`, `undo`, `checkout` |

Parámetros útiles: `model.has_propulsion`, `model.has_gear`, `model.has_details`, `gear.x`, `gear.nose_x`, `gear.track`, `gear.wheel_radius`, `gear.height_ref`, `gear.door_angle`, `window.count`, `window.x0`, `window.x1`, `window.w_side`.

Variables de entorno: `AIRCRAFT_PROJECT`, `AIRCRAFT_NO_SESSION=1` (fuerza reconstrucciones completas), `AIRCRAFT_VERBOSE=1`, `AIRCRAFT_WING_SEGMENTS=1`, `VSP_PYTHON`.

## 7. Componentes del modelo

- **Fuselaje:** N estaciones elípticas medidas de las siluetas. C2 está desactivado porque producía diente de sierra.
- **Ala y colas:** parámetros con ventanas absolutas. `Span` se compensa por el diedro para respetar la envergadura proyectada.
- **Góndolas y motores:** cuerpos de revolución con pilón hasta la parte inferior del ala.
- **Carenados:** cuerpos pequeños bajo el borde de salida.
- **Tren:** patas, bogie, ruedas, tren de morro y compuertas; solo en el modelo guardado.
- **Detalles:** superficies de mando y ventanas como subsuperficies; no afectan a las siluetas.

## 8. Qué se ha verificado y qué no

**Ejecutado en OpenVSP 3.53.1 en esta máquina, en una copia aislada:** `selftest`, `check-env`, `project-new`, `project-list`, `project-use`, `project-copy`, `build`, `wing-check`, `wing-stations`, `wing-fit`, `wing-export`, `calibrate`, `masks`, `fuselage-fit`, `simplify`, `compare`, `report-metrics`, `optimize`, `front-fit`, `nacelle-fit`, `fairings-fit`, `auto --phase airframe`, `set`, `delta`, `patch`, `diff`, `undo`, `checkout`, `history`, `inspect`, `landmarks`, `migrate`, `selfcheck`, `timing`, `probe-subsurf`.

**No ejecutado en esa prueba:** `erase-mask` y `restore-mask` en modo ventana, `wizard`, `auto` completo con sus tres fases y `fuselage-test`, porque son interactivos o largos.

**Fallo conocido:** `smooth` falla con `UnicodeEncodeError` al imprimir `≈` si la salida se redirige. Solución: `set PYTHONUTF8=1` antes de ejecutarlo.

**Límites:**

- Tres siluetas no determinan secciones, grosores ni perfiles: se tratan como valores inferidos.
- Las discrepancias entre dibujos hechos a mano no las corrige ningún parámetro.
- El parabrisas no está modelado.
- El programa construye perfiles simétricos (curvatura 0). Los perfiles supercríticos de `iteraciones_mom226` se aplicaron aparte, con sus scripts.
- La alineación es por referencia de la imagen (morro, eje): un error absoluto de posición de todo el avión no se ve sin puntos de referencia.

## 9. Criterios de diseño

- **Un solo agente:** el modelo de IA llama a herramientas deterministas y emite parches JSON, sin un segundo agente de visión.
- **Sin capturas para medir:** el modelo se exporta como malla STL y se proyecta de forma ortogonal.
- **La alineación no absorbe errores:** la escala sale de una cota fiable por vista.
- **Optimización por etapas:** pocos parámetros cada vez, con error por vista además del total.
- **Las cotas medidas son restricciones de igualdad,** no penalizaciones.
- **Puntos de referencia:** `landmarks` solo propone candidatos en `processed/*_landmarks.json`; no modifica parámetros ni geometría, y hay que verificarlos antes de usarlos.

## 10. Hoja de ruta

Interfaz web local para seleccionar motores y tren; localizar el error por regiones; mandos y ventanas medidos del dibujo; comparación de bordes; integrar VSPAERO en un comando.
