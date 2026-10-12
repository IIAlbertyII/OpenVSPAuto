# Reconstrucción paramétrica de aeronaves en OpenVSP — estado actual

Un solo archivo (`aircraft_recon.py`), sin instaladores, solo `pip install numpy opencv-python scipy` y OpenVSP 3.53.1 con su binding Python (compilado para Python 3.13).
Caso de validación: **Airbus A300-600** (lámina "2.2 General Airplane Dimensions" redibujada a mano, tres vistas por separado).

## 0. Novedades de esta versión

- **`auto --step`**: se detiene tras cada paso, escribe overlays y métricas y pregunta `[Enter] continuar | [u] deshacer este paso | [q] salir`.
- **Guarda anti-empeoramiento** (activa por defecto, `--no-guard` la quita): si una etapa empeora el error global más de un 2 %, se revierte sola.
- **Máscaras dentro de `auto`**: `auto --masks` (o, si faltan las variantes core/extras, pregunta al empezar) abre la selección de motores y tren por vista y deja listas las máscaras de las fases 1 y 2.
- **`calibrate --keep-parts`**: conserva las partes sueltas cercanas al avión (las ruedas del tren), que antes se descartaban, para poder seleccionarlas con `erase-mask`.
- **`init` flexible**: la primera calibración rellena `fuselage.length` y `wing.span` si siguen siendo valores de plantilla; el aviso de calibración incoherente solo compara con parámetros ya conocidos. Sirve para cualquier avión.
- **Actualización en el sitio con autocomprobación**: lee de vuelta `wing.x` tras cada actualización y, si no se aplicó, o si ocho evaluaciones distintas dan exactamente el mismo error, cambia sola a reconstrucciones completas. `selfcheck` compara la actualización en el sitio con una reconstrucción completa; `AIRCRAFT_NO_SESSION=1` fuerza reconstrucciones completas; `AIRCRAFT_VERBOSE=1` imprime cada evaluación.

## 1. Idea de diseño

> La IA interpreta; el software determinista mide y ejecuta.

- Las siluetas de referencia (máscaras) se comparan con las siluetas del modelo (malla STL exportada de OpenVSP y proyectada de forma ortogonal): IoU, distancia chamfer, Hausdorff-95, error de área y de extensión.
- Lo que se puede **medir** se mide (fuselaje por estaciones, góndolas, carenados). Lo que no, se **optimiza** con pocos parámetros por etapa (Powell, ventanas acotadas).
- Cada cambio genera una **versión** (`project/versions/vNNN.vsp3` + `.json`) que se puede deshacer.
- Las cotas medidas del plano (53,85 m, 5,64 m, 44,84 m...) están marcadas `measured` y `fixed`: el optimizador no las toca.

## 2. Estructura del proyecto

```
aircraft_recon.py
project/
  params.json         parámetros con evidencia (measured / calculated / inferred / unknown), confianza, fixed
  views.json          calibración de cada vista (px/m, bbox, longitud usada)
  fuselage.json       estaciones del fuselaje medidas de las máscaras + modo de construcción
  fairings.json       carenados de las guías de flaps (conteo/altura de la vista frontal, x de la planta)
  state.json          versión actual y escala de teselación
  processed/          máscaras: *_mask.png (activa), *_mask_full.png (original),
                      *_mask_core.png (sin motores), *_mask_extras.png (con motores)
  versions/ iterations/ renders/
```

## 3. Flujo de trabajo (el que seguimos)

**0. Entorno** (una vez): `check-env` debe decir `OpenVSP OK`. Usa siempre el entorno de Python 3.13.

**1. Máscaras y calibración**
1. `calibrate --view top|side|front --image ... --length L` con la longitud real: **53,85 m** (planta), **54,08 m** (perfil), **44,84 m** (frontal). Si te equivocas: `rescale --view V --length L` (no rehace la máscara). `inspect` y los comandos principales avisan si la escala no encaja.
2. `erase-mask --view V` para quitar motores y tren de cada máscara (rectángulos con el ratón; se guarda `*_mask_full.png`).
3. `restore-mask --view V` para volver a poner **solo los motores**. Esto guarda automáticamente `*_mask_core.png` (sin motores) y `*_mask_extras.png` (con motores). `masks --use core|extras` cambia la máscara activa.

**2. Reconstrucción automática: `auto`**

| Fase | Qué hace | Máscaras |
|---|---|---|
| 1 `airframe` | fuselaje por estaciones (solo), ala, estabilizadores, diedros, grosores; modelo sin motores | core (sin motores) |
| 2 `extras` | góndolas medidas de planta+frontal, pilones, carenados de flaps (vista frontal: cuántos, altura; planta: posición), refinado | extras (con motores) |
| 3 `finish` | tren, superficies de mando, ventanas, malla fina | — |

`python aircraft_recon.py auto` ejecuta las tres; `auto --phase airframe|extras|finish` una sola. `--passes`, `--iters`, `--refit`, `--no-finish`.

**3. Revisión**: `compare` (overlays rojo = solo dibujo, azul = solo modelo; `project/renders/`), `report-metrics`, `history --last 10`, `diff A B`, `undo`, `checkout N`.

## 4. Comandos

| Comando | Para qué |
|---|---|
| `check-env`, `probe`, `probe-geom --type T`, `probe-subsurf` | diagnóstico del OpenVSP instalado |
| `init`, `migrate` | crear proyecto / añadir parámetros nuevos |
| `calibrate`, `rescale`, `erase-mask`, `restore-mask`, `masks` | máscaras y escala |
| `fuselage-fit`, `fuselage-test`, `smooth` | fuselaje (medido de las siluetas), variantes de construcción, malla fina |
| `nacelle-fit`, `fairings-fit [--clear]` | góndolas y carenados medidos |
| `auto`, `optimize`, `front-fit` | optimización |
| `build`, `compare`, `report-metrics`, `inspect`, `timing`, `selfcheck` | construir / medir / diagnosticar / cronometrar / comprobar la ruta rápida |
| `set`, `delta`, `patch` | cambios manuales y parches estructurados |
| `history`, `diff`, `undo`, `checkout` | versiones |

## 5. Componentes del modelo

- **Fuselaje**: N estaciones elípticas (anchura de la planta; línea superior e inferior del perfil; ancho y alto limitados al diámetro medido). Espaciado, tangentes y extremos se eligen con `fuselage-test`. **C2 desactivado**: producía diente de sierra.
- **Ala, estabilizador horizontal, deriva**: parámetros con ventanas absolutas (grosor 0,08–0,20, diedro 0–9°/12°, flechas acotadas). `Span` se compensa por el diedro para que la envergadura proyectada coincida con la cota.
- **Góndolas y motores**: cuerpos de revolución simétricos + **pilón** (placa vertical hasta la parte inferior del ala). Se construyen con las mismas llamadas probadas del fuselaje (no con POD, que se colocaba mal).
- **Carenados de flaps**: cuerpos pequeños bajo el borde de salida; `fairings.json` guarda x, y, z, longitud, anchura y altura.
- **Tren** (solo en el modelo guardado): patas, viga de bogie, 4 ruedas por pata principal, tren de morro, paneles de compuertas. Suelo calculado con la cota de 16,66 m.
- **Detalles** (subsuperficies, no afectan a las siluetas): slat, 2 flaps, alerón, timón de profundidad y de dirección; ventanas laterales de cabina y fila de ventanillas opcional (`window.count`).

## 6. Banderas y parámetros útiles

`model.has_propulsion`, `model.has_gear`, `model.has_details` (0/1) · `gear.x` (25,27) · `gear.nose_x` (6,67) · `gear.track` (9,6) · `gear.wheel_radius` (≈0,63) · `gear.height_ref` (16,66) · `gear.door_angle` · `window.count`, `window.x0`, `window.x1`, `window.w_side`.

## 7. Rendimiento

La evaluación usa malla gruesa (`FAST_SCALE`), máscaras a mitad de resolución (`EVAL_DS`), el fuselaje en caché y **actualización de parámetros en el sitio** (sin reconstruir) mientras la estructura no cambie. Cada etapa se detiene sola si en 24 evaluaciones no mejora un 0,2 %. Se guarda **una** versión por pasada. `timing` muestra dónde se va el tiempo.

## 8. Qué se sabe y qué no

- **Verificado en tu máquina**: entorno, calibración, fuselaje por estaciones, ajuste de ala y cola, construcción de góndolas, pilones y carenados.
- **Verificado solo con simulación de la API** (no en OpenVSP real): sentido de giro de las patas del tren, ángulo de compuertas, nombres de parámetros de subsuperficies, tapa redonda del fuselaje. `probe-subsurf` y una captura bastan para corregirlo.
- **Límites conocidos**: tres siluetas no determinan secciones, grosores ni perfiles; las discrepancias entre dibujos hechos a mano (p. ej. envergadura en planta ≈1 % mayor que en la frontal) no las corrige ningún parámetro; el morro/parabrisas no está modelado (depende de cómo OpenVSP numera la vuelta de la sección).

## 9. Hoja de ruta

Interfaz web local (FastAPI) para seleccionar motores/tren sobre la imagen; localizar por regiones dónde está el error y optimizar solo ahí; superficies de mando y ventanas con posiciones medidas del dibujo; VSPAERO para validación aerodinámica.
