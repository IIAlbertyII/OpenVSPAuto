# AI Aircraft Reconstruction — Review + Quick Start

## A. Review of the proposal

### Keep (strong decisions)
- "AI interprets, deterministic code measures" (section 18).
- Evidence classes + provenance, versioning/undo, protected measured dimensions.
- OpenVSP behind an adapter; structured, validated patches instead of raw API calls.
- Multi-view objective instead of per-image optimisation.

### Change before implementing
1. **Python/OpenVSP compatibility (section 10).** The compiled `_vsp*.pyd` is built for one specific Python version (look for the `cpXYZ` tag). "Python 3.13" is an assumption; run `check-env`. If the tag differs, create a venv with the matching Python. Install the packages from the OpenVSP `python` folder into that venv, not a PyPI package that merely shares the name.
2. **One agent, not two.** Agent 2 (vision) is mostly deterministic CV. Use one LLM orchestrator that calls deterministic tools (calibrate, measure, compare, optimise) and emits JSON patches. A second LLM adds cost and a new hallucination surface without a new capability.
3. **Three silhouettes do not determine a 3D aircraft.** Width comes from top, height from side, cross-section shape from neither. Wing thickness, twist, airfoil and fuselage section shape are not observable in outlines. Treat them as `inferred` priors (e.g. elliptical sections, a known airfoil), and add cross-section station lines or photos when available. The system must report "not identifiable" instead of optimising it.
4. **Silhouette-only comparison is blind to interior features** (control surfaces, canopy, nacelles, panel lines). Add line/edge comparison (chamfer between drawing edges and model feature edges) as a second stage.
5. **Do not screenshot OpenVSP.** Export a mesh (STL) and rasterise it orthographically with NumPy/OpenCV. Headless, exact scale, deterministic. (Implemented.)
6. **Registration must not absorb errors.** Calibrate scale from one trusted dimension per view; align only by datum (nose, centreline). Never let the optimiser's objective also choose the alignment. Current code aligns by bbox datum and says so; absolute position errors of the whole aircraft are therefore invisible until datums (e.g. wing LE, a landing gear point) are added as landmarks.
7. **Optimise in stages, few parameters at a time.** Wing planform from the top view first, vertical positions from the side, widths from top/front. A 20-parameter Nelder-Mead on a weighted sum is ill-conditioned and hides trade-offs. Report per-view errors, not only the total.
8. **Measured dimensions are equality constraints, not penalties.** Also detect contradictory dimensions in the same drawing (scale errors are common).
9. **Dimension extraction is the riskiest step.** Start semi-manual: you type/confirm the dimensions, the LLM only suggests. A value becomes `measured` only after confirmation. Vector sources (PDF/DXF) beat raster scans.
10. **Provenance:** add uncertainty intervals (`±`), source file hash, and for `calculated` values the list of parameters they depend on.
11. **Define tolerance numerically up front** (e.g. Hausdorff95 ≤ 0.3 % of length, IoU ≥ 0.995 per view).
12. **Validation (section 28.17):** (a) round-trip test — generate three views from a *known* OpenVSP model, reconstruct, and measure parameter error; (b) hold out a view or a dimension and check it is predicted; (c) compare derived quantities (planform area, MAC) with published values.
13. **OpenVSP limits:** excellent for wings/fuselages/nacelles, weaker for blended fairings, intakes and detailed canopies. Plan to import cross-sections/airfoils from files and accept simplified components.

### Order of work
Do now: environment check, adapter, evidence/versioning, calibration, silhouette comparison, constrained optimiser (all in `aircraft_recon.py`).
Next: landmarks, edge comparison, airfoil import, LLM patch generation (`patch` command is the interface).
Postpone: FastAPI/web UI, conversational layer, VSPAERO, advanced geometry.

## B. Quick start (Windows, text files only)

```
python -m venv venv && venv\Scripts\activate
pip install numpy opencv-python scipy
python aircraft_recon.py selftest                  # offline sanity test
python aircraft_recon.py check-env                 # OpenVSP import + version
python aircraft_recon.py init                      # creates project\
```
1. Edit `project\params.json`: real values, `evidence_class`, `source`, `confidence`; set `"fixed": true` for dimensions you trust. Optional `"min"`/`"max"` per parameter.
2. Calibrate each view (nose on the left; use `--flip` otherwise). `--length` is the real horizontal extent of the silhouette: overall length for top/side, span for front:
```
python aircraft_recon.py calibrate --view top   --image drawings\top.png   --length 12.40
python aircraft_recon.py calibrate --view side  --image drawings\side.png  --length 12.40
python aircraft_recon.py calibrate --view front --image drawings\front.png --length 11.00
```
   Open `project\processed\*_mask.png`. If dimension lines or text are in the mask, paint a clean silhouette (white on black) and pass it with `--mask`.
3. `python aircraft_recon.py build` then `compare` (overlays in `project\renders\overlay_*.png`: red = drawing only, blue = model only).
4. Optimise a small group of unfixed parameters:
```
python aircraft_recon.py optimize --free wing.sweep wing.x wing.tip_chord --weights top=1 side=0.3
```
5. Manual edits: `set wing.x 5.1`, `delta wing.x -0.3`, `history`, `undo`, `checkout 3`. Structured patches (the format an LLM would emit):
```json
[{"operation":"modify","parameter":"wing.x","delta":-0.3,"reason":"user: wing 300 mm too far forward"}]
```
   Apply with `patch my_patch.json`. Measured/fixed parameters are rejected unless `--force`.

## C. Limits of this first version (be aware)
- The OpenVSP adapter was written against the documented Python API but could not be run here (no OpenVSP in this environment); only the vision/metrics side was tested (`selftest`). If `build` fails, the error will name the OpenVSP parameter or call; fix it inside `OpenVSPAdapter` only (names can differ between versions).
- Fuselage uses 3 inner elliptical sections with a fixed shape prior (`FUSE_PROFILE`); detailed sections are a later phase.
- Alignment is by bounding-box datum; no landmark or edge comparison yet; no dimension OCR; no LLM agent or web UI yet.
