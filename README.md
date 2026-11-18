# Parametric aircraft reconstruction in OpenVSP

`aircraft_recon.py` is a single program that builds and checks aircraft models in OpenVSP 3.53.1, either from three drawn views or from JSON files. Dependencies: `numpy`, `opencv-python` and `scipy`, plus OpenVSP and its Python binding.

- **Step-by-step guide:** [aircraft_recon_guide.pdf](aircraft_recon_guide.pdf)
- **Validation case:** Airbus A300-600 (drawing "2.2 General Airplane Dimensions", redrawn by hand in three views)

> AI interprets; deterministic software measures and executes.

## 1. Installation

You need Windows, Git and Python 3.13, because the OpenVSP Python binding is built for that version.

1. Get the program:
   ```
   git clone https://github.com/IIAlbertyII/OpenVSPAuto.git
   cd OpenVSPAuto
   ```
2. Download OpenVSP for Windows (the Python 3.13 build) from [openvsp.org/download.php](https://openvsp.org/download.php) and unzip it. The program was tested with 3.53.1. It finds OpenVSP by itself in a folder whose name starts with `OpenVSP` inside `OpenVSPAuto`, next to it, or in your Downloads, Desktop or Documents folder. If OpenVSP is anywhere else, the program asks for its folder the first time and remembers it in `openvsp_path.txt`; you can also set the `VSP_PYTHON` variable to that folder.
3. Create the environment with Python 3.13 and install the dependencies:
   ```
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   pip install numpy opencv-python scipy six
   ```
   `six` is needed by the Python packages that come with OpenVSP; those packages are loaded from the OpenVSP folder, so they do not need installing. With `uv`, the first line is `uv venv --python 3.13 .venv`. In each new session, only the `activate` line is needed.
4. Check that everything works:
   ```
   python aircraft_recon.py selftest      # must end with SELFTEST OK
   python aircraft_recon.py check-env     # must print OpenVSP OK and the version
   ```

Keep the program and its temporary files in paths without accents or other non-ASCII characters. All commands and paths in this document start from the program folder.

## 2. Repository contents

```
aircraft_recon.py          the program
README.md                  this document
aircraft_recon_guide.pdf   step-by-step guide
```

OpenVSP, your models and your drawings are not part of the repository. The program creates these files and folders as you use it:

```
openvsp_path.txt           OpenVSP folder you gave when asked (section 1)
aircraft/
  current.txt              selected project
  <NAME>/                  one folder per aircraft (section 3)
project/                   legacy single-project folder, used when no project is selected
```

## 3. Projects

Each aircraft lives in `aircraft/<NAME>/`, with its parameters, masks, versions, `fuselage.json`, `wing.json` and `fairings.json`.

| Command | Purpose |
|---|---|
| `project-new NAME [--params F] [--fuselage F] [--wing F]` | Create and select a project |
| `project-list`, `project-use NAME` | List or select |
| `project-copy SOURCE TARGET [--full] [--use]` | Copy, without versions unless `--full` |
| `project-adopt NAME` | Move the legacy `project/` folder to `aircraft/NAME` |
| `--project NAME command` | Use another project for that command only; goes before the command |

With no project selected, `project/` is used. The `AIRCRAFT_PROJECT` variable also selects one. Every command prints `[project: ...]` when it starts.

Project layout:

```
params.json     parameters with evidence (measured / calculated / inferred / unknown), confidence and fixed
views.json      calibration of each view (px/m, bbox, length)
fuselage.json   fuselage stations
wing.json       multi-section wing
fairings.json   flap-track fairings
state.json      current version and tessellation scale
processed/      masks: *_mask.png (active), *_mask_full.png, *_mask_core.png (no engines), *_mask_extras.png (with engines)
versions/ iterations/ renders/
```

## 4. Workflow

### From three views

```
project-new MY_AIRCRAFT
calibrate --view top   --image top.png   --length 53.85 --keep-parts
calibrate --view side  --image side.png  --length 54.08 --keep-parts
calibrate --view front --image front.png --length 44.84 --keep-parts
erase-mask --view top        # remove engines and landing gear; repeat for each view
restore-mask --view top      # put back only the engines; creates core and extras
set fuselage.max_width 5.64 --force
set fuselage.max_height 5.64 --force
auto --step --masks
compare
report-metrics
```

`wizard` walks through these steps interactively. The lengths are the A300's. In `params.json`, mark reliable dimensions as `measured` with `"fixed": true`, and the optimiser will not change them. If a scale is wrong, `rescale --view V --length L` fixes it without redoing the mask.

`auto` has three phases:

| Phase | What it does | Masks |
|---|---|---|
| `airframe` | Fuselage stations, wing, tails, dihedral and thickness | core |
| `extras` | Nacelles, pylons, fairings and refinement | extras |
| `finish` | Landing gear, control surfaces, windows, fine mesh | — |

Options: `--phase`, `--passes`, `--iters`, `--step` (pause after each step), `--masks`, `--refit`, `--no-finish`, `--no-guard` (turn off the automatic rollback when a stage gets more than 2 % worse), `--pchip-simplify`.

### From JSON files, without images

```
project-new demo --params my_params.json --fuselage my_fuselage.json --wing my_wing.json
build
wing-check
```

### Review and undo

`compare` writes overlays to `renders/` (red: drawing only; blue: model only). `report-metrics` gives IoU, chamfer, Hausdorff-95 and area and extent errors. `history --last N`, `diff A B`, `undo` and `checkout N` manage versions. Every change saves `versions/vNNN.vsp3` and `.json`.

Manual changes: `set`, `delta` and `patch FILE.json`. Patch format:

```json
[{"operation":"modify","parameter":"wing.x","delta":-0.3,"reason":"wing 300 mm too far forward"}]
```

Parameters marked `measured` or `fixed` are refused unless you add `--force`.

## 5. Multi-section wing (`wing.json`)

Metres and degrees; `y` is the distance from the centreline on one half-wing.

| Field | Meaning |
|---|---|
| `stations[]` | `y`, `chord`, `x_le` (absolute), `z` (optional), `thickness` (t/c) or `t_rel`, `twist`, `winglet: true` (not counted in the planform span) |
| `root_perpendicular` | `true` (default): from the centreline to the fuselage side, chord and leading edge stay constant |
| `z_from_params` | `true`: `z = wing.z + (y − y_root)·tan(wing.dihedral)` |
| `twist_mode` | `absolute` (default) or `incremental`. OpenVSP does not chain section twists |
| `u_mode` | `span` (default) or `section`: how a spanwise position maps to the control surface U coordinate |
| `x_offset`, `z_offset` | Move the whole wing |
| `controls[]` | `name`, `y0`, `y1` (m), `chord_frac`, `le`, `enabled` |

Commands: `wing-fit` (measures the wing from the top mask), `wing-stations [--init | --load F]`, `wing-check` (compares expected values with those read back from OpenVSP), `wing-export F.json [--fuselage] [--segments] [--no-controls]`. `AIRCRAFT_WING_SEGMENTS=1` forces separate panels.

Technical note: each section needs its own driver group (`SetDriverGroup`) and an `Update()` after it is set; otherwise OpenVSP ignores `Span` or stores an outdated area and rescales the wing when the file is reopened. The program sets the airfoil first, then the parameters, and reads every section back.

## 6. All commands

| Group | Commands |
|---|---|
| Diagnostics | `check-env`, `selftest`, `selfcheck`, `timing`, `inspect`, `probe`, `probe-geom --type T`, `probe-subsurf`, `check-var-comp` |
| Project | `init`, `migrate`, `wizard` and those in section 3 |
| Masks | `calibrate`, `rescale`, `erase-mask`, `restore-mask`, `masks --use core\|extras`, `landmarks` |
| Fuselage | `fuselage-fit`, `fuselage-test`, `fuselage-simplify`, `simplify`, `smooth` |
| Nacelles and fairings | `nacelle-fit`, `fairings-fit [--clear]` |
| Optimisation | `auto`, `optimize --free ...`, `front-fit` |
| Build and measure | `build`, `compare`, `report-metrics` |
| Changes and versions | `set`, `delta`, `patch`, `history`, `diff`, `undo`, `checkout` |

`python aircraft_recon.py COMMAND --help` lists the options of each command.

Useful parameters: `model.has_propulsion`, `model.has_gear`, `model.has_details`, `gear.x`, `gear.nose_x`, `gear.track`, `gear.wheel_radius`, `gear.height_ref`, `gear.door_angle`, `window.count`, `window.x0`, `window.x1`, `window.w_side`.

Environment variables: `AIRCRAFT_PROJECT`, `AIRCRAFT_NO_SESSION=1` (forces full rebuilds), `AIRCRAFT_VERBOSE=1`, `AIRCRAFT_WING_SEGMENTS=1`, `VSP_PYTHON` (OpenVSP folder; takes priority over the automatic search). `check-env --vsp-path FOLDER` tests a given OpenVSP folder.

## 7. Model components

- **Fuselage:** N elliptical stations measured from the silhouettes. C2 continuity is off because it produced a sawtooth.
- **Wing and tails:** parameters with absolute bounds. `Span` is corrected for dihedral so the projected span is kept.
- **Nacelles and engines:** bodies of revolution with a pylon up to the wing's lower surface.
- **Fairings:** small bodies under the trailing edge.
- **Landing gear:** struts, bogie, wheels, nose gear and doors; only in the saved model.
- **Details:** control surfaces and windows as sub-surfaces; they do not change the silhouettes.

## 8. What has been tested

**Run in OpenVSP 3.53.1 on Windows, on an isolated copy:** `selftest`, `check-env`, `project-new`, `project-list`, `project-use`, `project-copy`, `build`, `wing-check`, `wing-stations`, `wing-fit`, `wing-export`, `calibrate`, `masks`, `fuselage-fit`, `simplify`, `compare`, `report-metrics`, `optimize`, `front-fit`, `nacelle-fit`, `fairings-fit`, `auto --phase airframe`, `set`, `delta`, `patch`, `diff`, `undo`, `checkout`, `history`, `inspect`, `landmarks`, `migrate`, `selfcheck`, `timing`, `probe-subsurf`.

**Not run in that test:** `erase-mask` and `restore-mask` in window mode, `wizard`, the full three-phase `auto` and `fuselage-test`, because they are interactive or long.

**Known issue:** `smooth` fails with `UnicodeEncodeError` when printing `≈` if the output is redirected. Workaround: run `set PYTHONUTF8=1` first.

**Limits:**

- Three silhouettes do not determine cross-sections, thickness or airfoils: these are treated as inferred values.
- No parameter can correct disagreements between hand-made drawings.
- The windscreen is not modelled.
- The program builds symmetric airfoils (zero camber).
- Alignment uses image references (nose, axis): an absolute position error of the whole aircraft cannot be seen without reference points.

## 9. Design principles

- **One agent:** the AI model calls deterministic tools and emits JSON patches, with no second vision agent.
- **No screenshots for measuring:** the model is exported as an STL mesh and projected orthographically.
- **Alignment does not absorb errors:** the scale comes from one reliable dimension per view.
- **Staged optimisation:** few parameters at a time, with an error per view as well as the total.
- **Measured dimensions are equality constraints,** not penalties.
- **Reference points:** `landmarks` only proposes candidates in `processed/*_landmarks.json`; it does not change parameters or geometry, and the candidates must be checked before use.

## 10. Roadmap

Local web interface to choose engines and landing gear; locate the error by region; control surfaces and windows measured from the drawing; edge comparison; VSPAERO in one command.
