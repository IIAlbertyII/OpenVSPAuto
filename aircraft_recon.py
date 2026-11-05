#!/usr/bin/env python3
"""
aircraft_recon.py - core reconstruction loop (single text file, pip-only deps)

pip install numpy opencv-python scipy

Commands:
  selftest | check-env | check-var-comp | init | migrate
  calibrate | rescale | landmarks | erase-mask | restore-mask
  auto | fuselage-fit | fuselage-test | smooth | front-fit | nacelle-fit [--no-init]
  build | compare | report-metrics
  optimize | set | delta | patch
  inspect | probe | probe-subsurf | probe-geom --type T [--all]
  fairings-fit | set model.has_gear 1 | set model.has_details 1
  wing-fit | wing-stations [--init|--load F] | wing-check | wing-export F.json [--fuselage]
  history | undo | checkout N | diff [A] [B]
  project-new NAME [--params F] [-a-fuselage F] [--wing F] | project-list | project-use NAME
  project-copy SRC DST [--full] [--use] | project-adopt NAME | --project NAME <command> (option goes before the command)
Coordinates: X nose->tail, Y right, Z up (OpenVSP native). Units: m, deg.
"""
import argparse, glob, json, os, re, shutil, sys, tempfile, time, platform, types
from pathlib import Path
import numpy as np
HOME = Path(__file__).resolve().parent / "project"
DIRS = ["drawings", "processed", "versions", "iterations", "renders"]
PARAMS, VIEWS_F, STATE = HOME / "params.json", HOME / "views.json", HOME / "state.json"
FUSE_F = HOME / "fuselage.json"
FAIR_F = HOME / "fairings.json"      # flap-track fairings measured from the plan view
WING_F = HOME / "wing.json"          # multi-section (yehudi) main wing: one record per spanwise station

KEYS = {
    "fuselage.length": (12.0, "m"), "fuselage.max_width": (1.4, "m"), "fuselage.max_height": (1.6, "m"),
    "wing.x": (4.8, "m"), "wing.z": (0.0, "m"), "wing.span": (11.0, "m"),
    "wing.root_chord": (2.1, "m"), "wing.tip_chord": (0.9, "m"),
    "wing.sweep": (20.0, "deg"), "wing.dihedral": (3.0, "deg"),
    "htail.x": (10.6, "m"), "htail.z": (0.4, "m"), "htail.span": (4.0, "m"),
    "htail.root_chord": (1.3, "m"), "htail.tip_chord": (0.7, "m"), "htail.sweep": (20.0, "deg"),
    "htail.dihedral": (5.0, "deg"),
    "vtail.x": (10.2, "m"), "vtail.z": (0.6, "m"), "vtail.height": (2.0, "m"),
    "vtail.root_chord": (2.0, "m"), "vtail.tip_chord": (1.0, "m"), "vtail.sweep": (35.0, "deg"),
    "wing.thickness": (0.12, "frac"), "htail.thickness": (0.10, "frac"), "vtail.thickness": (0.10, "frac"),
    "model.has_propulsion": (0.0, "flag"),
    "nacelle.x": (5.2, "m"), "nacelle.y": (2.5, "m"), "nacelle.z": (-0.35, "m"),
    "nacelle.length": (2.8, "m"), "nacelle.diameter": (1.0, "m"), "nacelle.sweep": (0.0, "deg"),
    "engine.x": (5.4, "m"), "engine.y": (2.5, "m"), "engine.z": (-0.35, "m"),
    "engine.length": (3.2, "m"), "engine.diameter": (0.9, "m"),
    "gear.x": (4.6, "m"), "gear.z": (-1.25, "m"),
    "gear.track": (3.0, "m"), "gear.wheel_radius": (0.28, "m"), "gear.strut_length": (1.2, "m"),
    "model.has_gear": (0.0, "flag"), "model.has_details": (0.0, "flag"),
    "gear.nose_x": (6.67, "m"), "gear.height_ref": (16.66, "m"), "gear.door_angle": (65.0, "deg"),
    "window.count": (0.0, "count"), "window.x0": (9.0, "m"), "window.x1": (46.0, "m"), "window.w_side": (0.25, "frac"),
}
# body-of-revolution profiles: (x / length, diameter / max diameter)
NAC_PROF = [(0.00, 0.0), (0.02, 0.80), (0.08, 0.94), (0.22, 1.0), (0.55, 1.0), (0.78, 0.90), (0.93, 0.68), (1.00, 0.0)]
ENG_PROF = [(0.0, 0.0), (0.05, 0.85), (0.30, 1.0), (0.70, 1.0), (0.95, 0.80), (1.0, 0.0)]
FAIR_PROF = [(0.0, 0.0), (0.10, 0.70), (0.35, 1.0), (0.70, 0.90), (0.95, 0.45), (1.0, 0.0)]
STRUT_PROF = [(0.0, 0.0), (0.01, 1.0), (0.50, 1.0), (0.99, 1.0), (1.0, 0.0)]
WHEEL_PROF = [(0.0, 0.0), (0.05, 0.82), (0.12, 0.97), (0.50, 1.0), (0.88, 0.97), (0.95, 0.82), (1.0, 0.0)]
# control surfaces as sub-surfaces: (name, span start, span end, chord fraction, leading edge?)
WING_CONTROLS = [("Slat", 0.20, 0.97, 0.14, True), ("Flap_inboard", 0.12, 0.40, 0.26, False),
                 ("Flap_outboard", 0.42, 0.70, 0.26, False), ("Aileron", 0.72, 0.96, 0.20, False)]
TAIL_CONTROLS = [("Elevator", 0.08, 0.96, 0.30, False)]
RUDDER = [("Rudder", 0.05, 0.95, 0.32, False)]
TESS_W, TESS_U = 33, 12
SIMPLIFY_TOL = 0.01      # fuselage stations within this fraction of the diameter of the line between neighbours are dropped
FAST_SCALE = 0.6      # mesh quality used while evaluating silhouettes (final models keep the fine mesh)
EVAL_DS = 0.5         # evaluation masks are downscaled by this factor during optimisation
TESS_W_FINE, TESS_U_FINE = 61, 25
FUSE_STATIONS_FINE = 51
FUSE_PROFILE = [0.8, 1.0, 0.7]
VIEWS = {"top": (0, 1, 1), "side": (0, 2, -1), "front": (1, 2, -1)}
ALIGN = {"top": ("min", "center"), "side": ("min", "min"), "front": ("center", "min")}
DEFAULT_W = {"top": 1.0, "side": 1.0, "front": 1.0}

_TESS_SCALE_CACHE = None
_FAST = False
_AIRFRAME_ONLY = False   # True: evaluate/build the bare airframe (no nacelles, fairings, gear, details)          # True while optimising/comparing: coarse mesh (fast); committed models use the persisted scale


class airframe_only:
    """Phase 1 works on the airframe alone, against masks without engines."""
    def __enter__(self):
        global _AIRFRAME_ONLY
        self.prev, _AIRFRAME_ONLY = _AIRFRAME_ONLY, True
    def __exit__(self, *exc):
        global _AIRFRAME_ONLY
        _AIRFRAME_ONLY = self.prev


class fast_tess:
    """Coarse tessellation while evaluating silhouettes; the saved .vsp3 keeps the fine one."""
    def __enter__(self):
        global _FAST
        self.prev, _FAST = _FAST, True
    def __exit__(self, *exc):
        global _FAST
        _FAST = self.prev


def _get_tess_scale():
    global _TESS_SCALE_CACHE
    if _FAST:
        return FAST_SCALE
    if _TESS_SCALE_CACHE is None:
        st = jload(STATE, {})
        try:
            _TESS_SCALE_CACHE = float(st.get("tess_scale", 1.0))
        except Exception:
            _TESS_SCALE_CACHE = 1.0
    return _TESS_SCALE_CACHE

def _set_tess_scale(s):
    global _TESS_SCALE_CACHE
    _TESS_SCALE_CACHE = float(s)
    st = jload(STATE, {"current": 0}) or {"current": 0}
    st["tess_scale"] = float(s)
    jsave(STATE, st)


def jload(p, default=None):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else default

def jsave(p, o):
    Path(p).write_text(json.dumps(o, indent=2, ensure_ascii=False), encoding="utf-8")

_WARNED = set()

def resolve_wing_stations(wj, V):
    """wing.json -> absolute stations [{y, chord, x_le, z, thickness, twist, winglet}] (metres, degrees).
    Where there is no data (inside the fuselage) the wing is made perpendicular to it: constant chord and leading edge
    from the centre line to the fuselage side (`root_perpendicular`, default true)."""
    raw = sorted((dict(s) for s in wj.get("stations", [])), key=lambda s: float(s["y"]))
    if len(raw) < 2:
        raise ValueError("wing.json needs at least 2 stations")
    fus_half = float(V.get("fuselage.max_width", 0.0)) / 2.0
    dih = float(np.tan(np.radians(float(V.get("wing.dihedral", 0.0)))))
    wz = float(V.get("wing.z", 0.0)); tc0 = float(V.get("wing.thickness", 0.12))
    xoff = float(wj.get("x_offset", 0.0)); zoff = float(wj.get("z_offset", 0.0))
    st = []
    for s in raw:
        t = float(s["thickness"]) if "thickness" in s else tc0 * float(s.get("t_rel", 1.0))
        st.append({"y": float(s["y"]), "chord": float(s["chord"]), "x_le": float(s["x_le"]) + xoff,
                   "z": (None if s.get("z") is None else float(s["z"]) + zoff), "thickness": t,
                   "twist": float(s.get("twist", 0.0)), "winglet": bool(s.get("winglet", False))})
    if st[0]["y"] > 1e-6:
        st.insert(0, dict(st[0], y=0.0))
    yj = 0.0
    if bool(wj.get("root_perpendicular", True)):
        jn = next((s for s in st if s["y"] >= 0.8 * fus_half and s["y"] > 1e-6), st[-1])
        yj = jn["y"]
        for s in st:
            if s["y"] < yj - 1e-6:
                for k in ("chord", "x_le", "thickness", "twist", "z"):
                    s[k] = jn[k]
        st = [s for s in st if s["y"] < 1e-6 or s["y"] >= yj - 1e-6]
    zfp = bool(wj.get("z_from_params", False))
    for s in st:
        if s["z"] is None or zfp:
            # FIX: the dihedral line starts at the centre line, as in the single-panel wing and in wing.z/wing.dihedral (flat inside the
            # fuselage). Measuring from the fuselage side lowered the A300 tip 0.28 m after wing-fit: front IoU 0.73 -> 0.61.
            s["z"] = wz + max(s["y"], yj) * dih
    out = []
    for s in st:                                   # merge stations closer than 5 cm (zero-span sections break OpenVSP)
        if out and s["y"] - out[-1]["y"] < 0.05:
            out[-1] = s if s is st[-1] else out[-1]
            continue
        out.append(s)
    for s in out:
        if not (0.2 <= s["chord"] <= 40.0):
            raise ValueError(f"wing.json: chord {s['chord']:.2f} m at y={s['y']:.2f} is not plausible")
    if len(out) < 2:
        raise ValueError("wing.json: fewer than 2 usable stations")
    return out

def wing_half(st):
    """Half-span of the planform (the winglet, if any, is not counted)."""
    body = [s for s in st if not s.get("winglet")]
    return (body[-1] if body else st[-1])["y"]

def values(P):
    out = {k: e["value"] for k, e in P.items()}
    fs = jload(FUSE_F)
    if fs and fs.get("stations"):
        out["_stations"] = fs["stations"]
        out["_fuse_c2"] = bool((fs.get("mode") or {}).get("c2", False))
        out["_fuse_tang"] = (fs.get("mode") or {}).get("tang", "off")
        out["_fuse_ends"] = (fs.get("mode") or {}).get("ends", "point")
    fa = jload(FAIR_F)
    if fa and fa.get("items"):
        out["_fairings"] = fa["items"]
    wj = jload(WING_F)
    if wj and wj.get("stations") and not wj.get("disabled"):
        try:
            out["_wing_stations"] = resolve_wing_stations(wj, out)
            out["_wing_ctrl"] = wj.get("controls")
            out["_wing_umode"] = wj.get("u_mode", "span")
            out["_wing_twist_mode"] = wj.get("twist_mode", "absolute")
        except (ValueError, KeyError, TypeError) as e:
            if str(e) not in _WARNED:
                _WARNED.add(str(e)); print(f"WARNING: wing.json ignored ({e}); using the single-panel wing")
    if _AIRFRAME_ONLY:
        for k in ("model.has_propulsion", "model.has_gear", "model.has_details"):
            out[k] = 0.0
        out.pop("_fairings", None)
    return out

LIMITS = {   # physically plausible absolute ranges (the search window never leaves them)
    "wing.thickness": (0.08, 0.20), "htail.thickness": (0.07, 0.16), "vtail.thickness": (0.07, 0.16),
    "wing.dihedral": (0.0, 9.0), "htail.dihedral": (0.0, 12.0),
    "wing.sweep": (20.0, 40.0), "htail.sweep": (20.0, 45.0), "vtail.sweep": (30.0, 55.0),
}

def bounds(e, wide=1.0, key=None):
    if e["unit"] == "flag":
        return 0.0, 1.0
    if "min" in e and "max" in e:
        return e["min"], e["max"]
    v = e["value"]
    if e["unit"] == "frac":
        lo, hi = max(0.04, v - 0.05 * wide), v + 0.05 * wide
    else:
        d = 10.0 * wide if e["unit"] == "deg" else max(0.2 * wide * abs(v), 0.3)
        lo, hi = v - d, v + d
    if key in LIMITS:
        l0, l1 = LIMITS[key]
        lo, hi = max(lo, l0), min(hi, l1)
        if lo >= hi:
            lo, hi = l0, l1
    return lo, hi

def cmd_init(a):
    for d in DIRS:
        (HOME / d).mkdir(parents=True, exist_ok=True)
    if not PARAMS.exists():
        jsave(PARAMS, {k: {"value": v, "unit": u, "evidence_class": "unknown", "source": None,
                           "source_view": None, "confidence": 0.0, "fixed": False,
                           "notes": "PLACEHOLDER - replace with drawing value"}
                       for k, (v, u) in KEYS.items()})
    jsave(VIEWS_F, jload(VIEWS_F, {}))
    st = jload(STATE, {"current": 0}) or {"current": 0}
    st.setdefault("tess_scale", 1.0)
    jsave(STATE, st)
    print(f"Project ready at {HOME}")

def cmd_migrate(a):
    P = jload(PARAMS)
    if P is None:
        sys.exit("no params.json yet; run `init` first")
    added, kept = [], []
    for k, (v, u) in KEYS.items():
        if k not in P:
            P[k] = {"value": v, "unit": u, "evidence_class": "unknown", "source": None,
                    "source_view": None, "confidence": 0.0, "fixed": False,
                    "notes": "added by migrate - adjust before relying on it"}
            added.append(k)
        else:
            kept.append(k)
    jsave(PARAMS, P)
    print(f"migrate: {len(added)} key(s) added, {len(kept)} already present.")
    for k in added:
        print(f"  + {k} = {P[k]['value']} {P[k]['unit']}")


def _vsp_roots(extra=None):
    here = Path(__file__).resolve().parent
    roots = [extra, os.environ.get("VSP_PYTHON")]
    roots += [str(p) for p in here.glob("OpenVSP*")]
    roots += [str(Path.home() / "Documents" / "diseño" / "OpenVSPx")]
    return [Path(r) for r in roots if r and Path(r).exists()]

def find_pyds(extra=None):
    out = []
    for r in _vsp_roots(extra):
        out += [Path(p) for p in glob.glob(str(r / "**" / "_vsp*.pyd"), recursive=True)]
    return out

def linked_python_dll(pyd):
    m = set(re.findall(rb"python3\d+\.dll", Path(pyd).read_bytes(), re.I))
    return sorted(x.decode() for x in m)

def import_vsp(extra=None):
    for r in _vsp_roots(extra):
        for d in [r, r / "python", r / "python" / "openvsp"]:
            if d.exists():
                if str(d) not in sys.path: sys.path.insert(0, str(d))
        for d in [r, r / "python", r / "python" / "openvsp", r / "python" / "openvsp" / "openvsp"]:
            if d.exists() and hasattr(os, "add_dll_directory"):
                try: os.add_dll_directory(str(d))
                except OSError: pass
    import openvsp as vsp
    return vsp

def cmd_check_env(a):
    print("Python", platform.python_version(), platform.architecture()[0])
    for p in find_pyds(a.vsp_path):
        print("Found", p, "-> built against:", linked_python_dll(p) or "unknown")
    try:
        vsp = import_vsp(a.vsp_path)
        print("OpenVSP OK, version:", vsp.GetVSPVersion())
    except Exception as e:
        print("FAILED to import openvsp:", e)
        sys.exit(1)

def _iter_geom_types(v):
    for fn in ("GetGeomTypes", "GetGeomTypeNames", "GetAllGeomTypes", "GetGeomTypeList"):
        try:
            res = getattr(v, fn)()
            try: res = list(res)
            except TypeError: res = [res]
            names = []
            for x in res:
                if isinstance(x, (bytes, bytearray)):
                    x = x.decode(errors="ignore")
                if isinstance(x, str):
                    names.append(x)
                else:
                    try: names.append(str(v.GetGeomTypeName(int(x))))
                    except Exception: pass
            if names:
                return sorted(set(names)), f"native ({fn})"
        except Exception:
            continue
    return [], "unavailable"

def cmd_check_var_comp(a):
    A = OpenVSPAdapter(); v = A.v
    names, how = _iter_geom_types(v)
    if not names:
        sys.exit("no geom types discovered via native API")
    print(f"OpenVSP {v.GetVSPVersion()} -> {len(names)} geometry type(s) found via {how}:")
    print("-" * 72)
    for nm in names:
        try:
            v.ClearVSPModel()
            gid = v.AddGeom(nm)
            if gid:
                dsg = [v.GetParmName(p) for p in v.GetGeomParmIDs(gid)
                       if v.GetParmGroupName(p) == "Design"]
                hint = ", ".join(dsg[:6]) if dsg else "(no Design params exposed)"
            else:
                hint = "(AddGeom returned empty id)"
        except Exception as e:
            hint = f"(error: {e})"
        print(f"  {nm:22s}  Design: {hint}")
    try: v.ClearVSPModel()
    except Exception: pass
    print("-" * 72)

def wing_at(P, y):
    """Leading-edge x, chord, mid-plane z and thickness of the main wing at lateral position y."""
    st = P.get("_wing_stations")
    if st:
        a = min(abs(y), st[-1]["y"]); ys = [s["y"] for s in st]
        ch = float(np.interp(a, ys, [s["chord"] for s in st]))
        return (float(np.interp(a, ys, [s["x_le"] for s in st])), ch,
                float(np.interp(a, ys, [s["z"] for s in st])), float(np.interp(a, ys, [s["thickness"] for s in st])) * ch)
    half = P["wing.span"] / 2.0; t = min(abs(y) / half, 1.0)
    chord = P["wing.root_chord"] + (P["wing.tip_chord"] - P["wing.root_chord"]) * t
    xle = P["wing.x"] + abs(y) * np.tan(np.radians(P["wing.sweep"]))
    zc = P["wing.z"] + abs(y) * np.tan(np.radians(P.get("wing.dihedral", 0.0)))
    return xle, chord, zc, P.get("wing.thickness", 0.12) * chord

def belly_z(P, x):
    st = P.get("_stations")
    if st:
        return float(np.interp(x, [s["x"] for s in st], [s["z_bottom"] for s in st]))
    return -float(P.get("fuselage.max_height", 5.0)) / 2.0


def pylon_dims(P):
    """Pylon between the nacelle top and the wing underside: (x0, chord, span, t/c, location) or None."""
    L, D = P["nacelle.length"], P["nacelle.diameter"]
    x, y, z = P["nacelle.x"], abs(P["nacelle.y"]), P["nacelle.z"]
    xle, ch, zc, th = wing_at(P, y)
    z_top = z + D / 2.0
    h = (zc - 0.5 * th) - (z_top - 0.2)
    if h <= 0.15:
        return None
    x0 = x + 0.12 * L
    chord = max(xle + 0.40 * ch - x0, 1.0)
    return x0, chord, h + 0.1, min(0.65 / chord, 0.2), (x0, y, z_top - 0.2)

def fair_dims(P, it):
    """Length, width, height and location of one flap-track fairing."""
    y = abs(it["y"]); wd = it["width"]; h = it.get("height", wd)
    xle, ch, zc, th = wing_at(P, y)
    x0 = it["x"]
    L = min(it["length"], xle + ch - x0)               # never past the trailing edge of the wing at that span
    if L < 0.5:                                        # too short to be a fairing: discarded
        return None
    z = it["z"] if "z" in it else (zc - 0.5 * th) - 0.45 * h
    return L, wd, h, (x0, y, z)

def fair_items(P):
    """[(item, dims)] of the fairings that survive fair_dims (the same list is used to build and to update)."""
    out = []
    for it in P.get("_fairings") or []:
        d = fair_dims(P, it)
        if d is not None:
            out.append((it, d))
    return out


class OpenVSPAdapter:
    _last_report = None
    _wing_mode = "multi"            # "multi": one WING geom with N sections | "segments": one WING geom per panel
    _wing_force_segments = bool(os.environ.get("AIRCRAFT_WING_SEGMENTS"))

    def __init__(self, vsp_path=None):
        self.v = import_vsp(vsp_path)
        self._pc = {}

    def _geom_params(self, gid):
        """Parameters of a geometry, cached for the lifetime of one build (GetGeomParmIDs is slow)."""
        if gid in self._pc:
            return self._pc[gid]
        v = self.v
        out = {}
        try:
            for pid in v.GetGeomParmIDs(gid):
                out[(v.GetParmName(pid), v.GetParmGroupName(pid))] = pid
        except Exception:
            pass
        self._pc[gid] = out
        return out

    def _set_safe(self, gid, name, group, value):
        pid = self._geom_params(gid).get((name, group))
        if pid:
            try:
                self.v.SetParmVal(pid, float(value)); return True
            except Exception:
                return False
        return False

    def _set_first(self, gid, group, candidates, value):
        params = self._geom_params(gid)
        for nm in candidates:
            pid = params.get((nm, group))
            if pid:
                try:
                    self.v.SetParmVal(pid, float(value)); return nm
                except Exception:
                    pass
        return None

    def _kill_symmetry(self, gid):
        self._set_safe(gid, "Sym_Ancestor", "Sym", 0.0)
        self._set_safe(gid, "Sym_Ancestor_Origin_Flag", "Sym", 0.0)
        self._set_safe(gid, "Sym_Planar_Flag", "Sym", 0.0)
        self._set_safe(gid, "Sym_Axial_Flag", "Sym", 0.0)

    def _bbox(self, gid):
        params = self._geom_params(gid); out = {}
        for k in ("X_Len", "Y_Len", "Z_Len"):
            pid = params.get((k, "BBox"))
            if pid:
                try: out[k] = self.v.GetParmVal(pid)
                except Exception: pass
        return out

    def geom_report(self):
        v = self.v
        return [(v.GetGeomName(g), self._bbox(g)) for g in v.FindGeoms()]

    _fuse_sig = None
    _fuse_id = None
    _sess = None
    _no_session = False        # True: every evaluation rebuilds the model (set automatically if updates are not effective)

    @staticmethod
    def _struct_sig(P, fsig):
        prop = (float(P.get("model.has_propulsion", 0.0)) >= 0.5 and "nacelle.x" in P
                and P.get("nacelle.length", 0) >= 0.1 and P.get("nacelle.diameter", 0) >= 0.05)
        return (fsig, bool(prop), bool(prop and pylon_dims(P)), len(fair_items(P)),
                len(P.get("_wing_stations") or []), OpenVSPAdapter._wing_mode)

    def build(self, P):
        v = self.v
        self._pc = {}; self._ids = {}
        s = _get_tess_scale()
        fw = int(round((TESS_W + 8) * s)); fu = int(round(TESS_U * s))
        rep_ = (round(s, 3), _FAST)
        if OpenVSPAdapter._last_report != rep_:               # report only when the mesh quality changes
            print(f"[build] TESS_SCALE={s:.2f}{' (fast, evaluation only)' if _FAST else ''}  "
                  f"fuselage Tess_W={fw} Tess_U={fu}")
            OpenVSPAdapter._last_report = rep_
        sig = None
        if _FAST and P.get("_stations") and not os.environ.get("AIRCRAFT_NO_SESSION"):                      # during optimisation the fuselage never changes
            sig = (round(float(P["fuselage.length"]), 4), round(s, 3),
                   tuple((round(t["x"], 3), round(t["width"], 3), round(t["height"], 3), round(t["z_center"], 3))
                         for t in P["_stations"]),
                   bool(P.get("_fuse_c2", False)), P.get("_fuse_tang", "off"), P.get("_fuse_ends", "point"))
        f = None
        if sig is not None and OpenVSPAdapter._fuse_sig == sig and OpenVSPAdapter._fuse_id:
            try:
                ids = list(v.FindGeoms())
                if OpenVSPAdapter._fuse_id in ids and v.GetGeomName(OpenVSPAdapter._fuse_id) == "Fuselage":
                    sess = OpenVSPAdapter._sess
                    if sess and sess["sig"] == self._struct_sig(P, sig):
                        flat = [g for x in sess["ids"].values() for g in (x if isinstance(x, list) else [x])]
                        if all(g in ids for g in flat) and not OpenVSPAdapter._no_session:   # only parameters change
                            self._ids = sess["ids"]
                            try:
                                self._update(P)
                                return
                            except Exception as e:                 # not effective in this OpenVSP: rebuild every time
                                OpenVSPAdapter._no_session = True
                                print(f"  note: in-place update failed ({e}); full rebuilds from now on")
                                self._ids = {}
                    for g in ids:
                        if g != OpenVSPAdapter._fuse_id:
                            v.DeleteGeom(g)
                    f = OpenVSPAdapter._fuse_id
            except Exception:
                f = None
        if f is None:
            v.ClearVSPModel()
            OpenVSPAdapter._fuse_sig = None; OpenVSPAdapter._fuse_id = None; OpenVSPAdapter._sess = None
            f = v.AddGeom("FUSELAGE"); v.SetGeomName(f, "Fuselage")
            self._tess(f, w=TESS_W + 8, u=TESS_U)
            v.SetParmVal(f, "Length", "Design", P["fuselage.length"])
            if P.get("_stations"):
                self._fuselage_stations(f, P["fuselage.length"], P["_stations"], bool(P.get("_fuse_c2", False)),
                                        P.get("_fuse_tang", "off"), P.get("_fuse_ends", "point"))
            else:
                surf = v.GetXSecSurf(f, 0); n = v.GetNumXSec(surf)
                for i in range(1, n - 1):
                    v.ChangeXSecShape(surf, i, v.XS_ELLIPSE)
                    xs = v.GetXSec(surf, i)
                    k = FUSE_PROFILE[min(i - 1, len(FUSE_PROFILE) - 1)]
                    v.SetParmVal(v.GetXSecParm(xs, "Ellipse_Width"), k * P["fuselage.max_width"])
                    v.SetParmVal(v.GetXSecParm(xs, "Ellipse_Height"), k * P["fuselage.max_height"])
            if sig is not None:
                OpenVSPAdapter._fuse_sig = sig; OpenVSPAdapter._fuse_id = f
        self._ids["Wing"] = self._wing("Wing", "wing", P, False)
        self._ids["HTail"] = self._wing("HTail", "htail", P, False)
        self._ids["VTail"] = self._wing("VTail", "vtail", P, True)
        self._verify_wings(P)
        extra = self._add_propulsion_and_gear(P)
        v.Update()
        ng = len(v.FindGeoms())
        expected = 4 + extra + len(self._ids.get("WingSegs", []))
        if ng != expected:
            raise RuntimeError(f"expected {expected} geometries in the model, found {ng}")
        L = float(P["fuselage.length"]); span = max(float(P["wing.span"]), float(P["htail.span"]))
        for name, bb in ([] if _FAST else self.geom_report()):     # (evaluate() has its own size check while optimising)
            if bb.get("X_Len", 0) > 2.2 * L or bb.get("Y_Len", 0) > 2.2 * span or bb.get("Z_Len", 0) > 2.2 * L:
                raise RuntimeError(f"geometry '{name}' is oversized: bbox={ {k: round(x, 1) for k, x in bb.items()} } "
                                   f"(fuselage.length={L}, span={span}). Run `inspect`.")
        OpenVSPAdapter._sess = {"sig": self._struct_sig(P, sig), "ids": self._ids} if sig is not None else None

    def _update(self, P):
        """Same model structure as the previous evaluation: only set the changed parameters."""
        v = self.v; ids = self._ids
        for key, pre, vert in (("Wing", "wing", False), ("HTail", "htail", False), ("VTail", "vtail", True)):
            if pre == "wing" and P.get("_wing_stations"):
                if OpenVSPAdapter._wing_mode == "segments":
                    self._seg_apply([ids["Wing"]] + list(ids.get("WingSegs", [])), P)
                else:
                    self._wing_apply(ids["Wing"], P)
                continue
            self._wing_params(ids[key], pre, P, vert)
            self._wing_thickness(ids[key], pre, P)
        if "nacelle" in ids:
            self._body_update(ids["nacelle"], P["nacelle.length"], NAC_PROF, P["nacelle.diameter"],
                              (P["nacelle.x"], abs(P["nacelle.y"]), P["nacelle.z"]))
        if "pylon" in ids:
            pd = pylon_dims(P)
            if pd:
                self._plate_update(ids["pylon"], pd[1], pd[2], pd[3], pd[4])
        for gid, (it, (L, wd, h, loc)) in zip(ids.get("fair", []), fair_items(P)):
            self._body_update(gid, L, FAIR_PROF, wd, loc, H=h)
        v.Update()
        got = v.GetParmVal(ids["Wing"], "X_Rel_Location", "XForm")          # read back: did the update take effect?
        exp_x = P["_wing_stations"][0]["x_le"] if P.get("_wing_stations") else P["wing.x"]
        if abs(got - exp_x) > 1e-6:
            raise RuntimeError(f"wing.x reads back {got:.4f}, expected {exp_x:.4f}")
        self._verify_wings(P)

    def _body_update(self, f, L, prof, D, loc, H=None):
        v = self.v
        L = max(float(L), 0.1); D = max(float(D), 0.05); H = D if H is None else max(float(H), 0.05)
        v.SetParmVal(f, "Length", "Design", L)
        for ax, val in zip("XYZ", loc):
            v.SetParmVal(f, f"{ax}_Rel_Location", "XForm", float(val))
        surf = v.GetXSecSurf(f, 0)
        for i, (fx, fd) in enumerate(prof):
            if 0 < i < len(prof) - 1:
                xs = v.GetXSec(surf, i)
                v.SetParmVal(v.GetXSecParm(xs, "Ellipse_Width"), max(fd * D, 0.05))
                v.SetParmVal(v.GetXSecParm(xs, "Ellipse_Height"), max(fd * H, 0.05))

    def _plate_update(self, w, chord, span, tc, loc):
        v = self.v
        for nm, val in (("Span", span), ("Root_Chord", chord), ("Tip_Chord", chord)):
            self._wset(w, nm, val)
        for ax, val in zip("XYZ", loc):
            v.SetParmVal(w, f"{ax}_Rel_Location", "XForm", float(val))
        if getattr(self, "thickness_ok", True):
            surf = v.GetXSecSurf(w, 0)
            for i in (0, 1):
                pid = self._curve_parm(v.GetXSec(surf, i), ("ThickChord", "Thick_Chord", "ThickChordRatio"))
                if pid:
                    v.SetParmVal(pid, float(tc))

    _tess_missing = set()

    def _tess(self, g, w=None, u=None):
        if getattr(self, "tess_ok", True) is False:
            return
        sc = _get_tess_scale(); v = self.v
        for nm, val, lo in (("Tess_W", w, 12), ("Tess_U", u, 3)):
            if val is None or nm in OpenVSPAdapter._tess_missing:
                continue
            try:
                pid = v.FindParm(g, nm, "Shape")
                if pid and v.ValidParm(pid):
                    v.SetParmVal(pid, float(max(int(round(val * sc)), lo)))
                else:
                    OpenVSPAdapter._tess_missing.add(nm)
            except Exception:
                OpenVSPAdapter._tess_missing.add(nm)

    def _fuselage_stations(self, f, L, st, c2=False, tang="off", ends="point"):
        v = self.v
        surf = v.GetXSecSurf(f, 0)
        N = len(st)
        while v.GetNumXSec(surf) < N:
            v.InsertXSec(f, 1, v.XS_ELLIPSE)
        while v.GetNumXSec(surf) > N:
            v.CutXSec(f, 1)
        for i in range(N):
            is_end = i in (0, N - 1)                          # XS_POINT only where the profile really closes (diameter 0)
            use_point = is_end and ends == "point" and (st[i]["width"] + st[i]["height"]) < 1e-6
            v.ChangeXSecShape(surf, i, v.XS_POINT if use_point else v.XS_ELLIPSE)
        xsecs = [v.GetXSec(surf, i) for i in range(N)]        # ids are stable once all shapes are set
        for i, s_ in enumerate(st):
            xs = xsecs[i]
            if v.GetXSecShape(xs) != v.XS_POINT:
                tiny = 0.02
                v.SetParmVal(v.GetXSecParm(xs, "Ellipse_Width"), max(s_["width"], tiny))
                v.SetParmVal(v.GetXSecParm(xs, "Ellipse_Height"), max(s_["height"], tiny))
            v.SetParmVal(v.GetXSecParm(xs, "ZLocPercent"), s_["z_center"] / L)
        # Section positions: OpenVSP keeps every section between its neighbours, so moving a section past the
        # *current* position of its neighbour is clipped. Stations clustered at nose and tail need that, which
        # produced a zig-zag fuselage. Sweep both directions until every section sits on its target.
        pids = [v.GetXSecParm(xsecs[i], "XLocPercent") for i in range(N)]
        target = [s_["x"] / L for s_ in st]
        for sweep in range(12):
            order = range(1, N - 1) if sweep % 2 == 0 else range(N - 2, 0, -1)
            for i in order:
                v.SetParmVal(pids[i], target[i])
            if max(abs(v.GetParmVal(pids[i]) - target[i]) for i in range(N)) < 1e-7:
                break
        else:
            print("WARNING: fuselage section positions did not fully converge")
        if tang in ("A", "B", "C"):
            # Every section defaults to "tangent angle 0, enforced": the skin leaves each section parallel to the
            # axis, which makes a staircase wherever the outline slopes (nose, tail cone). Give each section the
            # real slope of the silhouette instead. The sign convention differs between versions, so A/B/C are
            # tried by `fuselage-test` and the best one is kept.
            x = np.array([q["x"] for q in st]); zt = np.array([q["z_top"] for q in st])
            zb = np.array([q["z_bottom"] for q in st]); hw = np.array([q["width"] for q in st]) / 2.0
            ang = lambda y: np.clip(np.degrees(np.arctan(np.gradient(y, x))), -60.0, 60.0)
            top, bot, sd = ang(zt), ang(zb), ang(hw)
            if tang == "A":   t_, b_, s_ = top, -bot, sd           # positive = outward from the axis
            elif tang == "B": t_, b_, s_ = top, bot, sd            # positive = up
            else:             t_, b_, s_ = -top, -bot, -sd         # opposite of A
            both = getattr(v, "XSEC_BOTH_SIDES", 0)
            try:
                for i in range(1, N - 1):
                    v.SetXSecTanAngles(xsecs[i], both, float(t_[i]), float(s_[i]), float(b_[i]), float(s_[i]))
            except Exception as e:
                print(f"WARNING: could not set tangent angles ({e})")
        if c2:                                                # optional: C2 skin (can oscillate with clustered sections)
            for xs in xsecs:
                try:
                    v.SetXSecContinuity(xs, 2)
                except Exception:
                    pass

    def _wset(self, w, name, val):
        """Set a wing-section parameter through the section handle (it survives airfoil-shape changes, which rename the
        'XSec_1' group); the group name is only a last resort."""
        v = self.v
        surf = v.GetXSecSurf(w, 0)
        for idx in (1, 2, 0):
            try:
                if idx >= v.GetNumXSec(surf):
                    continue
                pid = v.GetXSecParm(v.GetXSec(surf, idx), name)
                if pid and v.ValidParm(pid):
                    v.SetParmVal(pid, float(val))
                    return True
            except Exception:
                continue
        v.SetParmVal(w, name, "XSec_1", float(val))
        return False

    def _wget(self, w, name):
        v = self.v
        surf = v.GetXSecSurf(w, 0)
        for idx in (1, 2, 0):
            try:
                if idx >= v.GetNumXSec(surf):
                    continue
                pid = v.GetXSecParm(v.GetXSec(surf, idx), name)
                if pid and v.ValidParm(pid):
                    return v.GetParmVal(pid)
            except Exception:
                continue
        return None

    # ---- multi-section main wing (wing.json) -------------------------------------------------------------------
    def _sec(self, w, i, name):
        """Parameter id of section i, fetched fresh (a shape change renews the section ids)."""
        v = self.v
        try:
            surf = v.GetXSecSurf(w, 0)
            if i >= v.GetNumXSec(surf):
                return ""
            pid = v.GetXSecParm(v.GetXSec(surf, i), name)
            return pid if (pid and v.ValidParm(pid)) else ""
        except Exception:
            return ""

    def _sec_set(self, w, i, name, val):
        pid = self._sec(w, i, name)
        if not pid:
            return False
        self.v.SetParmVal(pid, float(val)); return True

    def _sec_get(self, w, i, name):
        pid = self._sec(w, i, name)
        return self.v.GetParmVal(pid) if pid else None

    @staticmethod
    def _seg_geometry(st, i, twist_mode="absolute"):
        """OpenVSP section parameters of the panel between stations i-1 and i (Span is measured along the panel).
        FIX: the default twist is now "absolute". OpenVSP does not chain section twists: incremental values -1/-2.2/-1.8/0
        measured -1/-2.19/-1.8/0 at the section tips on the real surface instead of -1/-3.2/-5/-5."""
        a, b = st[i - 1], st[i]
        dy, dz, dx = b["y"] - a["y"], b["z"] - a["z"], b["x_le"] - a["x_le"]
        panel = float(np.hypot(dy, dz))
        return {"Span": panel, "Root_Chord": a["chord"], "Tip_Chord": b["chord"],
                "Sweep": float(np.degrees(np.arctan2(dx, panel))), "Dihedral": float(np.degrees(np.arctan2(dz, dy))),
                "Twist": (b["twist"] - a["twist"]) if twist_mode == "incremental" else b["twist"]}

    def _wing_apply(self, w, P):
        """Set every section parameter of the multi-section wing (shape already fixed, driver groups already set)."""
        v = self.v; st = P["_wing_stations"]; tm = P.get("_wing_twist_mode", "absolute")
        v.SetParmVal(w, "X_Rel_Location", "XForm", st[0]["x_le"])
        v.SetParmVal(w, "Z_Rel_Location", "XForm", st[0]["z"])
        for _ in range(2):                    # a root chord follows the previous tip chord: the second pass settles it
            for i in range(1, len(st)):
                g = self._seg_geometry(st, i, tm)
                self._sec_set(w, i, "Sweep_Location", 0.0)
                for nm in ("Span", "Tip_Chord", "Root_Chord", "Sweep", "Dihedral", "Twist"):
                    self._sec_set(w, i, nm, g[nm])
                # FIX: Update after EVERY section. Without it the derived Area of the earlier sections stays stale (22.5 and 1.0
                # instead of 96.5 and 52.4 m2), TotalArea is saved wrong and OpenVSP rescales the whole wing when the .vsp3 is
                # opened (y tip 25.7 m instead of 30.2 m), although the model in memory looks right.
                v.Update()
        if getattr(self, "thickness_ok", True):
            for i in range(len(st)):
                pid = self._curve_parm(v.GetXSec(v.GetXSecSurf(w, 0), i), ("ThickChord", "Thick_Chord", "ThickChordRatio"))
                if not pid:
                    self.thickness_ok = False
                    print("WARNING: airfoil thickness parameter not found; thickness kept at default.")
                    break
                v.SetParmVal(pid, st[i]["thickness"])
                cid = self._curve_parm(v.GetXSec(v.GetXSecSurf(w, 0), i), ("Camber",))
                if cid:
                    v.SetParmVal(cid, 0.0)
        v.Update()

    def _wing_check(self, w, P):
        """Read every section back; returns the list of (section, parameter, read, expected) that did not take."""
        st = P["_wing_stations"]; tm = P.get("_wing_twist_mode", "absolute"); bad = []
        for i in range(1, len(st)):
            g = self._seg_geometry(st, i, tm)
            for nm in ("Span", "Tip_Chord", "Sweep", "Dihedral") + (("Root_Chord",) if i == 1 else ()):
                got = self._sec_get(w, i, nm)
                tol = 1e-3 * max(1.0, abs(g[nm])) + (0.02 if nm in ("Sweep", "Dihedral") else 0.0)
                if got is None or abs(got - g[nm]) > tol:
                    bad.append((i, nm, got, round(g[nm], 4)))
            area = 0.5 * (g["Root_Chord"] + g["Tip_Chord"]) * g["Span"]      # FIX: stale Area = wrong .vsp3 on reload (see _wing_apply)
            got = self._sec_get(w, i, "Area")
            if got is None or abs(got - area) > 1e-2 * max(1.0, area):
                bad.append((i, "Area", got, round(area, 4)))
        return bad

    def _wing_multi(self, w, P):
        v = self.v; N = len(P["_wing_stations"])
        surf = v.GetXSecSurf(w, 0); guard = 0
        while v.GetNumXSec(surf) < N and guard < 64:
            v.InsertXSec(w, 1, v.XS_FOUR_SERIES); guard += 1
        while v.GetNumXSec(surf) > N and guard < 128:
            v.CutXSec(w, 1); guard += 1
        for i in range(N):                    # 1) airfoil shape first: it renews the section ids and resets the parameters
            v.ChangeXSecShape(v.GetXSecSurf(w, 0), i, v.XS_FOUR_SERIES)
        v.Update()
        failed = 0
        for i in range(1, N):                 # 2) EVERY section needs its own driver group, otherwise Span/chords are derived
            try:
                v.SetDriverGroup(w, i, v.SPAN_WSECT_DRIVER, v.ROOTC_WSECT_DRIVER, v.TIPC_WSECT_DRIVER)
            except Exception:
                failed += 1
        if failed and not getattr(OpenVSPAdapter, "_drv_warned", False):
            print(f"  note: driver group rejected in {failed} section(s); checking the read-back")
            OpenVSPAdapter._drv_warned = True
        self._wing_apply(w, P)                # 3) parameters, through freshly fetched section handles

    def _seg_check(self, gids, P):
        st = P["_wing_stations"]; tm = P.get("_wing_twist_mode", "absolute"); bad = []
        for i, g in enumerate(gids, start=1):
            e = self._seg_geometry(st, i, tm)
            for nm in ("Span", "Root_Chord", "Tip_Chord"):
                got = self._wget(g, nm)
                if got is None or abs(got - e[nm]) > 1e-3 * max(1.0, abs(e[nm])):
                    bad.append((i, nm, got, round(e[nm], 4)))
        return bad

    def _seg_apply(self, gids, P):
        v = self.v; st = P["_wing_stations"]; tm = P.get("_wing_twist_mode", "absolute")
        for i, g in enumerate(gids, start=1):
            a, b = st[i - 1], st[i]; e = self._seg_geometry(st, i, tm)
            v.SetParmVal(g, "X_Rel_Location", "XForm", a["x_le"])
            v.SetParmVal(g, "Y_Rel_Location", "XForm", a["y"])
            v.SetParmVal(g, "Z_Rel_Location", "XForm", a["z"])
            self._wset(g, "Sweep_Location", 0.0)
            for nm in ("Span", "Root_Chord", "Tip_Chord", "Sweep", "Dihedral", "Twist"):
                self._wset(g, nm, e[nm])
            if getattr(self, "thickness_ok", True):
                for k, t in ((0, a["thickness"]), (1, b["thickness"])):
                    pid = self._curve_parm(v.GetXSec(v.GetXSecSurf(g, 0), k), ("ThickChord", "Thick_Chord", "ThickChordRatio"))
                    if pid:
                        v.SetParmVal(pid, t)
        v.Update()

    def _wing_segments(self, P):
        """Fallback: one 2-section WING per panel (the same construction as the tail planes), joined edge to edge."""
        v = self.v; gids = []
        for i in range(1, len(P["_wing_stations"])):
            g = v.AddGeom("WING"); v.SetGeomName(g, "Wing" if i == 1 else f"Wing_s{i}")
            self._tess(g, w=TESS_W)
            v.SetParmVal(g, "Sym_Planar_Flag", "Sym", v.SYM_XZ)
            v.SetDriverGroup(g, 1, v.SPAN_WSECT_DRIVER, v.ROOTC_WSECT_DRIVER, v.TIPC_WSECT_DRIVER)
            for k in (0, 1):
                v.ChangeXSecShape(v.GetXSecSurf(g, 0), k, v.XS_FOUR_SERIES)
            gids.append(g)
        self._seg_apply(gids, P)
        return gids

    def _wing_stations_build(self, name, P):
        v = self.v
        self._ids["WingSegs"] = []
        err = "forced by AIRCRAFT_WING_SEGMENTS" if OpenVSPAdapter._wing_force_segments else None
        if err is None:
            w = v.AddGeom("WING"); v.SetGeomName(w, name)
            self._tess(w, w=TESS_W)
            v.SetParmVal(w, "Sym_Planar_Flag", "Sym", v.SYM_XZ)
            try:
                self._wing_multi(w, P)
                bad = self._wing_check(w, P)
                if bad:
                    err = f"{len(bad)} section parameter(s) not applied, e.g. {bad[:3]}"
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            if err is None:
                OpenVSPAdapter._wing_mode = "multi"
                return w
            try:
                v.DeleteGeom(w)
            except Exception:
                pass
        if not getattr(OpenVSPAdapter, "_seg_warned", False):
            print(f"  note: multi-section wing not applied ({err}); building the wing as separate panels instead")
            OpenVSPAdapter._seg_warned = True
        OpenVSPAdapter._wing_force_segments = True
        gids = self._wing_segments(P)
        bad = self._seg_check(gids, P)
        if bad:
            raise RuntimeError(f"Wing panels were not applied (read-back {bad[:3]}); run `wing-check` and send the output")
        OpenVSPAdapter._wing_mode = "segments"
        self._ids["WingSegs"] = gids[1:]
        return gids[0]

    def _wing(self, name, pre, P, vertical):
        v = self.v
        if pre == "wing" and P.get("_wing_stations"):
            return self._wing_stations_build(name, P)
        w = v.AddGeom("WING"); v.SetGeomName(w, name)
        self._tess(w, w=TESS_W)
        v.SetParmVal(w, "Sym_Planar_Flag", "Sym", 0 if vertical else v.SYM_XZ)
        if vertical:
            v.SetParmVal(w, "X_Rel_Rotation", "XForm", 90.0)
        v.SetDriverGroup(w, 1, v.SPAN_WSECT_DRIVER, v.ROOTC_WSECT_DRIVER, v.TIPC_WSECT_DRIVER)
        thick = pre + ".thickness" in P and getattr(self, "thickness_ok", True)
        if thick:                                              # airfoil shape first: it resets the section parameters
            surf = v.GetXSecSurf(w, 0)
            for i in (0, 1):
                v.ChangeXSecShape(surf, i, v.XS_FOUR_SERIES)
        self._wing_params(w, pre, P, vertical)                 # ...then the parameters, through the section handles
        if thick:
            self._wing_thickness(w, pre, P)
        return w

    def _wing_span(self, pre, P, vertical):
        dih = 0.0 if vertical else float(P.get(pre + ".dihedral", 0.0))
        return P[pre + ".height"] if vertical else P[pre + ".span"] / 2.0 / np.cos(np.radians(dih))

    def _wing_params(self, w, pre, P, vertical):
        v = self.v
        v.SetParmVal(w, "X_Rel_Location", "XForm", P[pre + ".x"])
        v.SetParmVal(w, "Z_Rel_Location", "XForm", P[pre + ".z"])
        items = [("Span", self._wing_span(pre, P, vertical)), ("Root_Chord", P[pre + ".root_chord"]),
                 ("Tip_Chord", P[pre + ".tip_chord"]), ("Sweep_Location", 0.0), ("Sweep", P[pre + ".sweep"])]
        if pre + ".dihedral" in P:
            items.append(("Dihedral", P[pre + ".dihedral"]))
        for nm, val in items:
            self._wset(w, nm, val)
        return w

    def _wing_thickness(self, w, pre, P):
        v = self.v
        if pre + ".thickness" in P and getattr(self, "thickness_ok", True):
            surf = v.GetXSecSurf(w, 0)
            for i in (0, 1):
                xs = v.GetXSec(surf, i)
                pid = self._curve_parm(xs, ("ThickChord", "Thick_Chord", "ThickChordRatio"))
                if not pid:
                    self.thickness_ok = False
                    print("WARNING: airfoil thickness parameter not found; thickness kept at default.")
                    break
                v.SetParmVal(pid, P[pre + ".thickness"])
                cid = self._curve_parm(xs, ("Camber",))
                if cid:
                    v.SetParmVal(cid, 0.0)

    def _verify_wings(self, P):
        """A wing whose parameters were silently ignored stays at OpenVSP's default size: fail loudly instead."""
        for key, pre, vert in (("Wing", "wing", False), ("HTail", "htail", False), ("VTail", "vtail", True)):
            if pre == "wing" and P.get("_wing_stations"):
                if OpenVSPAdapter._wing_mode == "segments":
                    bad = self._seg_check([self._ids["Wing"]] + list(self._ids.get("WingSegs", [])), P)
                else:
                    bad = self._wing_check(self._ids["Wing"], P)
                if bad:
                    raise RuntimeError(f"Wing: sections were not applied (read-back {bad[:3]}); run `wing-check`")
                continue
            exp = self._wing_span(pre, P, vert)
            got = self._wget(self._ids[key], "Span")
            if got is None or abs(got - exp) > 1e-3 * max(1.0, abs(exp)):
                raise RuntimeError(f"{key}: parameters were not applied (Span reads {got}, expected {exp:.3f}); "
                                   f"run `probe` and send the output")

    # ---- components made only with calls already proven on this machine (fuselage-type bodies and wing-type plates)
    def _place(self, g, loc, rot=(0.0, 0.0, 0.0), mirror=False):
        v = self.v
        for ax, val in zip("XYZ", loc):
            v.SetParmVal(g, f"{ax}_Rel_Location", "XForm", float(val))
        for ax, val in zip("XYZ", rot):
            if val:
                v.SetParmVal(g, f"{ax}_Rel_Rotation", "XForm", float(val))
        if mirror:
            v.SetParmVal(g, "Sym_Planar_Flag", "Sym", v.SYM_XZ)

    def _body(self, name, L, prof, D, loc, rot=(0.0, 0.0, 0.0), mirror=False, H=None):
        v = self.v
        L = max(float(L), 0.1); D = max(float(D), 0.05); H = D if H is None else max(float(H), 0.05)
        f = v.AddGeom("FUSELAGE"); v.SetGeomName(f, name); self._made += 1
        self._tess(f, w=TESS_W, u=TESS_U)
        v.SetParmVal(f, "Length", "Design", L)
        self._place(f, loc, rot, mirror)
        st = [{"x": fx * L, "width": fd * D, "height": fd * H, "z_center": 0.0} for fx, fd in prof]
        self._fuselage_stations(f, L, st)
        return f

    def _thickness(self, w, tc):
        if not getattr(self, "thickness_ok", True):
            return
        v = self.v; surf = v.GetXSecSurf(w, 0)
        for i in (0, 1):
            v.ChangeXSecShape(surf, i, v.XS_FOUR_SERIES)
        for i in (0, 1):
            xs = v.GetXSec(surf, i)
            pid = self._curve_parm(xs, ("ThickChord", "Thick_Chord", "ThickChordRatio"))
            if not pid:
                self.thickness_ok = False
                return
            v.SetParmVal(pid, float(tc))
            cid = self._curve_parm(xs, ("Camber",))
            if cid:
                v.SetParmVal(cid, 0.0)

    def _plate(self, name, chord, span, tc, loc, rot=(0.0, 0.0, 0.0), mirror=False):
        v = self.v
        w = v.AddGeom("WING"); v.SetGeomName(w, name); self._made += 1
        self._tess(w, w=TESS_W)
        self._place(w, loc, rot, mirror)
        v.SetDriverGroup(w, 1, v.SPAN_WSECT_DRIVER, v.ROOTC_WSECT_DRIVER, v.TIPC_WSECT_DRIVER)
        surf = v.GetXSecSurf(w, 0)
        if getattr(self, "thickness_ok", True):
            for i in (0, 1):
                if i < v.GetNumXSec(surf):
                    v.ChangeXSecShape(surf, i, v.XS_FOUR_SERIES)        # shape first, parameters afterwards
        for nm, val in (("Span", span), ("Root_Chord", chord), ("Tip_Chord", chord),
                        ("Sweep_Location", 0.0), ("Sweep", 0.0)):
            self._wset(w, nm, val)
        if getattr(self, "thickness_ok", True):
            for i in (0, 1):
                if i < v.GetNumXSec(surf):
                    pid = self._curve_parm(v.GetXSec(surf, i), ("ThickChord", "Thick_Chord", "ThickChordRatio"))
                    if pid:
                        v.SetParmVal(pid, float(tc))
                        cid = self._curve_parm(v.GetXSec(surf, i), ("Camber",))
                        if cid:
                            v.SetParmVal(cid, 0.0)
        return w

    def _propulsion(self, P):
        L, D = P["nacelle.length"], P["nacelle.diameter"]
        x, y, z = P["nacelle.x"], abs(P["nacelle.y"]), P["nacelle.z"]
        self._ids["nacelle"] = self._body("Nacelle", L, NAC_PROF, D, (x, y, z), mirror=True)
        if not _FAST and P.get("engine.length", 0) >= 0.1 and P.get("engine.diameter", 0) >= 0.05:
            self._body("Engine", P["engine.length"], ENG_PROF, P["engine.diameter"],
                       (P["engine.x"], abs(P["engine.y"]), P["engine.z"]), mirror=True)
        pd = pylon_dims(P)                                    # vertical plate from nacelle top to the wing underside
        if pd:
            self._ids["pylon"] = self._plate("Pylon", pd[1], pd[2], pd[3], pd[4], rot=(90.0, 0.0, 0.0), mirror=True)

    def _fairings(self, P):
        self._ids["fair"] = []
        for i, (it, (L, wd, h, loc)) in enumerate(fair_items(P)):
            self._ids["fair"].append(self._body(f"TrackFairing_{i + 1}", L, FAIR_PROF, wd, loc, mirror=True, H=h))

    def _gear(self, P):
        """Landing-gear skeleton: struts, bogie beam, wheels and door panels, attached to the wing/belly."""
        R, track, gx = P["gear.wheel_radius"], P["gear.track"], P["gear.x"]
        nx = P.get("gear.nose_x", 6.67)
        z_ground = P["vtail.z"] + P["vtail.height"] - P.get("gear.height_ref", 16.66)
        axle = z_ground + R; y = track / 2.0
        xle, ch, zc, th = wing_at(P, y)
        z_att = zc - 0.5 * th + 0.1
        Ls = z_att - axle
        if Ls < 0.3:
            print(f"  note: main gear skipped (attach {z_att:.2f} m is not above the axle {axle:.2f} m; check gear.* / vtail.*)")
        else:
            tw = max(0.7 * R, 0.3)
            self._body("MainStrut", Ls, STRUT_PROF, 0.30, (gx, y, z_att), rot=(0.0, 90.0, 0.0), mirror=True)
            self._body("MainBogie", 1.5, STRUT_PROF, 0.22, (gx - 0.75, y, axle), mirror=True)
            for dx in (-0.7, 0.7):
                for dy in (-0.55, 0.55):
                    self._body("MainWheel", tw, WHEEL_PROF, 2 * R, (gx + dx, y + dy - tw / 2, axle),
                               rot=(0.0, 0.0, 90.0), mirror=True)
        Rn = 0.8 * R; axle_n = z_ground + Rn; zb = belly_z(P, nx) + 0.2
        Ln = zb - axle_n
        if Ln > 0.3:
            twn = max(0.7 * Rn, 0.25)
            self._body("NoseStrut", Ln, STRUT_PROF, 0.26, (nx, 0.0, zb), rot=(0.0, 90.0, 0.0))
            for dy in (-0.32, 0.32):
                self._body("NoseWheel", twn, WHEEL_PROF, 2 * Rn, (nx, dy - twn / 2, axle_n), rot=(0.0, 0.0, 90.0))
        ang = float(P.get("gear.door_angle", 65.0))
        self._plate("MainDoor", 3.4, 1.1, 0.02, (gx - 1.7, 1.5, belly_z(P, gx) + 0.05), rot=(-ang, 0.0, 0.0), mirror=True)
        self._plate("NoseDoor", 1.8, 0.6, 0.02, (nx - 0.9, 0.45, belly_z(P, nx) + 0.05), rot=(-ang, 0.0, 0.0), mirror=True)

    def _subsurface(self, geom, kind, name, want):
        """Add a sub-surface and set whichever of its parameters this OpenVSP version exposes (found by name)."""
        v = self.v
        try:
            before = set(v.GetGeomParmIDs(geom)); sid = v.AddSubSurf(geom, kind, 0)
            new = [p for p in v.GetGeomParmIDs(geom) if p not in before]
        except Exception as e:
            if not getattr(OpenVSPAdapter, "_ss_warned", False):
                print(f"  note: sub-surfaces unavailable ({e})"); OpenVSPAdapter._ss_warned = True
            return 0
        try:
            v.SetSubSurfName(geom, sid, name)
        except Exception:
            pass
        n = 0
        for pid in new:
            key = v.GetParmName(pid).lower().replace("_", "")
            if key in want:
                try:
                    v.SetParmVal(pid, float(want[key])); n += 1
                except Exception:
                    pass
        return n

    @staticmethod
    def _y_to_u(st, y, mode):
        """Sub-surface U of a spanwise position.
        FIX: U is not proportional to span. Measured with CompPnt01 on OpenVSP 3.53.1: each panel takes one U unit whatever
        its span, and the root and tip caps take one unit each, so U = (1 + panel + fraction) / (panels + 2). Both former
        modes ignored the caps (y=3.4 m fell inside the root cap). `mode` is kept for old JSON files and no longer matters."""
        y = min(max(y, 0.0), wing_half(st)); panels = len(st) - 1
        for i in range(panels):
            if st[i]["y"] <= y <= st[i + 1]["y"] and st[i + 1]["y"] > st[i]["y"]:
                return (1 + i + (y - st[i]["y"]) / (st[i + 1]["y"] - st[i]["y"])) / (panels + 2)
        return (1 + panels) / (panels + 2)

    def _wing_controls(self, P, ids, ctrl):
        """Control surfaces of the multi-section wing, defined by span position in metres (wing.json "controls")."""
        st = P["_wing_stations"]; half = wing_half(st); specs = P.get("_wing_ctrl")
        if specs:
            items = [(c["name"], float(c["y0"]), float(c["y1"]), float(c.get("chord_frac", 0.25)), bool(c.get("le", False)))
                     for c in specs if c.get("enabled", True)]
        else:
            items = [(n, a * half, b * half, cf, le) for n, a, b, cf, le in WING_CONTROLS]
        made = nset = 0; mode = P.get("_wing_umode", "span")
        seg = OpenVSPAdapter._wing_mode == "segments"
        names = ["Wing"] + [f"Wing_s{i}" for i in range(2, len(st))]
        for name, y0, y1, cf, le in items:
            want = {"lengthcstart": cf, "lengthcend": cf, "leflag": 1.0 if le else 0.0}
            if not seg:
                if "Wing" in ids:
                    made += 1
                    nset += self._subsurface(ids["Wing"], ctrl, name, dict(want, ustart=self._y_to_u(st, y0, mode),
                                                                           uend=self._y_to_u(st, y1, mode)))
                continue
            for i in range(1, len(st)):                      # separate panels: clip the surface to every panel it crosses
                ya, yb = st[i - 1]["y"], st[i]["y"]
                lo, hi = max(y0, ya), min(y1, yb)
                if hi - lo < 0.05 or names[i - 1] not in ids:
                    continue
                made += 1
                nset += self._subsurface(ids[names[i - 1]], ctrl, name if (lo, hi) == (y0, y1) else f"{name}_{i}",
                                         dict(want, ustart=(lo - ya) / (yb - ya), uend=(hi - ya) / (yb - ya)))
        return made, nset

    def _details(self, P):
        """Control surfaces (flaps, slats, ailerons, elevator, rudder) and cockpit/passenger windows as sub-surfaces."""
        v = self.v
        ids = {v.GetGeomName(g): g for g in v.FindGeoms()}
        ctrl = getattr(v, "SS_CONTROL", None); rect = getattr(v, "SS_RECTANGLE", None)
        n_ctrl = n_win = n_made = 0
        if ctrl is not None:
            for gname, specs in (("Wing", WING_CONTROLS), ("HTail", TAIL_CONTROLS), ("VTail", RUDDER)):
                if gname == "Wing" and P.get("_wing_stations"):
                    m_, c_ = self._wing_controls(P, ids, ctrl)
                    n_made += m_; n_ctrl += c_
                    continue
                for name, u0, u1, cf, le in specs:
                    if gname in ids:
                        n_made += 1
                        # FIX: on these one-panel surfaces U = (1 + span fraction) / 3, the caps take one unit each (see _y_to_u).
                        # Used directly, 0.08-0.96 fell inside the caps and elevator and rudder covered the whole span.
                        n_ctrl += self._subsurface(ids[gname], ctrl, name, {
                            "ustart": (1 + u0) / 3, "uend": (1 + u1) / 3, "lengthcstart": cf, "lengthcend": cf, "leflag": 1.0 if le else 0.0})
        if rect is not None and "Fuselage" in ids:
            L = P["fuselage.length"]; w = P.get("window.w_side", 0.25)
            spots = [(1.9, 0.7), (2.8, 0.7)]                                   # cockpit side windows
            cnt = int(P.get("window.count", 0))
            if cnt > 0:
                x0, x1 = P.get("window.x0", 9.0), P.get("window.x1", 46.0)
                spots += [(x0 + (x1 - x0) * i / max(cnt - 1, 1), 0.45) for i in range(cnt)]
            for i, (xm, ln) in enumerate(spots):
                for side, ww in (("R", w), ("L", 1.0 - w)):
                    n_made += 1
                    n_win += self._subsurface(ids["Fuselage"], rect, f"Window_{side}{i + 1}",
                                              {"centeru": xm / L, "centerw": ww, "ulength": ln / L, "wlength": 0.035})
        if not getattr(OpenVSPAdapter, "_details_reported", False):
            print(f"  [details] {n_made} sub-surfaces added; parameters recognised: control {n_ctrl}, windows {n_win}"
                  f"{'' if (n_ctrl or n_win or not n_made) else '  (none: names differ in this OpenVSP; see `probe-subsurf`)'}")
            OpenVSPAdapter._details_reported = True

    def _add_propulsion_and_gear(self, P):
        self._made = 0
        def run(label, fn):
            try:
                fn(P)
            except Exception as e:
                print(f"WARNING: {label} build failed: {e}")
        if P.get("_fairings"):
            run("track fairings", self._fairings)
        if float(P.get("model.has_propulsion", 0.0)) >= 0.5 and "nacelle.x" in P \
                and P.get("nacelle.length", 0) >= 0.1 and P.get("nacelle.diameter", 0) >= 0.05:
            run("nacelle", self._propulsion)
        if float(P.get("model.has_gear", 0.0)) >= 0.5 and not _FAST and "gear.x" in P:
            run("gear", self._gear)
        if float(P.get("model.has_details", 0.0)) >= 0.5 and not _FAST:
            run("details", self._details)
        return self._made

    def geom_extents(self):
        """Real extent of every component, measured on its own exported mesh (independent of BBox parameters)."""
        v = self.v; out = {}; SETI = 3
        ids = list(v.FindGeoms())
        for g in ids:
            v.SetSetFlag(g, SETI, False)
        for g in ids:
            v.SetSetFlag(g, SETI, True); v.Update()
            t = self._tmp("one.stl"); t.unlink(missing_ok=True)
            self._drop_mesh(v.ExportFile(str(t), SETI, v.EXPORT_STL))   # FIX: see _drop_mesh (every component measured as the first one)
            if t.exists() and t.stat().st_size > 0:
                pts = load_stl(t).reshape(-1, 3)
                out[v.GetGeomName(g)] = (pts.min(0), pts.max(0))
            v.SetSetFlag(g, SETI, False)
        return out

    def _curve_parm(self, xsec, names):
        for nm in names:
            try:
                pid = self.v.GetXSecParm(xsec, nm)
                if pid and self.v.ValidParm(pid):
                    return pid
            except Exception:
                continue
        return ""

    def _tmp(self, name):
        d = Path(tempfile.gettempdir()) / "vsp_recon"
        d.mkdir(exist_ok=True)
        return d / name

    def _drop_mesh(self, mesh_id):
        """FIX: in OpenVSP 3.53.1 ExportFile leaves a MeshGeom in the model (and returns its id). While it exists, later
        exports of a set containing it write that OLD mesh and ignore parameter changes (measured: +6 m of span, same
        exported y), so in-place updates looked ineffective and geom_extents repeated the first component."""
        if mesh_id:
            try:
                self.v.DeleteGeom(mesh_id)
            except Exception:
                pass

    def export_stl(self, path):
        self.v.Update()
        t = self._tmp("export.stl")
        t.unlink(missing_ok=True)
        self._drop_mesh(self.v.ExportFile(str(t), self.v.SET_ALL, self.v.EXPORT_STL))   # FIX: see _drop_mesh (stale exports)
        if not t.exists() or t.stat().st_size == 0:
            raise RuntimeError(f"STL export failed ({t})")
        shutil.copy(t, path)
        return load_stl(t)

    def save(self, path):
        t = self._tmp("model.vsp3")
        t.unlink(missing_ok=True)
        self.v.WriteVSPFile(str(t), self.v.SET_ALL)
        if not t.exists() or t.stat().st_size < 500:
            raise RuntimeError(f".vsp3 not written correctly ({t})")
        shutil.copy(t, path)


def load_stl(path):
    b = Path(path).read_bytes()
    if b[:5].lower() == b"solid" and b"facet" in b[:1000]:
        a = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", b)
        return np.array(a, dtype=float).reshape(-1, 3, 3)
    n = int(np.frombuffer(b[80:84], "<u4")[0])
    d = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return d["v"].astype(float).reshape(-1, 3, 3)

def _map(a, r0, r1, ppm, mode):
    if mode == "min":
        return r0 + (a - a.min()) * ppm
    return (r0 + r1) / 2.0 + (a - (a.min() + a.max()) / 2.0) * ppm

def model_mask(tris, view, cal, shape):
    import cv2
    ia, ib, s = VIEWS[view]
    u, v = tris[:, :, ia], s * tris[:, :, ib]
    x0, x1, y0, y1 = cal["bbox"]
    px = _map(u, x0, x1, cal["ppm"], ALIGN[view][0])
    py = _map(v, y0, y1, cal["ppm"], ALIGN[view][1])
    pts = np.round(np.stack([px, py], -1)).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    for t in pts:                                             # one at a time: overlapping surfaces must not cancel
        cv2.fillConvexPoly(m, t, 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ext = (float(u.max() - u.min()), float(v.max() - v.min()))
    return m, ext

def imread(path, flags=0):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)

def imwrite(path, img):
    import cv2
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    data, path = buf.tobytes(), Path(path)
    for attempt in range(8):
        try:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data); os.replace(tmp, path); return
        except OSError:
            time.sleep(0.4 * (attempt + 1))
    alt = path.with_name(path.stem + "_new" + path.suffix)
    alt.write_bytes(data)
    print(f"WARNING: {path.name} is locked; wrote {alt.name} instead.")


def extract_mask(img_path, flip=False, keep_length=None, keep_min_m2=0.15):
    """Silhouette from a drawing. With keep_length (the real length of the main outline) detached parts near the
    aircraft (landing-gear wheels...) are kept too, so they can be selected with erase-mask / restore-mask."""
    import cv2
    g = imread(img_path, 0)
    if g is None:
        raise FileNotFoundError(img_path)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    ff = 255 - b
    cv2.floodFill(ff, np.zeros((b.shape[0] + 2, b.shape[1] + 2), np.uint8), (0, 0), 0)
    sil = cv2.bitwise_or(b, ff)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sil)
    if n > 1:
        main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        out = lab == main
        if keep_length:
            mx, my, mw, mh = [int(t) for t in st[main, :4]]
            ppm = mw / float(keep_length)
            for i in range(1, n):
                if i == main or st[i, cv2.CC_STAT_AREA] < keep_min_m2 * ppm ** 2:
                    continue
                x, y, w, h = [int(t) for t in st[i, :4]]
                cx, cy = x + w / 2.0, y + h / 2.0
                if mx - 0.01 * mw <= cx <= mx + 1.01 * mw and my - 0.4 * mh <= cy <= my + 1.4 * mh:
                    out = out | (lab == i)
        sil = (out * 255).astype(np.uint8)
    return cv2.flip(sil, 1) if flip else sil

def metrics(ref, mod, ppm):
    import cv2
    r, m = ref > 0, mod > 0
    union = (r | m).sum()
    iou = float((r & m).sum() / union) if union else 0.0
    edge = lambda x: cv2.morphologyEx(x.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    er, em = edge(r), edge(m)
    if not em.any() or not er.any():
        return {"iou": iou, "chamfer_m": 1e3, "hausdorff95_m": 1e3, "area_err_pct": 100.0}
    dr = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    dm = cv2.distanceTransform((~em).astype(np.uint8), cv2.DIST_L2, 3)
    a, b = dr[em], dm[er]
    return {"iou": iou, "chamfer_m": float((a.mean() + b.mean()) / 2 / ppm),
            "hausdorff95_m": float(max(np.percentile(a, 95), np.percentile(b, 95)) / ppm),
            "area_err_pct": float(100.0 * (m.sum() - r.sum()) / max(r.sum(), 1))}

def load_refs(scale=1.0):
    import cv2
    views = jload(VIEWS_F, {})
    out = {}
    for v, c in views.items():
        m = imread(c["mask"], 0)
        if scale != 1.0:                                      # same silhouette at lower resolution: much faster metrics
            m = ((cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) > 127) * 255).astype(np.uint8)
            c = dict(c); c["ppm"] = c["ppm"] * scale; c["bbox"] = [int(round(b * scale)) for b in c["bbox"]]
        out[v] = (c, m)
    if not out:
        sys.exit("No calibrated views. Run: calibrate --view top --image ... --length ...")
    return out

def evaluate(tris, refs, weights, Lref):
    per, total = {}, 0.0
    for v, (c, ref) in refs.items():
        if weights.get(v, 1.0) == 0:
            continue
        m, ext = model_mask(tris, v, c, ref.shape)
        mt = metrics(ref, m, c["ppm"])
        x0, x1, y0, y1 = c["bbox"]
        ru, rv = (x1 - x0 + 1) / c["ppm"], (y1 - y0 + 1) / c["ppm"]
        if ext[0] > 1.6 * ru or ext[1] > 1.8 * rv:
            raise RuntimeError(f"the model is {ext[0]:.0f} x {ext[1]:.0f} m in the {v} view but the drawing is "
                               f"{ru:.0f} x {rv:.0f} m: a component is far outside the aircraft. Run `inspect`.")
        mt["extent_err_m"] = [ext[0] - ru, ext[1] - rv]
        per[v] = mt
        total += weights.get(v, 1.0) * ((1.0 - mt["iou"]) + mt["chamfer_m"] / Lref)
    return total, per

def calibration_problems(P=None, V=None):
    """Views whose --length disagrees with the aircraft (top/side: fuselage length, front: wing span)."""
    P = P or jload(PARAMS) or {}; V = V if V is not None else jload(VIEWS_F, {})
    if "fuselage.length" not in P or "wing.span" not in P:
        return []
    key_of = {"top": "fuselage.length", "side": "fuselage.length", "front": "wing.span"}
    exp = {"top": P["fuselage.length"]["value"], "side": P["fuselage.length"]["value"] * 54.08 / 53.85,
           "front": P["wing.span"]["value"]}
    out = []
    for v, c in V.items():
        L = c.get("length_m")
        if P[key_of.get(v, "fuselage.length")].get("evidence_class") == "unknown":
            continue                                              # placeholder parameters (fresh `init`): nothing to compare with
        if L and v in exp and abs(L - exp[v]) / exp[v] > 0.06:
            out.append((v, L, round(exp[v], 2)))
    return out

def require_calibration():
    bad = calibration_problems()
    if bad:
        lines = [f"  {v}: calibrated with --length {L} but the aircraft needs about {e}  ->  "
                 f"python aircraft_recon.py rescale --view {v} --length {e}" for v, L, e in bad]
        sys.exit("WRONG CALIBRATION (pixels per metre are off, so every comparison is meaningless):\n" + "\n".join(lines))

def cmd_rescale(a):
    """Fix the scale of an already calibrated view without touching its mask."""
    V = jload(VIEWS_F, {})
    if a.view not in V: sys.exit(f"view {a.view} is not calibrated")
    c = V[a.view]; x0, x1 = c["bbox"][0], c["bbox"][1]
    old = c["ppm"]
    c["length_m"] = a.length; c["ppm"] = (x1 - x0 + 1) / a.length
    jsave(VIEWS_F, V)
    print(f"{a.view}: {old:.2f} -> {c['ppm']:.2f} px/m (length {a.length} m)")


def _sync_dims_from_view(view, length):
    """Fresh `init` parameters are placeholders: the first calibration of a view gives the real size (any aircraft)."""
    P = jload(PARAMS)
    key = {"top": "fuselage.length", "front": "wing.span"}.get(view)
    if P and key in P and P[key].get("evidence_class") == "unknown":
        P[key].update({"value": float(length), "evidence_class": "calculated", "confidence": 0.6,
                       "source": f"calibration of the {view} view", "notes": "set from the calibration length"})
        jsave(PARAMS, P)
        print(f"  {key} was a placeholder: set to {length} m from this calibration (edit params.json to refine/fix it)")

def cmd_calibrate(a):
    import cv2
    HOME.mkdir(exist_ok=True)
    for d in DIRS: (HOME / d).mkdir(exist_ok=True)
    src = a.image
    mask = imread(a.mask, 0) if a.mask else extract_mask(src, a.flip, a.length if a.keep_parts else None, a.keep_min_area)
    if a.mask and a.flip: mask = cv2.flip(mask, 1)
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    ppm = (x1 - x0 + 1) / a.length
    mp = HOME / "processed" / f"{a.view}_mask.png"
    imwrite(mp, mask)
    V = jload(VIEWS_F, {})
    V[a.view] = {"image": str(src), "mask": str(mp), "ppm": ppm, "bbox": [x0, x1, y0, y1],
                 "flip": a.flip, "length_m": a.length, "unit": "m", "confidence": None}
    jsave(VIEWS_F, V)
    print(f"{a.view}: {ppm:.2f} px/m, bbox {x0, x1, y0, y1}. CHECK processed/{a.view}_mask.png.")
    _sync_dims_from_view(a.view, a.length)
    for v_, L_, e_ in calibration_problems(None, {a.view: V[a.view]}):
        print(f"  WARNING: --length {L_} looks wrong for the {v_} view (expected about {e_} m: "
              f"{'wing span' if v_ == 'front' else 'overall length'}). Fix: rescale --view {v_} --length {e_}")


def _clean_landmark_mask(img_path, supplied_mask=None, flip=False):
    import cv2
    if supplied_mask:
        m = imread(supplied_mask, 0)
        if m is None:
            raise FileNotFoundError(supplied_mask)
        m = (m > 0).astype(np.uint8) * 255
    else:
        g = imread(img_path, 0)
        if g is None:
            raise FileNotFoundError(img_path)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        hlen = max(25, m.shape[1] // 18); vlen = max(25, m.shape[0] // 18)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))
        m = cv2.subtract(m, cv2.morphologyEx(m, cv2.MORPH_OPEN, hk))
        m = cv2.subtract(m, cv2.morphologyEx(m, cv2.MORPH_OPEN, vk))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    if flip:
        m = cv2.flip(m, 1)
    return m


def _largest_relevant_component(mask):
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask.copy(), 0.0
    areas = st[1:, cv2.CC_STAT_AREA].astype(float)
    scores = []
    for i, area in enumerate(areas, start=1):
        x, y, w, h = st[i, :4]
        scores.append(area * (1.0 + min(w / max(h, 1), 10.0) * 0.05))
    idx = 1 + int(np.argmax(scores))
    out = ((lab == idx) * 255).astype(np.uint8)
    return out, float(st[idx, cv2.CC_STAT_AREA] / mask.size)


def _contour_points(mask):
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no aircraft contour found")
    c = max(contours, key=cv2.contourArea)
    pts = c[:, 0, :].astype(float)
    if len(pts) < 20:
        raise RuntimeError("aircraft contour has too few points")
    return pts


def _point(name, xy, confidence, method, note=""):
    return {"name": name, "pixel": [float(xy[0]), float(xy[1])],
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "method": method, "note": note}


def _detect_top(mask):
    pts = _contour_points(mask)
    xlo, xhi = np.percentile(pts[:, 0], [1, 99]); L = max(xhi - xlo, 1.0)
    xnorm = (pts[:, 0] - xlo) / L; out = {}
    out["nose_tip"] = _point("nose_tip", pts[np.argmin(pts[:, 0])], 0.98, "contour_longitudinal_extreme")
    out["aftmost_outline"] = _point("aftmost_outline", pts[np.argmax(pts[:, 0])], 0.96, "contour_longitudinal_extreme")
    mid = pts[(xnorm > 0.22) & (xnorm < 0.72)]
    if len(mid):
        out["main_wing_tip_left_candidate"] = _point("main_wing_tip_left_candidate", mid[np.argmin(mid[:, 1])], 0.78, "midbody_lateral_extreme", "")
        out["main_wing_tip_right_candidate"] = _point("main_wing_tip_right_candidate", mid[np.argmax(mid[:, 1])], 0.78, "midbody_lateral_extreme", "")
    aft = pts[(xnorm > 0.68) & (xnorm < 0.96)]
    if len(aft):
        out["horizontal_tail_tip_left_candidate"] = _point("horizontal_tail_tip_left_candidate", aft[np.argmin(aft[:, 1])], 0.70, "aftzone_lateral_extreme", "")
        out["horizontal_tail_tip_right_candidate"] = _point("horizontal_tail_tip_right_candidate", aft[np.argmax(aft[:, 1])], 0.70, "aftzone_lateral_extreme", "")
    central = pts[(xnorm > 0.10) & (xnorm < 0.90)]
    if len(central):
        out["fuselage_centerline_y_candidate"] = _point("fuselage_centerline_y_candidate",
                                                        [(xlo + xhi) / 2.0, float(np.median(central[:, 1]))],
                                                        0.55, "contour_median", "")
    return out


def _detect_side(mask):
    pts = _contour_points(mask)
    xlo, xhi = np.percentile(pts[:, 0], [1, 99]); L = max(xhi - xlo, 1.0)
    xnorm = (pts[:, 0] - xlo) / L; out = {}
    out["nose_tip"] = _point("nose_tip", pts[np.argmin(pts[:, 0])], 0.98, "contour_longitudinal_extreme")
    out["aftmost_outline"] = _point("aftmost_outline", pts[np.argmax(pts[:, 0])], 0.96, "contour_longitudinal_extreme")
    aft = pts[(xnorm > 0.68) & (xnorm < 0.98)]
    if len(aft):
        out["vertical_tail_tip_candidate"] = _point("vertical_tail_tip_candidate", aft[np.argmin(aft[:, 1])], 0.72, "aftzone_vertical_extreme", "")
    mid = pts[(xnorm > 0.20) & (xnorm < 0.80)]
    if len(mid):
        out["upper_envelope_candidate"] = _point("upper_envelope_candidate", mid[np.argmin(mid[:, 1])], 0.62, "midbody_vertical_extreme", "")
        out["lower_envelope_candidate"] = _point("lower_envelope_candidate", mid[np.argmax(mid[:, 1])], 0.62, "midbody_vertical_extreme", "")
    return out


def _detect_front(mask):
    pts = _contour_points(mask)
    ylo, yhi = np.percentile(pts[:, 1], [1, 99]); H = max(yhi - ylo, 1.0)
    ynorm = (pts[:, 1] - ylo) / H; out = {}
    out["leftmost_outline"] = _point("leftmost_outline", pts[np.argmin(pts[:, 0])], 0.96, "contour_lateral_extreme")
    out["rightmost_outline"] = _point("rightmost_outline", pts[np.argmax(pts[:, 0])], 0.96, "contour_lateral_extreme")
    upper = pts[ynorm < 0.35]
    if len(upper):
        out["vertical_tail_tip_candidate"] = _point("vertical_tail_tip_candidate", upper[np.argmin(upper[:, 1])], 0.72, "upper_vertical_extreme", "")
    middle = pts[(ynorm > 0.30) & (ynorm < 0.80)]
    if len(middle):
        out["wing_tip_left_candidate"] = _point("wing_tip_left_candidate", middle[np.argmin(middle[:, 0])], 0.68, "middleheight_lateral_extreme", "")
        out["wing_tip_right_candidate"] = _point("wing_tip_right_candidate", middle[np.argmax(middle[:, 0])], 0.68, "middleheight_lateral_extreme", "")
    out["aircraft_centerline_x_candidate"] = _point("aircraft_centerline_x_candidate",
                                                    [float(np.median(pts[:, 0])), (ylo + yhi) / 2.0],
                                                    0.55, "contour_median", "")
    return out


def _landmark_physical_coordinates(points, view):
    cal = jload(VIEWS_F, {}).get(view)
    if not cal or "ppm" not in cal or "bbox" not in cal:
        return points
    ppm = float(cal["ppm"]); x0, x1, y0, y1 = cal["bbox"]
    out = dict(points)
    for _, p in out.items():
        px, py = p["pixel"]
        p["plane_m"] = {"u": float((px - x0) / ppm), "v_image_down": float((py - y0) / ppm)}
        p["calibration_length_m"] = cal.get("length_m")
    return out


def cmd_landmarks(a):
    import cv2
    mask = _clean_landmark_mask(a.image, a.mask, a.flip)
    clean, area_frac = _largest_relevant_component(mask)
    if area_frac < 0.0005:
        raise SystemExit("Automatic landmark tracing found no reliable component.")
    pts = _detect_top(clean) if a.view == "top" else _detect_side(clean) if a.view == "side" else _detect_front(clean)
    pts = _landmark_physical_coordinates(pts, a.view)
    out_dir = HOME / "processed"; out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{a.view}_landmarks.json"
    jsave(json_path, {"view": a.view, "image": str(a.image), "mask": str(a.mask) if a.mask else None,
                      "flip": bool(a.flip), "method": "automatic_contour_landmarks_v1",
                      "warning": "Candidates require geometric verification.",
                      "component_area_fraction": area_frac, "landmarks": pts})
    src = imread(a.image, 1)
    if src is None:
        raise FileNotFoundError(a.image)
    if a.flip:
        src = cv2.flip(src, 1)
    for name, p in pts.items():
        x, y = map(int, np.round(p["pixel"]))
        cv2.circle(src, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(src, f"{name} {p['confidence']:.2f}", (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    imwrite(out_dir / f"{a.view}_landmarks.png", src)
    print(json.dumps({"view": a.view, "landmarks": len(pts), "json": str(json_path),
                      "component_area_fraction": round(area_frac, 6)}, indent=2))

def cmd_erase_mask(a):
    import cv2
    V = jload(VIEWS_F, {})
    if a.view not in V:
        sys.exit(f"view {a.view} is not calibrated yet")
    c = V[a.view]
    mask = imread(c["mask"], 0)
    full = Path(c["mask"]).with_name(f"{a.view}_mask_full.png")
    if not full.exists():
        imwrite(full, mask)
    rects = [tuple(r) for r in (a.rect or [])]
    if not rects:
        z = a.zoom
        small = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
        print("Draw a rectangle around each part to remove, ENTER/SPACE to accept it, ESC when finished.")
        sel = cv2.selectROIs(f"erase {a.view}", small, showCrosshair=False)
        cv2.destroyAllWindows()
        rects = [(int(x / z), int(y / z), int((x + w) / z) + 1, int((y + h) / z) + 1) for x, y, w, h in sel]
    for x0, y0, x1, y1 in rects:
        mask[y0:y1, x0:x1] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = ((lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) * 255).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    c["bbox"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
    imwrite(c["mask"], mask); jsave(VIEWS_F, V)
    print(f"{a.view}: erased {len(rects)} region(s); bbox now {c['bbox']}; original kept as {full.name}")


def versions():
    return sorted(int(re.findall(r"v(\d+)\.json", p.name)[0]) for p in (HOME / "versions").glob("v*.json"))

def commit(P, command, changes, adapter=None, reason=None):
    A = adapter or OpenVSPAdapter()
    A.build(values(P))
    n = (versions() or [0])[-1] + 1
    A.save(HOME / "versions" / f"v{n:03d}.vsp3")
    st = jload(STATE, {"current": 0}) or {"current": 0}
    jsave(HOME / "versions" / f"v{n:03d}.json",
          {"version": n, "parent": st.get("current") or None, "command": command, "reason": reason,
           "changes": changes, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "params": P, "fuselage": jload(FUSE_F), "fairings": jload(FAIR_F), "wing": jload(WING_F)})
    st["current"] = n
    jsave(STATE, st); jsave(PARAMS, P)
    print(f"-> version {n}")
    return n

def change(P, key, new, command, protect=True):
    e = P[key]; lo, hi = bounds(e)
    if protect and (e.get("fixed") or e.get("evidence_class") == "measured"):
        raise SystemExit(f"{key} is measured/fixed; use --force to override")
    if e["unit"] == "flag":
        new = float(int(round(min(max(new, lo), hi))))
    else:
        new = float(min(max(new, lo), hi)) if "min" in e else float(new)
    ch = {"parameter": key, "old": e["value"], "new": new, "unit": e["unit"]}
    e["value"] = new
    return ch

def cmd_set(a, delta=False):
    P = jload(PARAMS)
    if P is None:
        sys.exit("no params.json; run `init` first")
    if a.key not in P:
        if a.key in KEYS:
            v, u = KEYS[a.key]
            P[a.key] = {"value": v, "unit": u, "evidence_class": "unknown", "source": None,
                        "source_view": None, "confidence": 0.0, "fixed": False,
                        "notes": "added on first set"}
            print(f"  (key {a.key} was missing; added with default {v} {u})")
        else:
            sys.exit(f"unknown parameter {a.key}")
    cur = P[a.key]["value"]
    ch = change(P, a.key, cur + a.value if delta else a.value,
                f"{'delta' if delta else 'set'} {a.key} {a.value}", not a.force)
    commit(P, f"{'delta' if delta else 'set'} {a.key} {a.value}", [ch])

def cmd_patch(a):
    P, chs = jload(PARAMS), []
    for p in jload(a.file):
        k = p["parameter"]
        if k not in P:
            if k in KEYS:
                v, u = KEYS[k]
                P[k] = {"value": v, "unit": u, "evidence_class": "unknown", "source": None,
                        "source_view": None, "confidence": 0.0, "fixed": False, "notes": "added on first patch"}
            else:
                sys.exit(f"unknown parameter {k}")
        new = P[k]["value"] + p["delta"] if p["operation"] == "modify" and "delta" in p else p["value"]
        chs.append(change(P, k, new, "patch", not a.force))
    commit(P, f"patch {a.file}", chs, reason="; ".join(p.get("reason", "") for p in jload(a.file)))

def cmd_diff(a):
    vs = versions()
    if not vs: sys.exit("no versions yet")
    n1 = a.a if a.a else vs[0]
    n2 = a.b if a.b else (jload(STATE, {}).get("current") or vs[-1])
    A = jload(HOME / "versions" / f"v{n1:03d}.json")["params"]
    B = jload(HOME / "versions" / f"v{n2:03d}.json")["params"]
    print(f"v{n1} -> v{n2}")
    moved = 0
    for k in B:
        if k in A and abs(B[k]["value"] - A[k]["value"]) > 1e-6:
            moved += 1
            d = B[k]["value"] - A[k]["value"]
            print(f"  {k:22s} {A[k]['value']:9.3f} -> {B[k]['value']:9.3f}  ({d:+.3f} {B[k]['unit']})")
    print(f"{moved} parameter(s) changed")

def _dp_keep(xs, ys, tol, i0, i1, keep):
    """Douglas-Peucker on the polyline (xs, ys): add the indices worth keeping to `keep`."""
    if i1 <= i0 + 1:
        return
    x0, x1 = xs[i0], xs[i1]; dx = x1 - x0
    if abs(dx) < 1e-12:
        return
    y0, y1 = ys[i0], ys[i1]
    dmax, imax = 0.0, i0
    for i in range(i0 + 1, i1):
        t = (xs[i] - x0) / dx
        d = abs(ys[i] - (y0 + t * (y1 - y0)))
        if d > dmax:
            dmax, imax = d, i
    if dmax > tol:
        keep.add(imax)
        _dp_keep(xs, ys, tol, i0, imax, keep)
        _dp_keep(xs, ys, tol, imax, i1, keep)

def simplify_stations(st, tol_frac=SIMPLIFY_TOL):
    """Drop stations that lie (within tol_frac * max dimension) on the straight line between their neighbours."""
    if len(st) <= 3 or tol_frac <= 0:
        return st
    scale = max(max(s_["width"] for s_ in st), max(s_["height"] for s_ in st), 1e-6)
    tol = tol_frac * scale
    xs = [s_["x"] for s_ in st]
    keep = {0, len(st) - 1}
    for key in ("width", "height", "z_center"):
        _dp_keep(xs, [s_[key] for s_ in st], tol, 0, len(st) - 1, keep)
    return [st[i] for i in sorted(keep)]

def cmd_simplify(a):
    """Drop fuselage stations that do not change the shape (fewer sections for VSPAERO)."""
    fs = jload(FUSE_F)
    if not fs or not fs.get("stations"):
        sys.exit("no fuselage.json with stations; run `fuselage-fit` first")
    n0 = len(fs["stations"])
    fs["stations"] = simplify_stations(fs["stations"], a.tol)
    fs.setdefault("mode", {})["simplify_tol"] = a.tol
    fs["mode"]["stations"] = len(fs["stations"])
    jsave(FUSE_F, fs)
    print(f"fuselage: {n0} -> {len(fs['stations'])} stations (tolerance {a.tol * 100:.2f} % of the diameter)")
    commit(jload(PARAMS), f"simplify fuselage {n0}->{len(fs['stations'])} stations", [],
           reason="drop redundant fuselage sections")


def fit_fuselage(P, N=31, M=400, spacing=None, c2=None, tang=None, ends=None, simplify_tol=None):
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    V = jload(VIEWS_F, {})
    if "top" not in V or "side" not in V:
        raise SystemExit("fuselage fitting needs calibrated 'top' and 'side' views")
    top, side = imread(V["top"]["mask"], 0) > 0, imread(V["side"]["mask"], 0) > 0
    L = P["fuselage.length"]["value"]
    Dw, Dh = P["fuselage.max_width"]["value"], P["fuselage.max_height"]["value"]
    x_fin = P["vtail.x"]["value"] - 0.3
    xf = np.linspace(0.0, L, M)
    tx0, tppm = V["top"]["bbox"][0], V["top"]["ppm"]
    sx0, sppm = V["side"]["bbox"][0], V["side"]["ppm"]
    w, rt, rb = np.full(M, np.nan), np.full(M, np.nan), np.full(M, np.nan)
    for i, x in enumerate(xf):
        c = min(int(round(tx0 + x * tppm)), top.shape[1] - 1)
        r = np.where(top[:, c])[0]
        if len(r) and (r.max() - r.min() + 1) / tppm <= 1.08 * Dw:
            w[i] = (r.max() - r.min() + 1) / tppm
        c = min(int(round(sx0 + x * sppm)), side.shape[1] - 1)
        r = np.where(side[:, c])[0]
        if len(r):
            rb[i] = r.max()
            if x < x_fin:
                rt[i] = r.min()
    fill = lambda a: np.interp(np.arange(M), np.where(np.isfinite(a))[0], a[np.isfinite(a)])
    rt[-1] = rb[-1] - 0.3 * sppm
    rt, rb = fill(rt), fill(rb)
    win = (M // 12) | 1
    sm = lambda a: savgol_filter(median_filter(a, 23, mode="nearest"), win, 3, mode="interp")
    rt, rb = sm(rt), sm(rb)
    h = (rb - rt) / sppm
    clean = np.isfinite(w) & (h > 0.3)
    ratio = np.clip(np.where(clean, w / np.maximum(h, 1e-6), np.nan), 0.5, 1.5)
    ratio = fill(ratio) if clean.any() else np.ones(M)
    wf = np.where(np.isfinite(w), w, ratio * h)
    w_raw, h_raw = float(np.max(sm(wf))), float(np.max(h))
    print(f"  fuselage inputs: L={L:.2f} m, params width/height={Dw:.2f}/{Dh:.2f} m, "
          f"masks show {w_raw:.2f}/{h_raw:.2f} m (px/m top={tppm:.1f}, side={sppm:.1f})")
    if Dw < 0.7 * w_raw or Dh < 0.7 * h_raw:
        sys.exit(f"fuselage.max_width/max_height in params.json ({Dw:.2f}/{Dh:.2f}) are far from what the masks show "
                 f"({w_raw:.2f}/{h_raw:.2f}). Fix them (`set fuselage.max_height 5.64 --force`) or check the calibration.")
    if Dh > 1.5 * h_raw or Dw > 1.5 * w_raw:
        sys.exit(f"the masks give a much smaller fuselage ({w_raw:.2f}/{h_raw:.2f} m) than params.json "
                 f"({Dw:.2f}/{Dh:.2f} m): wrong calibration (--length) or a damaged mask.")
    wf = np.minimum(sm(wf), Dw)
    imax = int(np.argmax(wf))                                 # behind the widest point the width only decreases
    wf[imax:] = np.minimum.accumulate(wf[imax:])
    h = np.minimum(h, Dh)                                     # engines/belly fairing below the cylinder are capped
    rb = rt + h * sppm
    ref = np.where(h >= 0.97 * h.max())[0]
    row_c = np.mean((rt[ref] + rb[ref]) / 2)
    prev = (jload(FUSE_F) or {}).get("mode") or {}
    spacing = spacing or prev.get("spacing", "uniform")       # uniform + no C2 is the combination known to work
    c2 = prev.get("c2", False) if c2 is None else bool(c2)
    tang = tang or prev.get("tang", "off")
    ends = ends or prev.get("ends", "point")
    xs = np.linspace(0.0, L, N) if spacing == "uniform" else L * (1.0 - np.cos(np.pi * np.linspace(0, 1, N))) / 2.0
    pick = lambda a: np.interp(xs, xf, a)
    W, H, RT, RB = pick(wf), pick(h), pick(rt), pick(rb)
    W[0] = W[-1] = 0.0
    st = [{"x": float(x), "width": float(a_), "height": float(h_),
           "z_center": float((row_c - (t + b) / 2) / sppm),
           "z_top": float((row_c - t) / sppm), "z_bottom": float((row_c - b) / sppm)}
          for x, a_, h_, t, b in zip(xs, W, H, RT, RB)]
    n0 = len(st)
    tol = SIMPLIFY_TOL if simplify_tol is None else float(simplify_tol)
    st = simplify_stations(st, tol)
    tag = f"{n0}->{len(st)}" if len(st) < n0 else f"{len(st)}"
    jsave(FUSE_F, {"source": "silhouette masks (smoothed); ellipse sections",
                   "evidence_class": "calculated", "mode": {"spacing": spacing, "c2": c2, "tang": tang, "ends": ends,
                                                            "stations": len(st), "simplify_tol": tol},
                   "stations": st})
    print(f"  fuselage: {tag} smooth stations ({spacing} spacing, C2 {'on' if c2 else 'off'}, tangents {tang}, {ends} ends) "
          f"(max width {W.max():.2f} m, max height {H.max():.2f} m)")
    return st

def cmd_fuselage_test(a):
    """Try the fuselage construction variants and keep the one that matches the silhouettes best."""
    require_calibration()
    refs, P = load_refs(), jload(PARAMS)
    if not {"top", "side"} <= set(refs): sys.exit("needs calibrated top and side views")
    Lref = P["fuselage.length"]["value"]; W = {"top": 1.0, "side": 1.0, "front": 0.0}
    combos = [("uniform", "off"), ("uniform", "A"), ("uniform", "B"), ("uniform", "C"),
              ("cosine", "A"), ("cosine", "B"), ("cosine", "C")]
    res = []
    for spacing, tang in combos:
        fit_fuselage(P, a.stations, spacing=spacing, c2=False, tang=tang)
        with fast_tess():
            A = OpenVSPAdapter(); A.build(values(P))
            tris = A.export_stl(HOME / "renders" / "opt.stl")
        tot, per = evaluate(tris, refs, W, Lref)
        res.append((tot, spacing, tang))
        print(f"  >> {spacing:8s} tangents {tang:3s}  error={tot:.4f}  "
              f"top IoU={per['top']['iou']:.3f}  side IoU={per['side']['iou']:.3f}")
    res.sort(); tot, spacing, tang = res[0]
    print(f"best: {spacing} spacing, tangents {tang} (error {tot:.4f})")
    ends = "point"
    fit_fuselage(P, a.stations, spacing=spacing, c2=False, tang=tang, ends="round")      # nose/tail as small rounded ends
    with fast_tess():
        A = OpenVSPAdapter(); A.build(values(P)); tris = A.export_stl(HOME / "renders" / "opt.stl")
    tot_r, per_r = evaluate(tris, refs, W, Lref)
    print(f"  >> {spacing:8s} tangents {tang:3s} round ends  error={tot_r:.4f}  "
          f"top IoU={per_r['top']['iou']:.3f}  side IoU={per_r['side']['iou']:.3f}")
    if tot_r < tot * 0.999:
        ends, tot = "round", tot_r
    print(f"chosen: {spacing} spacing, tangents {tang}, {ends} ends (error {tot:.4f})")
    fit_fuselage(P, a.stations, spacing=spacing, c2=False, tang=tang, ends=ends, simplify_tol=getattr(a, "simplify_tol", None))
    commit(P, f"fuselage test -> {spacing}, tangents {tang}, {ends} ends", [], reason="best fuselage construction by silhouette error")


def simplify_stations_pchip(st, tol):
    """Fewest stations whose monotone (PCHIP) interpolation reproduces width, height and z of the dense profile
    within tol metres: stations that barely change are dropped (cylinder), nose and tail keep theirs."""
    from scipy.interpolate import PchipInterpolator
    x = np.array([t["x"] for t in st], float)
    keys = ("width", "height", "z_center", "z_top", "z_bottom")
    Y = np.array([[t[k] for t in st] for k in keys], float)
    keep = [0, len(st) - 1]
    while len(keep) < len(st):
        err = np.zeros(len(st))
        for j in range(len(keys)):
            err = np.maximum(err, np.abs(PchipInterpolator(x[keep], Y[j][keep])(x) - Y[j]))
        err[keep] = 0.0
        i = int(np.argmax(err))
        if err[i] <= tol:
            break
        keep = sorted(keep + [i])
    return [st[i] for i in keep]

def cmd_fuselage_simplify(a):
    """Reduce the number of fuselage sections (easier for aerodynamics) while the silhouettes stay the same."""
    require_calibration()
    P = jload(PARAMS); refs = load_refs(EVAL_DS)
    W = {v: 1.0 for v in VIEWS if v in refs}; Lref = P["fuselage.length"]["value"]
    dense = fit_fuselage(P, a.dense, simplify_tol=0)
    base = jload(FUSE_F)
    def score(stations):
        d = dict(base); d["stations"] = stations; d["mode"] = dict(base["mode"], stations=len(stations))
        jsave(FUSE_F, d)
        with fast_tess():
            A = OpenVSPAdapter(); A.build(values(P))
            return evaluate(A.export_stl(HOME / "renders" / "opt.stl"), refs, W, Lref)[0]
    tot0 = score(dense)
    print(f"  {len(dense):3d} stations (dense)       error {tot0:.4f}")
    best = (len(dense), dense, tot0)
    for tol in sorted(a.tols, reverse=True):
        red = simplify_stations_pchip(dense, tol)
        tot = score(red)
        print(f"  {len(red):3d} stations (tol {tol:5.3f} m)  error {tot:.4f}  ({tot - tot0:+.4f})")
        if tot <= tot0 + a.allow and len(red) < best[0]:
            best = (len(red), red, tot)
    d = dict(base); d["stations"] = best[1]; d["mode"] = dict(base["mode"], stations=best[0]); d["simplified"] = True
    jsave(FUSE_F, d)
    print(f"kept {best[0]} of {len(dense)} stations (error {best[2]:.4f} vs {tot0:.4f})")
    commit(P, f"fuselage simplified to {best[0]} stations", [], reason="fewer fuselage sections for the aerodynamic model")


def cmd_fuselage_fit(a):
    require_calibration()
    P = jload(PARAMS)
    fit_fuselage(P, a.stations, spacing=a.spacing, c2=None if a.c2 is None else a.c2 == "on", tang=a.tang,
                 ends=a.ends, simplify_tol=getattr(a, "simplify_tol", None))
    commit(P, "fuselage stations from silhouettes", [], reason="fuselage fitted from top/side masks")


def cmd_inspect(a):
    """Build the current parameters and report what is wrong: component sizes, key parameters, calibration."""
    P = jload(PARAMS)
    if P is None: sys.exit("no params.json")
    Q = values(P)
    with fast_tess():
        A = OpenVSPAdapter()
        try:
            A.build(Q)
        except RuntimeError as e:
            print("BUILD CHECK FAILED:", e)
        print("components (extent of each exported mesh, m):")
        try:
            for name, (lo, hi) in A.geom_extents().items():
                print(f"  {name:16s} x {lo[0]:8.2f} .. {hi[0]:8.2f}   y {lo[1]:8.2f} .. {hi[1]:8.2f}   z {lo[2]:8.2f} .. {hi[2]:8.2f}")
        except Exception as e:
            print("  (per-component export failed:", e, ") bounding boxes instead:")
            for name, bb in A.geom_report():
                print(f"  {name:12s} " + "  ".join(f"{k}={x:8.2f}" for k, x in bb.items()))
    print("key parameters:")
    for k in ("fuselage.length", "fuselage.max_width", "fuselage.max_height", "wing.span", "wing.x", "wing.z",
              "vtail.x", "vtail.z", "vtail.height", "model.has_propulsion", "nacelle.x", "nacelle.y", "nacelle.z",
              "nacelle.length", "nacelle.diameter", "gear.x", "gear.z", "gear.track"):
        if k in P:
            e = P[k]; print(f"  {k:22s} {e['value']:10.3f} {e['unit']:5s} [{e.get('evidence_class')}, fixed={e.get('fixed')}]")
    fs = jload(FUSE_F)
    if fs:
        st = fs["stations"]
        print(f"fuselage stations: {len(st)} | width max {max(s['width'] for s in st):.2f} | height max "
              f"{max(s['height'] for s in st):.2f} | z_center {min(s['z_center'] for s in st):.2f}..{max(s['z_center'] for s in st):.2f} m")
    print("views (px/m, bbox):")
    for v, c in jload(VIEWS_F, {}).items():
        print(f"  {v:5s} ppm={c['ppm']:.2f} bbox={c['bbox']} length_m={c.get('length_m')}")
    for v, L, e in calibration_problems(P):
        print(f"  !! {v}: calibrated with length {L} m, expected about {e} m -> "
              f"python aircraft_recon.py rescale --view {v} --length {e}")


def use_masks(kind):
    """Activate the mask variant 'core' (no engines/gear) or 'extras' (with engines) for every view that has it."""
    V = jload(VIEWS_F, {}); changed = []
    for view, c in V.items():
        p = Path(c["mask"]).with_name(f"{view}_mask_{kind}.png")
        if p.exists():
            m = imread(p, 0); imwrite(c["mask"], m)
            ys, xs = np.where(m > 0)
            c["bbox"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
            changed.append(view)
    jsave(VIEWS_F, V)
    return changed

def cmd_masks(a):
    ch = use_masks(a.use)
    print(f"active masks: {a.use} for {ch or 'no view (no *_mask_%s.png found)' % a.use}")

def cmd_restore_mask(a):
    """Put back parts that erase-mask removed (e.g. the engines), choosing only the rectangles you select."""
    import cv2
    V = jload(VIEWS_F, {})
    if a.view not in V: sys.exit(f"view {a.view} is not calibrated yet")
    c = V[a.view]; cur = imread(c["mask"], 0)
    full_p = Path(c["mask"]).with_name(f"{a.view}_mask_full.png")
    if not full_p.exists(): sys.exit("nothing was erased in this view (no *_mask_full.png)")
    full = imread(full_p, 0)
    core_p = Path(c["mask"]).with_name(f"{a.view}_mask_core.png")
    if not core_p.exists():
        imwrite(core_p, cur)                                  # engine-free mask, used by phase 1 of `auto`
    rects = [tuple(r) for r in (a.rect or [])]
    if not rects:
        z = a.zoom
        small = cv2.resize(cv2.cvtColor(full, cv2.COLOR_GRAY2BGR), None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
        print("Draw a rectangle around each part to put back (the engines), ENTER/SPACE to accept, ESC to finish.")
        sel = cv2.selectROIs(f"restore {a.view}", small, showCrosshair=False)
        cv2.destroyAllWindows()
        rects = [(int(x / z), int(y / z), int((x + w) / z) + 1, int((y + h) / z) + 1) for x, y, w, h in sel]
    for x0, y0, x1, y1 in rects:
        cur[y0:y1, x0:x1] = np.maximum(cur[y0:y1, x0:x1], full[y0:y1, x0:x1])
    ys, xs = np.where(cur > 0)
    c["bbox"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
    imwrite(c["mask"], cur); jsave(VIEWS_F, V)
    imwrite(Path(c["mask"]).with_name(f"{a.view}_mask_extras.png"), cur)   # mask with engines, used by phase 2
    print(f"{a.view}: restored {len(rects)} region(s); bbox {c['bbox']}  (core/extras variants saved)")


def cmd_selfcheck(a):
    """Is the fast in-place update identical to a full rebuild? (settles 'the optimiser does not move the wing')"""
    P = jload(PARAMS); Q = values(P)
    stat = lambda t: (len(t), t[:, :, 0].min(), t[:, :, 0].max(), t[:, :, 1].max(), t[:, :, 2].min(), t[:, :, 2].max(), float(t[:, :, 0].sum()))
    Q2 = dict(Q); Q2["wing.x"] = Q["wing.x"] + 2.0; Q2["wing.sweep"] = Q["wing.sweep"] + 3.0
    Q2["htail.x"] = Q["htail.x"] + 2.0      # FIX: with wing.json the two wing keys above are ignored, which reported "fast path NOT reliable" falsely
    with fast_tess():
        A = OpenVSPAdapter(); OpenVSPAdapter._no_session = False
        A.build(Q); t1 = A.export_stl(HOME / "renders" / "opt.stl")
        A.build(Q2); t2 = A.export_stl(HOME / "renders" / "opt.stl")
        used = OpenVSPAdapter._sess is not None and not OpenVSPAdapter._no_session
        OpenVSPAdapter._no_session = True
        A.build(Q2); t3 = A.export_stl(HOME / "renders" / "opt.stl")
        OpenVSPAdapter._no_session = False
    s1, s2, s3 = stat(t1), stat(t2), stat(t3)
    print("                     triangles   x min   x max   y max   z min   z max   sum x")
    for nm, st_ in (("original", s1), ("in-place update", s2), ("full rebuild", s3)):
        print(f"  {nm:16s} {st_[0]:9d} " + " ".join(f"{x:7.2f}" for x in st_[1:6]) + f" {st_[6]:10.1f}")
    moved = abs(s2[6] - s1[6]) > 1e-3
    same = s2[0] == s3[0] and all(abs(x - y) < 1e-3 for x, y in zip(s2[1:], s3[1:]))
    print(f"\n  the parameters changed the mesh: {'YES' if moved else 'NO'}   in-place == full rebuild: {'YES' if same else 'NO'}"
          f"   (in-place path used: {'yes' if used else 'no, it fell back'})")
    if moved and same:
        print("  -> the fast path is correct. If the optimiser stalls, the reason is the objective, not the update.")
    else:
        OpenVSPAdapter._no_session = True
        print("  -> the fast path is NOT reliable here: use AIRCRAFT_NO_SESSION=1 (full rebuild for every evaluation).")


def cmd_probe(a):
    A = OpenVSPAdapter(); v = A.v
    v.ClearVSPModel()
    w = v.AddGeom("WING"); surf = v.GetXSecSurf(w, 0)
    v.ChangeXSecShape(surf, 0, v.XS_FOUR_SERIES)
    print("wing xsec count:", v.GetNumXSec(surf), "| xsec 0 shape type:", v.GetXSecShape(v.GetXSec(surf, 0)))
    for pid in v.GetGeomParmIDs(w):
        nm, grp = v.GetParmName(pid), v.GetParmGroupName(pid)
        if any(t in nm.lower() for t in ("thick", "camber", "tess")):
            print(f"  group={grp!r:20} name={nm!r}")

def cmd_probe_geom(a):
    A = OpenVSPAdapter(); v = A.v
    v.ClearVSPModel()
    try:
        g = v.AddGeom(a.type.upper())
    except Exception as e:
        sys.exit(f"AddGeom({a.type!r}) failed: {e}")
    if not g:
        sys.exit(f"AddGeom({a.type!r}) returned empty id: type not available")
    print(f"geom type = {a.type.upper()}, id = {g}")
    pids = v.GetGeomParmIDs(g)
    groups = sorted({v.GetParmGroupName(pid) for pid in pids})
    print("groups:", groups)
    for pid in pids:
        nm, grp = v.GetParmName(pid), v.GetParmGroupName(pid)
        if not a.all and grp not in ("Design", "XForm", "Sym"):
            continue
        try: val = v.GetParmVal(pid)
        except Exception: val = "?"
        print(f"  [{grp:12s}] {nm:30s} = {val}")


def cmd_history(a):
    cur = jload(STATE, {}).get("current")
    vs = versions()
    if getattr(a, "last", None):
        vs = vs[-a.last:]
    for n in vs:
        v = jload(HOME / "versions" / f"v{n:03d}.json")
        print(f"{'*' if n == cur else ' '} v{n} (parent {v['parent']}) {v['command']} "
              f"{[(c['parameter'], c['old'], c['new']) for c in v['changes']]}")

def cmd_checkout(a, n=None):
    n = n or a.n
    v = jload(HOME / "versions" / f"v{n:03d}.json")
    jsave(PARAMS, v["params"])
    st = jload(STATE, {"current": 0}) or {"current": 0}
    st["current"] = n
    jsave(STATE, st)
    if v.get("fuselage"): jsave(FUSE_F, v["fuselage"])
    elif FUSE_F.exists(): FUSE_F.unlink()
    if v.get("fairings"): jsave(FAIR_F, v["fairings"])
    elif FAIR_F.exists(): FAIR_F.unlink()
    if v.get("wing"): jsave(WING_F, v["wing"])
    elif WING_F.exists(): WING_F.unlink()
    print("current version:", n)

def cmd_undo(a):
    cur = jload(STATE)["current"]
    parent = jload(HOME / "versions" / f"v{cur:03d}.json")["parent"]
    if not parent: sys.exit("nothing to undo")
    cmd_checkout(a, parent)


def cmd_compare(a):
    require_calibration()
    refs, P = load_refs(), jload(PARAMS)
    with fast_tess():                                         # same mesh quality as the optimiser -> comparable numbers
        A = OpenVSPAdapter(); A.build(values(P))
        tris = A.export_stl(HOME / "renders" / "current.stl")
    Lref = P["fuselage.length"]["value"]
    total, per = evaluate(tris, refs, DEFAULT_W, Lref)
    for v, (c, ref) in refs.items():
        m, _ = model_mask(tris, v, c, ref.shape)
        img = np.full(ref.shape + (3,), 255, np.uint8)
        r, mm = ref > 0, m > 0
        img[r & ~mm] = (0, 0, 255); img[mm & ~r] = (255, 0, 0); img[r & mm] = (180, 180, 180)
        imwrite(HOME / "renders" / f"overlay_{v}.png", img)
    out = {"total_error": total, "views": per, "tess_scale": _get_tess_scale()}
    jsave(HOME / "renders" / "metrics.json", out)
    print(json.dumps(out, indent=2))

def run_optimize(free, W, iters=80, method="Powell", allow_measured=False, wide=1.0, skip_fixed=False, commit_now=True,
                 patience=24):
    from scipy.optimize import minimize
    refs, P = load_refs(EVAL_DS), jload(PARAMS)
    free = [k for k in free if k in P]
    locked = [k for k in free if P[k].get("fixed") or P[k].get("evidence_class") == "measured"]
    if locked and not allow_measured:
        if not skip_fixed:
            sys.exit(f"{locked} measured/fixed (use --allow-measured to override)")
        free = [k for k in free if k not in locked]
    if not free:
        print("  nothing to optimise (all parameters fixed)"); return None, None, False
    Lref = P["fuselage.length"]["value"]
    A = OpenVSPAdapter()
    B = [bounds(P[k], wide, k) for k in free]
    x0 = np.clip(np.array([(P[k]["value"] - lo) / max(hi - lo, 1e-9) for k, (lo, hi) in zip(free, B)]), 0, 1)
    tmp = HOME / "renders" / "opt.stl"
    def apply(x):
        Q = values(P)
        for xi, k, (lo, hi) in zip(x, free, B):
            Q[k] = lo + float(np.clip(xi, 0, 1)) * (hi - lo)
        return Q
    class _Stop(Exception):
        pass
    state = {"best": float("inf"), "x": x0.copy(), "n": 0, "mark": 0, "ref": 1.0}
    def f(x):
        A.build(apply(x))
        val = evaluate(A.export_stl(tmp), refs, W, Lref)[0]
        state["n"] += 1
        state.setdefault("vals", []).append(val); state.setdefault("xs", set()).add(tuple(np.round(x, 6)))
        if os.environ.get("AIRCRAFT_VERBOSE"):
            print(f"    eval {state['n']:3d}: {val:.6f}")
        if state["n"] == 8 and not OpenVSPAdapter._no_session and len(state["xs"]) >= 3 \
                and max(state["vals"]) - min(state["vals"]) < 1e-12:
            OpenVSPAdapter._no_session = True; OpenVSPAdapter._sess = None
            print("  note: 8 different parameter sets gave exactly the same error -> in-place updates are not effective "
                  "here; switching to full rebuilds")
            state["mark"] = state["n"]
        if val < state["best"]:
            if val < state["best"] - 0.002 * state["ref"]:     # a real (>0.2 %) improvement resets the patience
                state["mark"] = state["n"]
            state["best"], state["x"] = val, np.array(x, float)
        if state["n"] - state["mark"] >= patience:
            raise _Stop()
        return val
    with fast_tess():                                         # coarse mesh while searching; fine mesh when saving
        A.build(values(P)); before_t, before = evaluate(A.export_stl(tmp), refs, W, Lref)
        state["best"], state["ref"] = before_t, max(before_t, 1e-9)
        try:
            res = minimize(f, x0, method=method, bounds=[(0, 1)] * len(x0),
                           options={"maxfev": iters, "xtol": 5e-3, "ftol": 1e-4} if method == "Powell"
                           else {"maxiter": iters})
        except _Stop:
            res = types.SimpleNamespace(nfev=state["n"])
        res.x = state["x"]                                    # best point seen, not just the last one
        res.nfev = state["n"]
        Q = apply(res.x)
        A.build(Q); after_t, after = evaluate(A.export_stl(tmp), refs, W, Lref)
    rec = {"iteration": len(list((HOME / "iterations").glob("*.json"))) + 1, "free": free,
           "weights": W, "before_error": before_t, "after_error": after_t, "before_views": before,
           "after_views": after, "method": method, "evals": int(res.nfev),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"  error {before_t:.5f} -> {after_t:.5f}  ({int(res.nfev)} evaluations)")
    improved = after_t < before_t * 0.999
    if improved:
        chs = []
        for k in free:
            ch = {"parameter": k, "old": P[k]["value"], "new": round(Q[k], 5), "unit": P[k]["unit"]}
            P[k]["value"] = round(Q[k], 5)
            P[k]["evidence_class"] = "inferred" if P[k].get("evidence_class") != "measured" else "measured"
            P[k]["notes"] = "optimised against silhouettes"
            chs.append(ch)
        if commit_now:
            rec["version"] = commit(P, f"optimize {' '.join(free)}", chs, A, "silhouette/chamfer optimisation")
        else:
            jsave(PARAMS, P)                                   # auto commits once per pass (saving a fine model is slow)
    else:
        print("  no reliable improvement; parameters unchanged.")
    jsave(HOME / "iterations" / f"iter{rec['iteration']:03d}.json", rec)
    return before_t, after_t, improved

def cmd_optimize(a):
    W = dict(DEFAULT_W); W.update({kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.weights})
    run_optimize(a.free, W, a.iters, a.method, a.allow_measured)

STAGES = [
    ("wing planform (top view)", ["wing.x", "wing.sweep", "wing.root_chord", "wing.tip_chord"], {"top": 1}),
    ("horizontal tail planform (top view)", ["htail.x", "htail.sweep", "htail.root_chord", "htail.tip_chord"], {"top": 1}),
    ("vertical tail (side + front)", ["vtail.x", "vtail.z", "vtail.height", "vtail.sweep", "vtail.root_chord",
                                      "vtail.tip_chord"], {"side": 1, "front": 0.3}),
    ("dihedral and heights (front view, side as support)",
     ["wing.dihedral", "wing.z", "htail.dihedral", "htail.z"], {"front": 1, "side": 0.3}),
    ("airfoil thickness (front view)", ["wing.thickness", "htail.thickness", "vtail.thickness"], {"front": 1}),
]
# Gear is not observable in the silhouettes (the wheels are detached from the outline), so it is not optimised;
# the engine core is hidden inside the nacelle and is derived from it (sync_engine).
PROP_STAGE = ("nacelles (all views)",
              ["nacelle.x", "nacelle.y", "nacelle.z", "nacelle.length", "nacelle.diameter"],
              {"top": 0.8, "front": 1.0, "side": 0.6})

def sync_engine(P):
    """Engine core inside the nacelle: shorter and thinner, same axis."""
    n = {k: P[f"nacelle.{k}"]["value"] for k in ("x", "y", "z", "length", "diameter")}
    new = {"engine.x": n["x"] + 0.18 * n["length"], "engine.y": n["y"], "engine.z": n["z"],
           "engine.length": 0.62 * n["length"], "engine.diameter": 0.55 * n["diameter"]}
    for k, val in new.items():
        if k in P:
            P[k]["value"] = round(val, 4); P[k]["evidence_class"] = "calculated"
            P[k]["notes"] = "derived from the nacelle"

def _total_error(calibrated, refs):
    P = jload(PARAMS)
    W = {v: 1.0 for v in VIEWS if v in calibrated}
    with fast_tess():
        A = OpenVSPAdapter(); A.build(values(P))
        return evaluate(A.export_stl(HOME / "renders" / "opt.stl"), refs, W, P["fuselage.length"]["value"])[0]

def _pause(a, label, snapshot=None):
    """--step: show overlays/metrics and let the user continue, undo the last step or quit."""
    if not getattr(a, "step", False):
        return "continue"
    try:
        cmd_compare(a)
    except SystemExit:
        pass
    if not sys.stdin.isatty():
        return "continue"
    while True:
        r = input(f"  >> after '{label}': overlays in {HOME / 'renders'}.  [Enter] continue | [u] undo this step | [q] quit : ").strip().lower()
        if r == "":
            return "continue"
        if r == "u" and snapshot is not None:
            jsave(PARAMS, snapshot); print("  step undone"); return "undo"
        if r == "q":
            sys.exit("stopped by the user (nothing else changed)")

def prepare_masks(a, calibrated):
    """Engines/gear selection inside auto: erase them (core masks), then put the engines back (extras masks)."""
    for v in ("top", "side", "front"):
        if v not in calibrated:
            continue
        V = jload(VIEWS_F, {}); c = V[v]; mp = Path(c["mask"])
        for kind in ("core", "extras"):
            mp.with_name(f"{v}_mask_{kind}.png").unlink(missing_ok=True)
        full = mp.with_name(f"{v}_mask_full.png")
        if full.exists():                                         # start again from the complete silhouette
            m = imread(full, 0); imwrite(mp, m); ys, xs = np.where(m > 0)
            c["bbox"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]; jsave(VIEWS_F, V)
        print(f"--- {v}: (1) rectangle around the ENGINES and the LANDING GEAR to remove; (2) then around the ENGINES to put back")
        cmd_erase_mask(types.SimpleNamespace(view=v, rect=None, zoom=a.zoom))
        cmd_restore_mask(types.SimpleNamespace(view=v, rect=None, zoom=a.zoom))

def _stages_loop(a, calibrated, stages, label):
    last = None
    for p in range(1, a.passes + 1):
        print(f"=== {label}: pass {p}/{a.passes} ===")
        if {"top", "side"} <= calibrated and label == "airframe":
            print("- fuselage from silhouettes")
            fit_fuselage(jload(PARAMS), a.stations, simplify_tol=getattr(a, "simplify_tol", SIMPLIFY_TOL))
        if label == "airframe" and "top" in calibrated and wing_json_is_auto():
            print("- wing sections from the plan silhouette")
            try:
                fit_wing_from_mask(jload(PARAMS))
            except SystemExit as e:
                print(f"  wing-fit skipped ({e}); the single-panel wing stays")
        with fast_tess():
            probe_a = OpenVSPAdapter(); probe_a.build(values(jload(PARAMS)))
        thick_ok = getattr(probe_a, "thickness_ok", True)
        for name, free, w in stages:
            if name.startswith("wing planform") and WING_F.exists():
                print(f"- {name}: skipped (wing.json defines the planform)"); continue
            if not thick_ok and any(k.endswith(".thickness") for k in free):
                print(f"- {name}: skipped (thickness parameter not available)"); continue
            W = {v: (w.get(v, 0.0) if v in calibrated else 0.0) for v in VIEWS}
            if sum(W.values()) == 0:
                print(f"- {name}: skipped (needs a calibrated view)"); continue
            print(f"- {name}")
            snap = jload(PARAMS); guard = getattr(a, "guard", True)
            refs_all = load_refs(EVAL_DS) if guard else None
            tot0 = _total_error(calibrated, refs_all) if guard else None
            run_optimize(free, W, a.iters, "Powell", False, a.wide, skip_fixed=True, commit_now=False)
            if guard:
                tot1 = _total_error(calibrated, refs_all)
                if tot1 > tot0 * 1.02:                            # a stage must not make the whole aircraft worse
                    jsave(PARAMS, snap)
                    print(f"  guard: overall error {tot0:.4f} -> {tot1:.4f} (worse): stage reverted")
                else:
                    print(f"  overall error {tot0:.4f} -> {tot1:.4f}")
            _pause(a, name, snap)
        P1 = jload(PARAMS)
        if float(P1.get("model.has_propulsion", {"value": 0.0})["value"]) >= 0.5:
            sync_engine(P1); jsave(PARAMS, P1)
        commit(jload(PARAMS), f"auto {label} pass {p}", [], reason="stages of one automatic pass")
        W = {v: 1.0 for v in VIEWS if v in calibrated}
        refs, P = load_refs(EVAL_DS), jload(PARAMS)
        with fast_tess():
            A = OpenVSPAdapter(); A.build(values(P))
            tot = evaluate(A.export_stl(HOME / "renders" / "opt.stl"), refs, W, P["fuselage.length"]["value"])[0]
        print(f"  overall error after pass {p}: {tot:.5f}")
        if last is not None and abs(last - tot) < 0.01 * last:
            print("  converged (<1% change)"); break
        last = tot

def cmd_auto(a):
    """Phases: 1 airframe (fuselage, wing, tails; masks without engines) -> 2 nacelles + pylons + fairings
    (masks with engines) -> 3 finish (gear, control surfaces, windows, fine mesh)."""
    require_calibration()
    calibrated = set(jload(VIEWS_F, {}))
    if not calibrated:
        sys.exit("no calibrated views")
    P0 = jload(PARAMS)
    added = [k for k in KEYS if k not in P0]
    for k in added:
        v, u = KEYS[k]
        P0[k] = {"value": v, "unit": u, "evidence_class": "inferred", "source": None, "source_view": None,
                 "confidence": 0.2, "fixed": False, "notes": "default estimate"}
    if added:
        jsave(PARAMS, P0); print("Added new parameters with default estimates:", added)
    print("Calibrated views:", sorted(calibrated), "| phase:", a.phase)
    have = all(Path(jload(VIEWS_F)[v]["mask"]).with_name(f"{v}_mask_core.png").exists() for v in calibrated)
    if a.phase != "finish" and (getattr(a, "masks", False) or (not have and sys.stdin.isatty())):
        if getattr(a, "masks", False) or input("No core/extras masks yet. Select engines and gear now? [Y/n] ").strip().lower() != "n":
            prepare_masks(a, calibrated)
    if a.phase in ("airframe", "all"):
        print("##### PHASE 1: airframe (fuselage, wing, tails) on masks WITHOUT engines")
        ch = use_masks("core")
        if not ch:
            print("  (no *_mask_core.png: using the current masks)")
        with airframe_only():
            _stages_loop(a, calibrated, STAGES, "airframe")
    if a.phase in ("extras", "all"):
        print("##### PHASE 2: nacelles, pylons, track fairings on masks WITH engines")
        ch = use_masks("extras")
        if not ch:
            print("  (no *_mask_extras.png: using the current masks; put the engines back with restore-mask)")
        P = jload(PARAMS)
        if float(P["model.has_propulsion"]["value"]) < 0.5 or a.refit:
            snap2 = jload(PARAMS)
            cmd_nacelle_fit(types.SimpleNamespace(no_init=False, iters=a.iters, wide=1.0))
            _pause(a, "nacelles", snap2)
        cmd_fairings_fit(types.SimpleNamespace(clear=False, max_count=8, under=1.2))
        _pause(a, "fairings")
        run_optimize(PROP_STAGE[1], PROP_STAGE[2], a.iters, "Powell", False, 1.0, skip_fixed=True, commit_now=False)
        P1 = jload(PARAMS); sync_engine(P1); jsave(PARAMS, P1)
        commit(P1, "auto phase 2: nacelles and fairings", [])
    if a.phase in ("finish", "all") and not a.no_finish:
        print("##### PHASE 3: finish (gear, control surfaces, windows, fine mesh)")
        P = jload(PARAMS)
        for k in ("model.has_gear", "model.has_details"):
            if k in P:
                P[k]["value"] = 1.0
        if P["gear.x"]["value"] < 10:
            print("  note: gear.x is still the default; set it (A300-600: `set gear.x 25.27`, `set gear.track 9.6`, "
                  "`set gear.wheel_radius 0.63`) and run `auto --phase finish`")
        jsave(PARAMS, P)
        if getattr(a, "pchip_simplify", False):
            print("- fewer fuselage sections (PCHIP search against the silhouettes; slow)")
            cmd_fuselage_simplify(types.SimpleNamespace(dense=61, tols=[0.12, 0.08, 0.05, 0.03, 0.02, 0.01], allow=0.003))
        _set_tess_scale(max(_get_tess_scale(), 1.6))
        commit(P, "auto phase 3: gear, details, fine mesh", [])
    cmd_compare(a)
    print("Done.")

def cmd_smooth(a):
    _set_tess_scale(a.scale)
    P = jload(PARAMS)
    if a.stations and {"top", "side"} <= set(jload(VIEWS_F, {})):
        fit_fuselage(P, a.stations, simplify_tol=getattr(a, "simplify_tol", SIMPLIFY_TOL))
    commit(P, f"smooth scale={a.scale} stations={a.stations}", [],
           reason="higher tessellation for a smoother aircraft surface")
    print(f"Tessellation scale persisted = {_get_tess_scale():.2f}  "
          f"(fuselage Tess_W≈{int((TESS_W+8)*_get_tess_scale())}, Tess_U≈{int(TESS_U*_get_tess_scale())}); "
          f"optimisation keeps using the coarse mesh for speed.")

def cmd_front_fit(a):
    if "front" not in jload(VIEWS_F, {}):
        sys.exit("calibrate the front view first")
    free = ["vtail.height", "vtail.z", "wing.dihedral", "wing.z", "htail.dihedral", "htail.z"]
    run_optimize(free, {"front": 1.0, "side": 0.25}, a.iters, "Powell",
                 allow_measured=False, wide=a.wide, skip_fixed=True)


def _bare_model(P, **flags):
    """Mesh of the airframe with the given model.* flags switched off (default: no propulsion, no gear)."""
    Q = values(P)
    for k in ("model.has_propulsion", "model.has_gear", "model.has_details"):
        Q[k] = flags.get(k, 0.0)
    with fast_tess():
        A = OpenVSPAdapter(); A.build(Q)
        return A.export_stl(HOME / "renders" / "opt.stl")

def residual_blobs(P, view, tris, open_m, min_m2, max_m2):
    """Connected regions of the reference silhouette that the model does not cover (metres, plane coordinates)."""
    import cv2
    c = jload(VIEWS_F, {}).get(view)
    if not c:
        return None, []
    ref = imread(c["mask"], 0) > 0
    m, _ = model_mask(tris, view, c, ref.shape)
    resid = (ref & ~(m > 0)).astype(np.uint8) * 255
    k = max(3, int(open_m * c["ppm"]))
    resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(resid)
    ppm = c["ppm"]; out = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA] / ppm ** 2
        if min_m2 <= area <= max_m2:
            bx, by, bw, bh = [int(t) for t in st[i, :4]]
            out.append({"bx": bx, "by": by, "bw": bw, "bh": bh, "area": area, "cx": cen[i][0], "cy": cen[i][1]})
    return c, out

def measure_nacelle(P):
    """Nacelle position and size from the plan view (x, length, y, diameter) and the front view (z, y, diameter)."""
    c = jload(VIEWS_F, {}).get("top")
    if not c:
        sys.exit("calibrate the top view first")
    tris = _bare_model(P)
    c, big = residual_blobs(P, "top", tris, 0.8, 1.2 ** 2, 60.0)
    if not big:
        sys.exit("no nacelle-sized blob found in the plan view (are the engines in the top mask?)")
    x0, x1, y0, y1 = c["bbox"]; row_c = (y0 + y1) / 2.0; ppm = c["ppm"]
    lower = [b for b in big if b["cy"] > row_c] or big                     # +Y side
    b = max(lower, key=lambda j: j["area"])
    out = {"nacelle.x": (b["bx"] - x0) / ppm, "nacelle.length": b["bw"] / ppm,
           "nacelle.y": (b["by"] + b["bh"] / 2.0 - row_c) / ppm, "nacelle.diameter": b["bh"] / ppm}
    cf, fb = residual_blobs(P, "front", tris, 0.6, 1.0, 80.0)
    if cf and fb:
        fx0, fx1, fy0, fy1 = cf["bbox"]; col_c = (fx0 + fx1) / 2.0; fppm = cf["ppm"]
        right = [q for q in fb if q["cx"] > col_c] or fb
        q = max(right, key=lambda j: j["area"])
        zmax = float(tris[:, :, 2].max())                                     # fin tip = top of the front view
        d_f = q["bw"] / fppm
        z_bottom = zmax - (q["by"] + q["bh"] - fy0) / fppm
        y_f = (q["bx"] + q["bw"] / 2.0 - col_c) / fppm
        out["nacelle.z"] = z_bottom + d_f / 2.0
        out["nacelle.y"] = 0.5 * (out["nacelle.y"] + abs(y_f))
        out["nacelle.diameter"] = 0.5 * (out["nacelle.diameter"] + d_f)
        out["_from_front"] = True
    return out

def cmd_nacelle_fit(a):
    require_calibration()
    P = jload(PARAMS)
    if P is None:
        sys.exit("no params.json; run `init` first")
    for k, (v, u) in KEYS.items():
        P.setdefault(k, {"value": v, "unit": u, "evidence_class": "inferred", "source": None, "source_view": None,
                         "confidence": 0.2, "fixed": False, "notes": "default estimate"})
    if not a.no_init:
        meas = measure_nacelle(P)
        from_front = meas.pop("_from_front", False)
        for k, val in meas.items():
            P[k].update({"value": round(val, 3), "evidence_class": "calculated",
                         "source_view": "top+front" if from_front else "top", "confidence": 0.75 if from_front else 0.6,
                         "notes": "measured from the plan and front masks" if from_front else "measured from the plan mask"})
        if "nacelle.z" not in meas:
            wing_dih = np.tan(np.radians(P["wing.dihedral"]["value"]))
            z0 = P["wing.z"]["value"] + abs(meas["nacelle.y"]) * wing_dih - 0.6 * meas["nacelle.diameter"]
            P["nacelle.z"].update({"value": round(z0, 3), "evidence_class": "inferred", "confidence": 0.3,
                                   "notes": "initial guess: engine axis below the wing plane"})
            meas["nacelle.z"] = z0
        print({k: round(v, 2) for k, v in meas.items()}, "(top + front views)" if from_front else "(top view only)")
    P["model.has_propulsion"]["value"] = 1.0
    sync_engine(P)
    jsave(PARAMS, P)
    commit(P, "nacelles added", [], reason="nacelle geometry from the plan-view mask")
    run_optimize(PROP_STAGE[1], PROP_STAGE[2], a.iters, "Powell",
                 allow_measured=False, wide=a.wide, skip_fixed=True)
    P = jload(PARAMS); sync_engine(P); jsave(PARAMS, P)

def cmd_fairings_fit(a):
    """Flap-track fairings: count, height and lateral position from the FRONT view; x and length from the PLAN view."""
    require_calibration()
    if a.clear:
        if FAIR_F.exists():
            FAIR_F.unlink()
        print("fairings removed"); return
    P = jload(PARAMS)
    Q = values(P); Q["_fairings"] = None; Q["model.has_gear"] = 0.0; Q["model.has_details"] = 0.0
    with fast_tess():
        A = OpenVSPAdapter(); A.build(Q)
        tris = A.export_stl(HOME / "renders" / "opt.stl")
    cf, fb = residual_blobs(P, "front", tris, 0.06, 0.04, 1.6)
    ct, tb = residual_blobs(P, "top", tris, 0.06, 0.10, 6.0)
    if not cf:
        sys.exit("calibrate the front view first")
    fx0, fx1, fy0, fy1 = cf["bbox"]; col_c = (fx0 + fx1) / 2.0; fppm = cf["ppm"]
    zmax = float(tris[:, :, 2].max()); half = Q["wing.span"] / 2.0
    top_ok = bool(ct)
    if top_ok:
        x0, x1, y0, y1 = ct["bbox"]; row_c = (y0 + y1) / 2.0; ppm = ct["ppm"]
    items = []
    for q in sorted(fb, key=lambda b: b["cx"]):
        if q["cx"] <= col_c:
            continue                                           # right wing only; the model mirrors it
        y = (q["cx"] - col_c) / fppm
        w_f, h_f = q["bw"] / fppm, q["bh"] / fppm
        z_c = zmax - (q["by"] + q["bh"] / 2.0 - fy0) / fppm
        xle, ch, zc, th = wing_at(Q, y)
        if not (2.5 <= y <= 0.97 * half) or w_f > 1.4 or h_f > 1.4 or z_c > zc + 0.25:
            continue                                           # must hang below the wing, away from the fuselage
        item = {"y": round(y, 3), "z": round(z_c, 3), "height": round(max(h_f, 0.3), 3), "width": round(max(w_f, 0.3), 3),
                "x": round(xle + ch - 2.0, 3), "length": 2.8}  # fallback when the plan view does not show it
        if top_ok:
            cand = [t for t in tb if t["cy"] > row_c and abs((t["cy"] - row_c) / ppm - y) < 1.0
                    and abs((t["bx"] - x0) / ppm - (xle + ch)) < 3.0]
            if cand:
                t = min(cand, key=lambda t: abs((t["cy"] - row_c) / ppm - y))
                item["x"] = round((t["bx"] - x0) / ppm - a.under, 3)
                item["length"] = round(t["bw"] / ppm + a.under, 3)
                item["width"] = round(min(max(t["bh"] / ppm, 0.3), 1.6), 3)
        items.append(item)
    items = items[: a.max_count]
    if not items:
        if FAIR_F.exists():
            FAIR_F.unlink()
        sys.exit("no fairing-sized blobs under the wing in the front view (try --clear, or check the engines were restored)")
    jsave(FAIR_F, {"source": "front view (count, y, z, height) + plan view (x, length)", "evidence_class": "calculated",
                   "items": items})
    for it in items:
        print("  fairing", it)
    commit(P, f"{len(items)} track fairings", [], reason="flap-track fairings from front + plan views")

def cmd_probe_subsurf(a):
    A = OpenVSPAdapter(); v = A.v
    v.ClearVSPModel()
    w = v.AddGeom("WING")
    for label in ("SS_CONTROL", "SS_RECTANGLE"):
        kind = getattr(v, label, None)
        if kind is None:
            print(f"{label}: constant not available"); continue
        before = set(v.GetGeomParmIDs(w))
        try:
            v.AddSubSurf(w, kind, 0)
        except Exception as e:
            print(f"{label}: AddSubSurf failed: {e}"); continue
        print(f"{label}: new parameters")
        for pid in v.GetGeomParmIDs(w):
            if pid not in before:
                print(f"  [{v.GetParmGroupName(pid):14s}] {v.GetParmName(pid):28s} = {v.GetParmVal(pid)}")


def cmd_timing(a):
    """Where does an evaluation spend its time?"""
    P = jload(PARAMS); refs = load_refs(EVAL_DS)
    def T(label, fn):
        t = time.time(); r = fn(); print(f"  {label:36s} {time.time() - t:6.2f} s"); return r
    with fast_tess():
        A = OpenVSPAdapter()
        T("build (fuselage built)", lambda: A.build(values(P)))
        T("build (fuselage cached)", lambda: A.build(values(P)))
        tris = T("export STL + parse", lambda: A.export_stl(HOME / "renders" / "opt.stl"))
        print(f"  triangles: {len(tris)}")
        tot = 0.0
        for v, (c, ref) in refs.items():
            t = time.time(); m = model_mask(tris, v, c, ref.shape)[0]; t1 = time.time() - t
            t = time.time(); metrics(ref, m, c["ppm"]); t2 = time.time() - t
            print(f"  {v:5s} rasterise {t1:5.2f} s   metrics {t2:5.2f} s"); tot += t1 + t2
    print("(an optimisation stage costs about  evaluations x (cached build + export + the sum of the views))")

def apply_starting_values(P, L, S, D, H, hspan=None, ground_h=None):
    """Starting estimates for any transport-type aircraft (proportions of the A300-600 fit). The optimiser refines them."""
    est = {"wing.x": 0.30 * L, "wing.z": -0.07 * D, "wing.root_chord": 0.175 * L, "wing.tip_chord": 0.045 * L,
           "wing.sweep": 31.0, "wing.dihedral": 5.5, "wing.thickness": 0.14,
           "htail.x": 0.82 * L, "htail.z": 0.33 * D, "htail.root_chord": 0.09 * L, "htail.tip_chord": 0.04 * L,
           "htail.sweep": 33.0, "htail.dihedral": 6.0, "htail.thickness": 0.10,
           "vtail.x": 0.76 * L, "vtail.z": 0.40 * D, "vtail.height": 0.18 * L, "vtail.root_chord": 0.17 * L,
           "vtail.tip_chord": 0.057 * L, "vtail.sweep": 45.0, "vtail.thickness": 0.12,
           "gear.x": 0.47 * L, "gear.nose_x": 0.125 * L, "gear.track": 0.178 * L, "gear.wheel_radius": 0.11 * D}
    if not hspan:
        est["htail.span"] = 0.36 * S
    for k, v in est.items():
        if k in P:
            P[k].update({"value": round(float(v), 3), "evidence_class": "inferred", "confidence": 0.2,
                         "notes": "starting estimate (proportions of a typical transport aircraft)"})
    meas = {"fuselage.length": L, "wing.span": S, "fuselage.max_width": D, "fuselage.max_height": H}
    if hspan:
        meas["htail.span"] = hspan
    if ground_h:
        meas["gear.height_ref"] = ground_h
    for k, v in meas.items():
        P[k].update({"value": float(v), "evidence_class": "measured", "confidence": 0.9, "fixed": k != "gear.height_ref",
                     "source": "user", "notes": "confirmed by the user"})

def cmd_wizard(a):
    """A new aircraft from zero: drawings + key dimensions -> calibrated project -> auto (step by step)."""
    def ask(q, d=None):
        r_ = input(f"{q}" + (f" [{d}]" if d not in (None, "") else "") + ": ").strip().strip('"')
        return r_ if r_ else ("" if d is None else str(d))
    print("== New aircraft ==  (Enter keeps the value in brackets)")
    cmd_init(types.SimpleNamespace())
    lens = {}
    for v, label, hint in (("top", "plan (top) view", "overall fuselage length"), ("side", "side view", "overall length incl. tail"),
                           ("front", "front view", "wing span")):
        img = ask(f"Image of the {label}")
        length = float(ask(f"  {v}: real length of the silhouette's horizontal extent ({hint}), metres"))
        flip = ask("  nose on the right of the image? (y/N)", "n").lower().startswith("y")
        keep = ask("  keep detached parts such as landing-gear wheels? (Y/n)", "y").lower() != "n"
        cmd_calibrate(types.SimpleNamespace(view=v, image=img, length=length, mask=None, flip=flip, keep_parts=keep,
                                            keep_min_area=0.15))
        lens[v] = length
    D = float(ask("Fuselage max width / diameter, metres"))
    H = float(ask("Fuselage max height, metres", D))
    hs = ask("Horizontal tail span, metres (blank if unknown)", "")
    gh = ask("Height from the ground to the top of the fin, metres (blank if unknown)", "")
    P = jload(PARAMS)
    apply_starting_values(P, lens["top"], lens["front"], D, H, float(hs) if hs else None, float(gh) if gh else None)
    jsave(PARAMS, P)
    print("Starting values written to params.json (measured: lengths, span, diameter; the rest are estimates).")
    if ask("Run `auto --step --masks` now? (Y/n)", "y").lower() != "n":
        import subprocess
        sys.exit(subprocess.call([sys.executable, str(Path(__file__).resolve()), "auto", "--step", "--masks"]))
    print("Next: python aircraft_recon.py auto --step --masks")


def cmd_report_metrics(a):
    m = jload(HOME / "renders" / "metrics.json")
    if not m:
        sys.exit("no metrics.json yet; run `compare` first")
    print(f"total_error = {m['total_error']:.5f}   tess_scale = {m.get('tess_scale', '?')}")
    for v, d in m["views"].items():
        ex = d.get("extent_err_m", [0, 0]); flag = "OK"
        if abs(ex[0]) > 0.3: flag = "X-offset high"
        if abs(ex[1]) > 0.3: flag = "V-offset high (height)"
        print(f"  {v:5s} IoU={d['iou']:.3f}  chamfer={d['chamfer_m']:.3f} m  "
              f"H95={d['hausdorff95_m']:.3f} m  ext_err={ex}  -> {flag}")
    print("Tip: if `front.V-offset` is high -> `python aircraft_recon.py front-fit`")


def cmd_selftest(a):
    box = lambda L, W: np.array([[[0, -W/2, 0], [L, -W/2, 0], [L, W/2, 0]],
                                 [[0, -W/2, 0], [L, W/2, 0], [0, W/2, 0]]], float)
    ref_t = box(10, 2); ppm = 50.0
    ref, _ = model_mask(ref_t, "top", {"ppm": ppm, "bbox": [20, 519, 30, 129]}, (200, 600))
    cal = {"ppm": ppm, "bbox": [20, 519, 30, 129]}
    same, _ = model_mask(box(10, 2), "top", cal, ref.shape)
    diff, _ = model_mask(box(10.6, 2.4), "top", cal, ref.shape)
    s, d = metrics(ref, same, ppm), metrics(ref, diff, ppm)
    print("identical:", s, "\nperturbed:", d)
    assert s["iou"] > 0.99 and d["iou"] < s["iou"] and d["chamfer_m"] > s["chamfer_m"]
    # multi-section wing: perpendicular root, interpolation, merging of zero-span sections
    V = {"fuselage.max_width": 5.64, "wing.z": -1.5, "wing.dihedral": 5.0, "wing.thickness": 0.12}
    wj = {"stations": [{"y": 0, "chord": 11, "x_le": 15.0}, {"y": 3.3, "chord": 9, "x_le": 17.0, "t_rel": 1.0},
                       {"y": 12, "chord": 6, "x_le": 23.0}, {"y": 12.02, "chord": 6, "x_le": 23.1}, {"y": 22, "chord": 2.5, "x_le": 30.0}]}
    st = resolve_wing_stations(wj, V)
    assert st[0]["y"] == 0 and st[0]["x_le"] == st[1]["x_le"] == 17.0 and st[0]["chord"] == 9, st[:2]   # perpendicular root
    assert all(b["y"] - a["y"] >= 0.05 for a, b in zip(st, st[1:])), "zero-span section survived"
    xle, ch, zc, th = wing_at({"_wing_stations": st}, 7.0)
    assert 17.0 < xle < 23.0 and 6 < ch < 9 and abs(th - 0.12 * ch) < 1e-9
    # isolated projects: a copied project must not keep paths to the masks of its source
    with tempfile.TemporaryDirectory() as td:
        jsave(Path(td) / "views.json", {"top": {"mask": r"C:\other\project\processed\top_mask.png", "image": "project\\drawings\\top.png"}})
        _relocate_views(td)
        got = jload(Path(td) / "views.json")["top"]
        assert got["mask"] == str(Path(td) / "processed" / "top_mask.png") and got["image"] == "project\\drawings\\top.png", got
    print("SELFTEST OK")


def wing_summary(st, twist_mode="absolute"):
    """Plain-text description of a station list: panels (taper, sweeps, dihedral, twist) and global figures."""
    lines = [f"  {'y0':>6}{'y1':>7}  {'c_root':>7}{'c_tip':>7}{'taper':>7}{'LE sw':>7}{'c/4 sw':>8}{'dihed':>7}{'twist':>7}{'t/c':>6}"]
    for i in range(1, len(st)):
        a, b = st[i - 1], st[i]
        dy, dz, dx = b["y"] - a["y"], b["z"] - a["z"], b["x_le"] - a["x_le"]
        panel = float(np.hypot(dy, dz))
        le = np.degrees(np.arctan2(dx, panel)); c4 = np.degrees(np.arctan2(dx + 0.25 * (b["chord"] - a["chord"]), panel))
        lines.append(f"  {a['y']:6.2f}{b['y']:7.2f}  {a['chord']:7.2f}{b['chord']:7.2f}{b['chord'] / a['chord']:7.3f}"
                     f"{le:7.1f}{c4:8.1f}{np.degrees(np.arctan2(dz, dy)):7.1f}{b['twist']:7.1f}{b['thickness']:6.3f}"
                     f"{'  (winglet)' if b.get('winglet') else ''}")
    body = [s for s in st if not s.get("winglet")]
    half = body[-1]["y"]
    area = 2.0 * sum(0.5 * (a["chord"] + b["chord"]) * (b["y"] - a["y"]) for a, b in zip(body, body[1:]))
    cr = [(a["chord"], b["chord"], b["y"] - a["y"]) for a, b in zip(body, body[1:])]
    mac = 2.0 * sum((ca * ca + ca * cb + cb * cb) / 3.0 * dy for ca, cb, dy in cr) / area
    lines.append(f"  planform span {2 * half:.2f} m (total with winglet {2 * st[-1]['y']:.2f} m), area {area:.1f} m2, "
                 f"aspect ratio {(2 * half) ** 2 / area:.2f}, mean chord {area / (2 * half):.2f} m, MAC {mac:.2f} m")
    return "\n".join(lines)

def cmd_wing_stations(a):
    """Create / load / show wing.json (multi-section wing)."""
    if getattr(a, "load", None):
        wj = jload(Path(a.load))
        if not wj or not wj.get("stations"):
            sys.exit(f"{a.load}: no 'stations' list")
        HOME.mkdir(exist_ok=True); jsave(WING_F, wj)
        print(f"wing.json installed from {a.load} ({len(wj['stations'])} stations). Run `build` (or `wing-check`).")
    elif a.init:
        P = jload(PARAMS)
        span = P["wing.span"]["value"]; half = span / 2.0
        x0 = P["wing.x"]["value"]; rc = P["wing.root_chord"]["value"]; tc = P["wing.tip_chord"]["value"]
        sweep = P["wing.sweep"]["value"]
        fus_half = P["fuselage.max_width"]["value"] / 2.0
        ys = [0.0, fus_half + 0.5, 0.40 * half, 0.62 * half, half]
        sts = [{"y": round(y, 3), "chord": round(rc + (tc - rc) * y / half, 3),
                "x_le": round(x0 + y * np.tan(np.radians(sweep)), 3), "t_rel": round(1.0 - 0.33 * y / half, 3)} for y in ys]
        jsave(WING_F, {"source": "initial estimate from params.json", "root_perpendicular": True, "z_from_params": True,
                       "stations": sts})
        print(f"wing.json created with {len(sts)} stations; edit it by hand, then `build`.")
    wj = jload(WING_F)
    if not wj:
        sys.exit("no wing.json; use `wing-stations --init`, `--load FILE` or `wing-fit`")
    st = resolve_wing_stations(wj, values(jload(PARAMS)) if PARAMS.exists() else {"fuselage.max_width": wj.get("fuselage_width", 5.64)})
    print(wing_summary(st, wj.get("twist_mode", "absolute")))

def wing_json_is_auto():
    wj = jload(WING_F)
    return (not wj) or str(wj.get("source", "")).startswith("top mask")

def fit_wing_from_mask(P, n_y=250):
    """Measure the leading and trailing edge of the wing in the plan-view mask and write wing.json.
    The part inside the fuselage has no data: the wing is made perpendicular to it (root_perpendicular)."""
    from scipy.signal import savgol_filter
    from scipy.ndimage import median_filter
    c = jload(VIEWS_F, {}).get("top")
    if not c:
        raise SystemExit("calibrate the top view first")
    mask = imread(c["mask"], 0) > 0
    x0_px, x1_px, y0_px, y1_px = c["bbox"]
    ppm = c["ppm"]; row_c = (y0_px + y1_px) / 2.0
    half = P["wing.span"]["value"] / 2.0
    L = P["fuselage.length"]["value"]
    fus_half = P["fuselage.max_width"]["value"] / 2.0
    ys_m = np.linspace(fus_half + 1.0, half, n_y)
    le_x = np.full(n_y, np.nan); te_x = np.full(n_y, np.nan)
    x_le_expected = L * 0.35
    for k, y in enumerate(ys_m):
        row = int(round(row_c + y * ppm))
        if not (0 <= row < mask.shape[0]):
            continue
        cols = np.where(mask[row, :])[0]
        if len(cols) < 3:
            continue
        runs, start, prev = [], cols[0], cols[0]
        for cc in cols[1:]:
            if cc - prev > 4:
                runs.append((start, prev)); start = cc
            prev = cc
        runs.append((start, prev))
        best = None; best_d = 1e9
        for (r0, r1) in runs:                               # the run that looks most like a wing chord
            w_m = (r1 - r0) / ppm
            if w_m < 0.05 * L or w_m > 0.26 * L:
                continue
            le_val = (r0 - x0_px) / ppm
            if le_val < L * 0.20 or le_val > L * 0.65:
                continue
            d = abs(le_val - x_le_expected)
            if d < best_d:
                best_d = d; best = (le_val, (r1 - x0_px) / ppm)
        if best is not None:
            le_x[k], te_x[k] = best
    valid = np.isfinite(le_x) & np.isfinite(te_x)
    if valid.sum() < 20:
        raise SystemExit(f"only {valid.sum()} valid rows in the plan mask; check the top mask")
    y_v, le_v, te_v = ys_m[valid], le_x[valid], te_x[valid]
    w = max(5, min(15, (len(le_v) // 4) * 2 + 1))
    le_v = savgol_filter(median_filter(le_v, w, mode="nearest"), w, 2)
    te_v = savgol_filter(median_filter(te_v, w, mode="nearest"), w, 2)
    # Two-slope model (a kink) for the leading and the trailing edge, with a shared kink position found by least squares.
    def hinge(y, v, yk):
        A = np.column_stack([np.ones_like(y), y, np.maximum(0.0, y - yk)])
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return coef, float(np.sum((A @ coef - v) ** 2))
    span_y = y_v[-1] - y_v[0]
    cands = np.linspace(y_v[0] + 0.12 * span_y, y_v[-1] - 0.12 * span_y, 80)
    sse = [hinge(y_v, le_v, yk)[1] + hinge(y_v, te_v, yk)[1] for yk in cands]
    y_kink = float(cands[int(np.argmin(sse))])
    flat = np.linalg.lstsq(np.column_stack([np.ones_like(y_v), y_v]), np.column_stack([le_v, te_v]), rcond=None)[1].sum()
    if flat <= 0 or (flat - min(sse)) / flat < 0.15:          # no real kink in the data: put the section where wings usually have it
        y_kink = float(y_v[0] + 0.40 * span_y)
    c_le, _ = hinge(y_v, le_v, y_kink); c_te, _ = hinge(y_v, te_v, y_kink)
    model = lambda c, y: float(c[0] + c[1] * y + c[2] * max(0.0, y - y_kink))
    y_j = fus_half + 0.5                                     # fuselage side (junction): the root station copies it
    ys_st = [0.0, y_j]
    for yy in (y_kink, y_kink + (half - y_kink) * 0.5, half):
        if yy - ys_st[-1] >= 1.0 and (half - yy >= 1.0 or yy == half):
            ys_st.append(yy)
    if ys_st[-1] != half:
        ys_st[-1] = half
    st = []
    for y in ys_st:
        le_val, te_val = model(c_le, max(y, y_j)), model(c_te, max(y, y_j))
        st.append({"y": round(float(y), 3), "chord": round(float(te_val - le_val), 3), "x_le": round(float(le_val), 3),
                   "t_rel": round(1.0 - 0.33 * y / half, 3)})
    for s in st:
        if s["chord"] < 0.02 * L or s["chord"] > 0.30 * L or s["x_le"] < 0.05 * L or s["x_le"] > 0.8 * L:
            raise SystemExit(f"measured wing is not plausible at y={s['y']:.2f}: chord={s['chord']:.2f} x_le={s['x_le']:.2f}")
    jsave(WING_F, {"source": "top mask (auto-measured)", "kink_y": round(y_kink, 3), "root_perpendicular": True,
                   "z_from_params": True, "stations": st})
    print(f"  wing: {len(st)} stations measured from the plan mask (kink at y={y_kink:.2f} m; root perpendicular to the fuselage)")
    for s in st:
        print(f"    y={s['y']:6.2f}  chord={s['chord']:6.2f}  x_le={s['x_le']:6.2f}")
    return st

def cmd_wing_fit(a):
    P = jload(PARAMS)
    fit_wing_from_mask(P)
    commit(jload(PARAMS), "wing-fit from top mask", [], reason="wing stations measured from the plan mask")

def cmd_wing_check(a):
    """Build the model and read every wing section back from OpenVSP: expected vs actual."""
    P = jload(PARAMS); V = values(P)
    st = V.get("_wing_stations")
    if not st:
        sys.exit("no wing.json in use (create one with `wing-fit`, `wing-stations --init` or `--load`)")
    print(wing_summary(st, V.get("_wing_twist_mode", "absolute")))
    A = OpenVSPAdapter(); A.build(V)
    print(f"wing built as: {OpenVSPAdapter._wing_mode} ({1 + len(A._ids.get('WingSegs', []))} geometr"
          f"{'y' if OpenVSPAdapter._wing_mode == 'multi' else 'ies'})")
    if OpenVSPAdapter._wing_mode == "multi":
        print("  section parameters read back from OpenVSP (expected -> actual):")
        for i in range(1, len(st)):
            g = A._seg_geometry(st, i, V.get("_wing_twist_mode", "absolute"))
            print("   sec %d  " % i + "  ".join(f"{nm} {g[nm]:.3f}->{(lambda r: float('nan') if r is None else r)(A._sec_get(A._ids['Wing'], i, nm)):.3f}"
                                              for nm in ("Span", "Root_Chord", "Tip_Chord", "Sweep", "Dihedral", "Twist")))
    ext = {k: e for k, e in A.geom_extents().items() if k.startswith("Wing")}
    if ext:
        lo = np.min([e[0] for e in ext.values()], axis=0); hi = np.max([e[1] for e in ext.values()], axis=0)
        print(f"  exported wing mesh: x {lo[0]:.2f}..{hi[0]:.2f}  y {lo[1]:.2f}..{hi[1]:.2f}  z {lo[2]:.2f}..{hi[2]:.2f}"
              f"   (stations: tip y={st[-1]['y']:.2f}, root x_le={st[0]['x_le']:.2f})")

def cmd_wing_export(a):
    """Build ONLY the wing of a wing JSON (for example a340_wing.json) in OpenVSP and save it as .vsp3."""
    src_ = Path(a.json)
    wj = jload(src_)
    if not wj or not wj.get("stations"):
        sys.exit(f"{src_}: no 'stations' list")
    V = {"fuselage.max_width": wj.get("fuselage_width", 5.64), "wing.z": 0.0, "wing.dihedral": 0.0, "wing.thickness": 0.12}
    st = resolve_wing_stations(wj, V)
    P = {"_wing_stations": st, "_wing_ctrl": wj.get("controls"), "_wing_umode": wj.get("u_mode", "span"),
         "_wing_twist_mode": wj.get("twist_mode", "absolute"), "fuselage.length": wj.get("fuselage_length", 60.0)}
    print(wing_summary(st, P["_wing_twist_mode"]))
    A = OpenVSPAdapter(); v = A.v
    v.ClearVSPModel(); A._ids = {}; A._made = 0; A._pc = {}
    if a.segments:
        OpenVSPAdapter._wing_force_segments = True
    A._ids["Wing"] = A._wing_stations_build("Wing", P)
    if a.fuselage:
        try:
            D = float(wj.get("fuselage_width", 5.64)); Lf = float(wj.get("fuselage_length", 60.0))
            A._body("Fuselage", Lf, [(0.0, 0.0), (0.03, 0.55), (0.10, 0.95), (0.18, 1.0), (0.75, 1.0), (0.92, 0.55), (1.0, 0.0)],
                    D, (0.0, 0.0, 0.0))
        except Exception as e:
            print(f"  note: reference fuselage not added ({e})")
    ctrl = getattr(v, "SS_CONTROL", None)
    if ctrl is not None and not a.no_controls:
        ids = {v.GetGeomName(g): g for g in v.FindGeoms()}
        made, nset = A._wing_controls(P, ids, ctrl)
        print(f"  control surfaces: {made} added ({nset} parameters set)")
    v.Update()
    out = Path(a.out) if a.out else src_.with_suffix(".vsp3")
    A.save(out)
    print(f"wing built as '{OpenVSPAdapter._wing_mode}' -> {out}")

# ---- one isolated project per aircraft: aircraft/NAME (the old project/ folder keeps working until it is adopted) -------
ROOT = Path(__file__).resolve().parent
AIRCRAFT = ROOT / "aircraft"
CURRENT_F = AIRCRAFT / "current.txt"
LEGACY = "project"

def _project_dir(name):
    return ROOT / LEGACY if name == LEGACY else AIRCRAFT / name

def _is_project(name):
    return (_project_dir(name) / "params.json").exists()

def _new_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name == LEGACY:
        sys.exit(f"invalid project name '{name}' (use letters, digits, '_', '-', '.'; '{LEGACY}' is reserved)")
    if _project_dir(name).exists():
        sys.exit(f"project '{name}' already exists: {_project_dir(name)}")
    return name

def _set_project(path):
    """Point every project path of this process to one folder (all commands read these globals when they run)."""
    global HOME, PARAMS, VIEWS_F, STATE, FUSE_F, FAIR_F, WING_F, _TESS_SCALE_CACHE
    HOME = Path(path)
    PARAMS, VIEWS_F, STATE = HOME / "params.json", HOME / "views.json", HOME / "state.json"
    FUSE_F, FAIR_F, WING_F = HOME / "fuselage.json", HOME / "fairings.json", HOME / "wing.json"
    _TESS_SCALE_CACHE = None

def _select_project(requested):
    """--project > AIRCRAFT_PROJECT > aircraft/current.txt > the old project/ folder. A named project that is missing is an
    error: falling back silently would write one aircraft's results into another."""
    name = requested or os.environ.get("AIRCRAFT_PROJECT")
    if not name and CURRENT_F.exists():
        name = CURRENT_F.read_text(encoding="utf-8").strip()
    name = name or LEGACY
    if not _is_project(name) and name != LEGACY:
        sys.exit(f"project '{name}' not found. See `project-list`.")
    _set_project(_project_dir(name))
    return name

def _use_project(name):
    AIRCRAFT.mkdir(exist_ok=True)
    CURRENT_F.write_text(name, encoding="utf-8")
    print(f"current project: {name}  ({_project_dir(name)})")

def _relocate_views(home):
    """views.json holds absolute paths, and erase-mask/restore-mask/masks WRITE to them: a copied or moved project must point
    to its own files, otherwise it would read and overwrite the masks of the project it came from."""
    f = Path(home) / "views.json"; V = jload(f, {}) or {}
    for c in V.values():
        for key, sub in (("mask", "processed"), ("image", "drawings")):
            if c.get(key):
                new = Path(home) / sub / re.split(r"[\\/]", str(c[key]))[-1]
                if new.exists() or key == "mask":
                    c[key] = str(new)
    if f.exists():
        jsave(f, V)

def _checked_json(path, key=None):
    data = jload(path) if path and Path(path).exists() else None
    if not isinstance(data, dict) or not data or (key and not data.get(key)):
        sys.exit(f"{path}: file not found or not valid" + (f" (needs a '{key}' list)" if key else ""))
    return Path(path)

def cmd_project_new(a):
    _new_name(a.name)
    src = [(_checked_json(a.params), "params.json")] if a.params else []     # check every file before creating anything
    src += [(_checked_json(a.fuselage, "stations"), "fuselage.json")] if a.fuselage else []
    src += [(_checked_json(a.wing, "stations"), "wing.json")] if a.wing else []
    src += [(_checked_json(a.fairings, "items"), "fairings.json")] if a.fairings else []   # without images, fairings-fit cannot measure them
    _set_project(_project_dir(a.name)); HOME.mkdir(parents=True)
    for s, dst in src:
        shutil.copyfile(s, HOME / dst)
    cmd_init(a)
    cmd_migrate(a)
    _use_project(a.name)

def cmd_project_list(a):
    cur = CURRENT_F.read_text(encoding="utf-8").strip() if CURRENT_F.exists() else LEGACY
    names = sorted(d.name for d in AIRCRAFT.iterdir() if d.is_dir() and (d / "params.json").exists()) if AIRCRAFT.exists() else []
    for n in names + ([LEGACY] if _is_project(LEGACY) else []):
        d = _project_dir(n); st = jload(d / "state.json", {}) or {}
        print(f"{'*' if n == cur else ' '} {n:20s} current version {st.get('current', 0) or '-':>4}  "
              f"{len(list((d / 'versions').glob('v*.json'))):4d} saved  views {sorted(jload(d / 'views.json', {}) or {})}"
              f"{'  (old folder; move it with project-adopt NAME)' if n == LEGACY else ''}")
    if not names and not _is_project(LEGACY):
        print("no projects; create one with project-new NAME")

def cmd_project_use(a):
    if not _is_project(a.name):
        sys.exit(f"project '{a.name}' not found. See `project-list`.")
    _use_project(a.name)

def cmd_project_copy(a):
    """Copy a project; without --full the copy starts with no versions, iterations or renders."""
    if not _is_project(a.src):
        sys.exit(f"project '{a.src}' not found. See `project-list`.")
    _new_name(a.dst); AIRCRAFT.mkdir(exist_ok=True)
    dst = _project_dir(a.dst); tmp = dst.with_name(dst.name + ".copying")   # a failed copy never looks like a project
    if tmp.exists():
        sys.exit(f"leftover folder from a failed copy, remove it first: {tmp}")
    skip = None if a.full else shutil.ignore_patterns("versions", "iterations", "renders")
    shutil.copytree(_project_dir(a.src), tmp, ignore=skip)
    for d in DIRS:
        (tmp / d).mkdir(exist_ok=True)
    if not a.full:
        st = jload(tmp / "state.json", {}) or {}; st["current"] = 0; jsave(tmp / "state.json", st)
    os.replace(tmp, dst)
    _relocate_views(dst)
    print(f"{a.src} copied to {a.dst} ({'with' if a.full else 'without'} versions)")
    if a.use:
        _use_project(a.dst)

def cmd_project_adopt(a):
    """Move the old project/ folder to aircraft/NAME."""
    if not _is_project(LEGACY):
        sys.exit(f"no {ROOT / LEGACY} folder to adopt")
    _new_name(a.name); AIRCRAFT.mkdir(exist_ok=True)
    os.replace(ROOT / LEGACY, _project_dir(a.name))
    _relocate_views(_project_dir(a.name))
    print(f"{LEGACY}/ moved to {_project_dir(a.name)}")
    _use_project(a.name)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", help="project for this command only (goes BEFORE the command); also env AIRCRAFT_PROJECT")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("project-new", help="create aircraft/NAME and select it"); s.add_argument("name")
    s.add_argument("--params"); s.add_argument("--fuselage"); s.add_argument("--wing"); s.add_argument("--fairings"); s.set_defaults(fn=cmd_project_new)
    sp.add_parser("project-list", help="list projects (* = current)").set_defaults(fn=cmd_project_list)
    s = sp.add_parser("project-use", help="select the current project"); s.add_argument("name"); s.set_defaults(fn=cmd_project_use)
    s = sp.add_parser("project-copy", help="copy a project (without versions unless --full)"); s.add_argument("src"); s.add_argument("dst")
    s.add_argument("--full", action="store_true"); s.add_argument("--use", action="store_true"); s.set_defaults(fn=cmd_project_copy)
    s = sp.add_parser("project-adopt", help="move the old project/ folder to aircraft/NAME"); s.add_argument("name"); s.set_defaults(fn=cmd_project_adopt)
    sp.add_parser("selftest").set_defaults(fn=cmd_selftest)
    s = sp.add_parser("wing-stations", help="multi-section wing: show wing.json; --init estimate, --load FILE install")
    s.add_argument("--init", action="store_true"); s.add_argument("--load", help="install a wing JSON (e.g. a340_wing.json)")
    s.set_defaults(fn=cmd_wing_stations)
    sp.add_parser("wing-fit", help="measure the wing from the plan-view mask and write wing.json").set_defaults(fn=cmd_wing_fit)
    sp.add_parser("wing-check", help="build and read every wing section back from OpenVSP").set_defaults(fn=cmd_wing_check)
    s = sp.add_parser("wing-export", help="build only the wing of a wing JSON and save a .vsp3")
    s.add_argument("json"); s.add_argument("--out"); s.add_argument("--fuselage", action="store_true", help="add a plain reference fuselage")
    s.add_argument("--no-controls", action="store_true"); s.add_argument("--segments", action="store_true", help="one geometry per panel")
    s.set_defaults(fn=cmd_wing_export)
    s = sp.add_parser("check-env"); s.add_argument("--vsp-path"); s.set_defaults(fn=cmd_check_env)
    sp.add_parser("check-var-comp").set_defaults(fn=cmd_check_var_comp)
    sp.add_parser("init").set_defaults(fn=cmd_init)
    sp.add_parser("migrate").set_defaults(fn=cmd_migrate)
    s = sp.add_parser("calibrate"); s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--image", required=True); s.add_argument("--length", type=float, required=True)
    s.add_argument("--mask"); s.add_argument("--flip", action="store_true")
    s.add_argument("--keep-parts", action="store_true", help="also keep detached parts near the aircraft (landing-gear wheels)")
    s.add_argument("--keep-min-area", type=float, default=0.15, help="smallest detached part kept, m2")
    s.set_defaults(fn=cmd_calibrate)
    s = sp.add_parser("landmarks"); s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--image", required=True); s.add_argument("--mask")
    s.add_argument("--flip", action="store_true"); s.set_defaults(fn=cmd_landmarks)
    s = sp.add_parser("erase-mask"); s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--rect", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"))
    s.add_argument("--zoom", type=float, default=0.5); s.set_defaults(fn=cmd_erase_mask)
    s = sp.add_parser("auto"); s.add_argument("--passes", type=int, default=2)
    s.add_argument("--phase", choices=["airframe", "extras", "finish", "all"], default="all")
    s.add_argument("--refit", action="store_true", help="re-measure the nacelles even if they exist")
    s.add_argument("--step", action="store_true", help="pause after every step: overlays, then continue / undo / quit")
    s.add_argument("--masks", action="store_true", help="(re)do the engine/gear mask selection inside auto")
    s.add_argument("--no-guard", dest="guard", action="store_false", help="do not revert stages that worsen the overall error")
    s.add_argument("--zoom", type=float, default=0.5, help="display scale of the selection windows")
    s.add_argument("--no-finish", action="store_true")
    s.add_argument("--no-simplify", action="store_true", help="(kept for compatibility; fit_fuselage already drops redundant stations)")
    s.add_argument("--pchip-simplify", action="store_true", help="finish phase: also run the slow PCHIP section search (fuselage-simplify)")
    s.add_argument("--simplify-tol", type=float, default=SIMPLIFY_TOL,
                   help="drop fuselage stations within this fraction of the diameter of the line between neighbours (0 = keep all)")
    s.add_argument("--iters", type=int, default=80); s.add_argument("--wide", type=float, default=1.75)
    s.add_argument("--stations", type=int, default=31); s.set_defaults(fn=cmd_auto)
    sp.add_parser("build").set_defaults(fn=lambda a: commit(jload(PARAMS), "initial build", []))
    sp.add_parser("compare").set_defaults(fn=cmd_compare)
    s = sp.add_parser("optimize"); s.add_argument("--free", nargs="+", required=True)
    s.add_argument("--iters", type=int, default=150)
    s.add_argument("--method", default="Powell", choices=["Powell", "Nelder-Mead"])
    s.add_argument("--weights", nargs="*", default=[]); s.add_argument("--allow-measured", action="store_true")
    s.set_defaults(fn=cmd_optimize)
    for nm, dl in (("set", False), ("delta", True)):
        s = sp.add_parser(nm); s.add_argument("key"); s.add_argument("value", type=float)
        s.add_argument("--force", action="store_true"); s.set_defaults(fn=lambda a, d=dl: cmd_set(a, d))
    s = sp.add_parser("patch"); s.add_argument("file"); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_patch)
    s = sp.add_parser("fuselage-simplify", help="drop the fuselage sections that barely change (lighter for aero)")
    s.add_argument("--dense", type=int, default=61, help="starting number of stations")
    s.add_argument("--tols", type=float, nargs="+", default=[0.12, 0.08, 0.05, 0.03, 0.02, 0.01], help="tolerances to try, metres")
    s.add_argument("--allow", type=float, default=0.003, help="largest accepted rise of the silhouette error")
    s.set_defaults(fn=cmd_fuselage_simplify)
    s = sp.add_parser("fuselage-test", help="try uniform/cosine spacing and C2 on/off, keep the best")
    s.add_argument("--stations", type=int, default=31)
    s.add_argument("--simplify-tol", type=float, default=SIMPLIFY_TOL,
                   help="drop fuselage stations within this fraction of the diameter of the line between neighbours (0 = keep all)")
    s.set_defaults(fn=cmd_fuselage_test)
    s = sp.add_parser("fuselage-fit"); s.add_argument("--stations", type=int, default=31)
    s.add_argument("--spacing", choices=["uniform", "cosine"]); s.add_argument("--c2", choices=["on", "off"])
    s.add_argument("--simplify-tol", type=float, default=SIMPLIFY_TOL,
                   help="drop fuselage stations within this fraction of the diameter of the line between neighbours (0 = keep all)")
    s.add_argument("--tang", choices=["off", "A", "B", "C"], help="tangent-angle convention (see fuselage-test)")
    s.add_argument("--ends", choices=["point", "round"], help="nose/tail as a point or a small rounded end")
    s.set_defaults(fn=cmd_fuselage_fit)
    s = sp.add_parser("smooth"); s.add_argument("--scale", type=float, default=1.6)
    s.add_argument("--stations", type=int, default=FUSE_STATIONS_FINE); s.set_defaults(fn=cmd_smooth)
    s = sp.add_parser("front-fit"); s.add_argument("--iters", type=int, default=150)
    s.add_argument("--wide", type=float, default=1.5); s.set_defaults(fn=cmd_front_fit)
    s = sp.add_parser("nacelle-fit", help="measure nacelles from the plan view, add them and refine")
    s.add_argument("--iters", type=int, default=200); s.add_argument("--wide", type=float, default=1.0)
    s.add_argument("--no-init", action="store_true", help="skip the plan-view measurement, only refine")
    s.set_defaults(fn=cmd_nacelle_fit)
    s = sp.add_parser("masks", help="choose the active mask variant: core (no engines) or extras (with engines)")
    s.add_argument("--use", choices=["core", "extras"], required=True); s.set_defaults(fn=cmd_masks)
    s = sp.add_parser("restore-mask", help="put the engines back into a mask (inverse of erase-mask)")
    s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--rect", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"))
    s.add_argument("--zoom", type=float, default=0.5); s.set_defaults(fn=cmd_restore_mask)
    s = sp.add_parser("fairings-fit", help="flap-track fairings: count/height from the front view, x from the plan view")
    s.add_argument("--clear", action="store_true", help="remove the fairings from the model")
    s.add_argument("--max-count", type=int, default=8)
    s.add_argument("--under", type=float, default=1.5, help="metres the fairing extends forward under the wing")
    s.set_defaults(fn=cmd_fairings_fit)
    sp.add_parser("probe-subsurf", help="list the parameters of control-surface / window sub-surfaces").set_defaults(fn=cmd_probe_subsurf)
    sp.add_parser("timing", help="time the parts of one evaluation").set_defaults(fn=cmd_timing)
    sp.add_parser("wizard", help="new aircraft from zero: drawings, key dimensions, calibration, auto").set_defaults(fn=cmd_wizard)
    sp.add_parser("report-metrics").set_defaults(fn=cmd_report_metrics)
    sp.add_parser("inspect", help="diagnose a broken model: component sizes, parameters, calibration").set_defaults(fn=cmd_inspect)
    s = sp.add_parser("rescale", help="fix the scale of a calibrated view without redoing its mask")
    s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--length", type=float, required=True); s.set_defaults(fn=cmd_rescale)
    sp.add_parser("selfcheck", help="compare the fast in-place update with a full rebuild").set_defaults(fn=cmd_selfcheck)
    sp.add_parser("probe").set_defaults(fn=cmd_probe)
    s = sp.add_parser("probe-geom"); s.add_argument("--type", required=True)
    s.add_argument("--all", action="store_true"); s.set_defaults(fn=cmd_probe_geom)
    s = sp.add_parser("simplify", help="drop fuselage stations that do not change the shape (fewer sections for VSPAERO)")
    s.add_argument("--tol", type=float, default=SIMPLIFY_TOL, help="fraction of the fuselage diameter (0 = keep all)")
    s.set_defaults(fn=cmd_simplify)
    s = sp.add_parser("history"); s.add_argument("--last", type=int, help="only the last N versions")
    s.set_defaults(fn=cmd_history)
    s = sp.add_parser("diff"); s.add_argument("a", type=int, nargs="?"); s.add_argument("b", type=int, nargs="?")
    s.set_defaults(fn=cmd_diff)
    sp.add_parser("undo").set_defaults(fn=cmd_undo)
    s = sp.add_parser("checkout", help="restore version N (a whole number, e.g. `checkout 18`)")
    s.add_argument("n", type=int, help="version number from `history`"); s.set_defaults(fn=cmd_checkout)
    a = ap.parse_args()
    if not a.cmd.startswith("project-"):
        print(f"[project: {_select_project(a.project)}]")
    a.fn(a)

if __name__ == "__main__":
    main()
