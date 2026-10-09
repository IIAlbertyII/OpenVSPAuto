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
  history | undo | checkout N | diff [A] [B]
Coordinates: X nose->tail, Y right, Z up (OpenVSP native). Units: m, deg.
"""
import argparse, glob, json, os, re, shutil, sys, tempfile, time, platform
from pathlib import Path
import numpy as np
HOME = Path(__file__).resolve().parent / "project"
DIRS = ["drawings", "processed", "versions", "iterations", "renders"]
PARAMS, VIEWS_F, STATE = HOME / "params.json", HOME / "views.json", HOME / "state.json"
FUSE_F = HOME / "fuselage.json"
FAIR_F = HOME / "fairings.json"      # flap-track fairings measured from the plan view

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
FAST_SCALE = 0.6      # mesh quality used while evaluating silhouettes (final models keep the fine mesh)
EVAL_DS = 0.5         # evaluation masks are downscaled by this factor during optimisation
TESS_W_FINE, TESS_U_FINE = 61, 25
FUSE_STATIONS_FINE = 51
FUSE_PROFILE = [0.8, 1.0, 0.7]
VIEWS = {"top": (0, 1, 1), "side": (0, 2, -1), "front": (1, 2, -1)}
ALIGN = {"top": ("min", "center"), "side": ("min", "min"), "front": ("center", "min")}
DEFAULT_W = {"top": 1.0, "side": 1.0, "front": 1.0}

_TESS_SCALE_CACHE = None
_FAST = False          # True while optimising/comparing: coarse mesh (fast); committed models use the persisted scale


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


class OpenVSPAdapter:
    _last_report = None

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

    def build(self, P):
        v = self.v
        self._pc = {}
        s = _get_tess_scale()
        fw = int(round((TESS_W + 8) * s)); fu = int(round(TESS_U * s))
        rep_ = (round(s, 3), _FAST)
        if OpenVSPAdapter._last_report != rep_:               # report only when the mesh quality changes
            print(f"[build] TESS_SCALE={s:.2f}{' (fast, evaluation only)' if _FAST else ''}  "
                  f"fuselage Tess_W={fw} Tess_U={fu}")
            OpenVSPAdapter._last_report = rep_
        sig = None
        if _FAST and P.get("_stations"):                      # during optimisation the fuselage never changes
            sig = (round(float(P["fuselage.length"]), 4), round(s, 3),
                   tuple((round(t["x"], 3), round(t["width"], 3), round(t["height"], 3), round(t["z_center"], 3))
                         for t in P["_stations"]),
                   bool(P.get("_fuse_c2", False)), P.get("_fuse_tang", "off"), P.get("_fuse_ends", "point"))
        f = None
        if sig is not None and OpenVSPAdapter._fuse_sig == sig and OpenVSPAdapter._fuse_id:
            try:
                ids = list(v.FindGeoms())
                if OpenVSPAdapter._fuse_id in ids and v.GetGeomName(OpenVSPAdapter._fuse_id) == "Fuselage":
                    for g in ids:
                        if g != OpenVSPAdapter._fuse_id:
                            v.DeleteGeom(g)
                    f = OpenVSPAdapter._fuse_id
            except Exception:
                f = None
        if f is None:
            v.ClearVSPModel()
            OpenVSPAdapter._fuse_sig = None; OpenVSPAdapter._fuse_id = None
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
        self._wing("Wing", "wing", P, False)
        self._wing("HTail", "htail", P, False)
        self._wing("VTail", "vtail", P, True)
        extra = self._add_propulsion_and_gear(P)
        v.Update()
        ng = len(v.FindGeoms())
        expected = 4 + extra
        if ng != expected:
            raise RuntimeError(f"expected {expected} geometries in the model, found {ng}")
        L = float(P["fuselage.length"]); span = max(float(P["wing.span"]), float(P["htail.span"]))
        for name, bb in ([] if _FAST else self.geom_report()):     # (evaluate() has its own size check while optimising)                   # a component far bigger than the aircraft = broken parameters
            if bb.get("X_Len", 0) > 2.2 * L or bb.get("Y_Len", 0) > 2.2 * span or bb.get("Z_Len", 0) > 2.2 * L:
                raise RuntimeError(f"geometry '{name}' is oversized: bbox={ {k: round(x, 1) for k, x in bb.items()} } "
                                   f"(fuselage.length={L}, span={span}). Run `inspect`.")

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
            v.ChangeXSecShape(surf, i, v.XS_POINT if (ends == "point" and i in (0, N - 1)) else v.XS_ELLIPSE)
        xsecs = [v.GetXSec(surf, i) for i in range(N)]        # ids are stable once all shapes are set
        for i, s_ in enumerate(st):
            xs = xsecs[i]
            if ends == "round" or i not in (0, N - 1):           # "round": nose and tail are small ellipses + round cap
                tiny = 0.12 if i in (0, N - 1) else 0.05
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
        if ends == "round":
            ok = [self._set_safe(f, nm, "EndCap", 2.0) for nm in ("CapUMinOption", "CapUMaxOption")]   # 2 = round cap
            if not any(ok) and not getattr(OpenVSPAdapter, "_cap_warned", False):
                print("  note: this OpenVSP has no CapUMinOption/CapUMaxOption; the nose stays an open small ellipse")
                OpenVSPAdapter._cap_warned = True
        if c2:                                                # optional: C2 skin (can oscillate with clustered sections)
            for xs in xsecs:
                try:
                    v.SetXSecContinuity(xs, 2)
                except Exception:
                    pass

    def _wing(self, name, pre, P, vertical):
        v = self.v
        w = v.AddGeom("WING"); v.SetGeomName(w, name)
        self._tess(w, w=TESS_W)
        v.SetParmVal(w, "X_Rel_Location", "XForm", P[pre + ".x"])
        v.SetParmVal(w, "Z_Rel_Location", "XForm", P[pre + ".z"])
        v.SetParmVal(w, "Sym_Planar_Flag", "Sym", 0 if vertical else v.SYM_XZ)
        if vertical:
            v.SetParmVal(w, "X_Rel_Rotation", "XForm", 90.0)
        v.SetDriverGroup(w, 1, v.SPAN_WSECT_DRIVER, v.ROOTC_WSECT_DRIVER, v.TIPC_WSECT_DRIVER)
        dih = 0.0 if vertical else float(P.get(pre + ".dihedral", 0.0))
        span = P[pre + ".height"] if vertical else P[pre + ".span"] / 2.0 / np.cos(np.radians(dih))
        for nm, val in (("Span", span), ("Root_Chord", P[pre + ".root_chord"]),
                        ("Tip_Chord", P[pre + ".tip_chord"]), ("Sweep_Location", 0.0),
                        ("Sweep", P[pre + ".sweep"])):
            v.SetParmVal(w, nm, "XSec_1", val)
        if pre + ".dihedral" in P:
            v.SetParmVal(w, "Dihedral", "XSec_1", P[pre + ".dihedral"])
        if pre + ".thickness" in P and getattr(self, "thickness_ok", True):
            surf = v.GetXSecSurf(w, 0)
            for i in (0, 1):
                v.ChangeXSecShape(surf, i, v.XS_FOUR_SERIES)
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
        for nm, val in (("Span", span), ("Root_Chord", chord), ("Tip_Chord", chord),
                        ("Sweep_Location", 0.0), ("Sweep", 0.0)):
            v.SetParmVal(w, nm, "XSec_1", float(val))
        self._thickness(w, tc)
        return w

    def _propulsion(self, P):
        L, D = P["nacelle.length"], P["nacelle.diameter"]
        x, y, z = P["nacelle.x"], abs(P["nacelle.y"]), P["nacelle.z"]
        self._body("Nacelle", L, NAC_PROF, D, (x, y, z), mirror=True)
        if not _FAST and P.get("engine.length", 0) >= 0.1 and P.get("engine.diameter", 0) >= 0.05:
            self._body("Engine", P["engine.length"], ENG_PROF, P["engine.diameter"],
                       (P["engine.x"], abs(P["engine.y"]), P["engine.z"]), mirror=True)
        xle, ch, zc, th = wing_at(P, y)                       # pylon: vertical plate from nacelle top to the wing
        z_top = z + D / 2.0
        h = (zc - 0.5 * th) - (z_top - 0.2)
        if h > 0.15:
            x0 = x + 0.12 * L
            chord = max(xle + 0.40 * ch - x0, 1.0)
            self._plate("Pylon", chord, h + 0.1, min(0.65 / chord, 0.2), (x0, y, z_top - 0.2),
                        rot=(90.0, 0.0, 0.0), mirror=True)

    def _fairings(self, P):
        for i, it in enumerate(P.get("_fairings") or []):
            y = abs(it["y"]); xle, ch, zc, th = wing_at(P, y)
            self._body(f"TrackFairing_{i + 1}", it["length"], FAIR_PROF, it["width"],
                       (it["x"], y, zc - 0.5 * th - 0.35 * it["width"]), mirror=True)

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

    def _details(self, P):
        """Control surfaces (flaps, slats, ailerons, elevator, rudder) and cockpit/passenger windows as sub-surfaces."""
        v = self.v
        ids = {v.GetGeomName(g): g for g in v.FindGeoms()}
        ctrl = getattr(v, "SS_CONTROL", None); rect = getattr(v, "SS_RECTANGLE", None)
        n_ctrl = n_win = n_made = 0
        if ctrl is not None:
            for gname, specs in (("Wing", WING_CONTROLS), ("HTail", TAIL_CONTROLS), ("VTail", RUDDER)):
                for name, u0, u1, cf, le in specs:
                    if gname in ids:
                        n_made += 1
                        n_ctrl += self._subsurface(ids[gname], ctrl, name, {
                            "ustart": u0, "uend": u1, "lengthcstart": cf, "lengthcend": cf, "leflag": 1.0 if le else 0.0})
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
            v.ExportFile(str(t), SETI, v.EXPORT_STL)
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

    def export_stl(self, path):
        self.v.Update()
        t = self._tmp("export.stl")
        t.unlink(missing_ok=True)
        self.v.ExportFile(str(t), self.v.SET_ALL, self.v.EXPORT_STL)
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


def extract_mask(img_path, flip=False):
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
        sil = ((lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) * 255).astype(np.uint8)
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
    exp = {"top": P["fuselage.length"]["value"], "side": P["fuselage.length"]["value"] * 54.08 / 53.85,
           "front": P["wing.span"]["value"]}
    out = []
    for v, c in V.items():
        L = c.get("length_m")
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


def cmd_calibrate(a):
    import cv2
    HOME.mkdir(exist_ok=True)
    for d in DIRS: (HOME / d).mkdir(exist_ok=True)
    src = a.image
    mask = imread(a.mask, 0) if a.mask else extract_mask(src, a.flip)
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
           "params": P, "fuselage": jload(FUSE_F), "fairings": jload(FAIR_F)})
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

def fit_fuselage(P, N=31, M=400, spacing=None, c2=None, tang=None, ends=None):
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
    sm = lambda a: savgol_filter(median_filter(a, 11, mode="nearest"), win, 3, mode="interp")
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
    jsave(FUSE_F, {"source": "silhouette masks (smoothed); ellipse sections",
                   "evidence_class": "calculated", "mode": {"spacing": spacing, "c2": c2, "tang": tang, "ends": ends, "stations": N},
                   "stations": st})
    print(f"  fuselage: {N} smooth stations ({spacing} spacing, C2 {'on' if c2 else 'off'}, tangents {tang}, {ends} ends) (max width {W.max():.2f} m, max height {H.max():.2f} m)")
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
    fit_fuselage(P, a.stations, spacing=spacing, c2=False, tang=tang, ends=ends)
    commit(P, f"fuselage test -> {spacing}, tangents {tang}, {ends} ends", [], reason="best fuselage construction by silhouette error")


def cmd_fuselage_fit(a):
    require_calibration()
    P = jload(PARAMS)
    fit_fuselage(P, a.stations, spacing=a.spacing, c2=None if a.c2 is None else a.c2 == "on", tang=a.tang,
                 ends=a.ends)
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


def cmd_restore_mask(a):
    """Put back parts that erase-mask removed (e.g. the engines), choosing only the rectangles you select."""
    import cv2
    V = jload(VIEWS_F, {})
    if a.view not in V: sys.exit(f"view {a.view} is not calibrated yet")
    c = V[a.view]; cur = imread(c["mask"], 0)
    full_p = Path(c["mask"]).with_name(f"{a.view}_mask_full.png")
    if not full_p.exists(): sys.exit("nothing was erased in this view (no *_mask_full.png)")
    full = imread(full_p, 0)
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
    print(f"{a.view}: restored {len(rects)} region(s); bbox {c['bbox']}")


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

def run_optimize(free, W, iters=80, method="Powell", allow_measured=False, wide=1.0, skip_fixed=False, commit_now=True):
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
    def f(x):
        A.build(apply(x))
        return evaluate(A.export_stl(tmp), refs, W, Lref)[0]
    with fast_tess():                                         # coarse mesh while searching; fine mesh when saving
        A.build(values(P)); before_t, before = evaluate(A.export_stl(tmp), refs, W, Lref)
        res = minimize(f, x0, method=method, bounds=[(0, 1)] * len(x0),
                       options={"maxfev": iters, "xtol": 5e-3, "ftol": 1e-4} if method == "Powell"
                       else {"maxiter": iters})
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

def cmd_auto(a):
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
    print("Calibrated views:", sorted(calibrated))
    has_prop = float(P0.get("model.has_propulsion", {"value": 0.0})["value"]) >= 0.5
    print("Nacelles/gear in model:", has_prop, "| saved tess_scale:", _get_tess_scale())
    if not has_prop:
        print("  (run `nacelle-fit` once to add the nacelles if the masks contain the engines)")
    last = None
    for p in range(1, a.passes + 1):
        print(f"=== pass {p}/{a.passes} ===")
        if {"top", "side"} <= calibrated:
            print("- fuselage from silhouettes")
            P0 = jload(PARAMS)
            fit_fuselage(P0, a.stations)
        with fast_tess():
            probe_a = OpenVSPAdapter(); probe_a.build(values(jload(PARAMS)))
        thick_ok = getattr(probe_a, "thickness_ok", True)
        stages = list(STAGES) + ([PROP_STAGE] if has_prop else [])
        for name, free, w in stages:
            if not thick_ok and any(k.endswith(".thickness") for k in free):
                print(f"- {name}: skipped (thickness parameter not available)"); continue
            W = {v: (w.get(v, 0.0) if v in calibrated else 0.0) for v in VIEWS}
            if sum(W.values()) == 0:
                print(f"- {name}: skipped (needs a calibrated view)"); continue
            print(f"- {name}")
            run_optimize(free, W, a.iters, "Powell", False, a.wide, skip_fixed=True, commit_now=False)
        if has_prop:
            P1 = jload(PARAMS); sync_engine(P1); jsave(PARAMS, P1)
        commit(jload(PARAMS), f"auto pass {p}", [], reason="stages of one automatic pass")
        W = {v: 1.0 for v in VIEWS if v in calibrated}
        refs, P = load_refs(EVAL_DS), jload(PARAMS)
        with fast_tess():
            A = OpenVSPAdapter(); A.build(values(P))
            tot = evaluate(A.export_stl(HOME / "renders" / "opt.stl"), refs, W, P["fuselage.length"]["value"])[0]
        print(f"  overall error after pass {p}: {tot:.5f}")
        if last is not None and abs(last - tot) < 0.01 * last:
            print("  converged (<1% change)"); break
        last = tot
    cmd_compare(a)
    print("Done.")

def cmd_smooth(a):
    _set_tess_scale(a.scale)
    P = jload(PARAMS)
    if a.stations and {"top", "side"} <= set(jload(VIEWS_F, {})):
        fit_fuselage(P, a.stations)
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
    """Flap-track fairings (the 'points' behind the wing trailing edge): measured from the plan-view residual."""
    require_calibration()
    P = jload(PARAMS)
    with fast_tess():
        A = OpenVSPAdapter(); A.build({**values(P), "_fairings": None})
        tris = A.export_stl(HOME / "renders" / "opt.stl")
    c, bl = residual_blobs(P, "top", tris, 0.10, a.min_area, a.max_area)
    if not c:
        sys.exit("calibrate the top view first")
    x0, x1, y0, y1 = c["bbox"]; row_c = (y0 + y1) / 2.0; ppm = c["ppm"]
    Pv = values(P); half = Pv["wing.span"] / 2.0
    def plausible(b):                                          # thin, short, just behind the trailing edge, on the wing
        xm, ym = (b["bx"] - x0) / ppm, (b["cy"] - row_c) / ppm
        w_, l_ = b["bh"] / ppm, b["bw"] / ppm
        xle, ch, _, _ = wing_at(Pv, ym)
        return 3.0 <= ym <= 0.97 * half and 0.25 <= w_ <= 1.3 and l_ <= 4.5 and abs(xm - (xle + ch)) <= 2.2
    bl = sorted([b for b in bl if b["cy"] > row_c and plausible(b)], key=lambda b: b["bx"])[: a.max_count]   # +Y wing only
    items = [{"x": round((b["bx"] - x0) / ppm - a.under, 3), "y": round((b["by"] + b["bh"] / 2.0 - row_c) / ppm, 3),
              "length": round(b["bw"] / ppm + a.under, 3), "width": round(min(b["bh"] / ppm, 1.6), 3)} for b in bl]
    if not items:
        sys.exit("no fairing-sized blobs found (try --min-area 0.1)")
    jsave(FAIR_F, {"source": "plan-view residual", "evidence_class": "calculated", "items": items})
    for it in items:
        print("  fairing", it)
    commit(P, f"{len(items)} track fairings from the plan view", [], reason="flap-track fairings measured from the top mask")

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
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest").set_defaults(fn=cmd_selftest)
    s = sp.add_parser("check-env"); s.add_argument("--vsp-path"); s.set_defaults(fn=cmd_check_env)
    sp.add_parser("check-var-comp").set_defaults(fn=cmd_check_var_comp)
    sp.add_parser("init").set_defaults(fn=cmd_init)
    sp.add_parser("migrate").set_defaults(fn=cmd_migrate)
    s = sp.add_parser("calibrate"); s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--image", required=True); s.add_argument("--length", type=float, required=True)
    s.add_argument("--mask"); s.add_argument("--flip", action="store_true"); s.set_defaults(fn=cmd_calibrate)
    s = sp.add_parser("landmarks"); s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--image", required=True); s.add_argument("--mask")
    s.add_argument("--flip", action="store_true"); s.set_defaults(fn=cmd_landmarks)
    s = sp.add_parser("erase-mask"); s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--rect", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"))
    s.add_argument("--zoom", type=float, default=0.5); s.set_defaults(fn=cmd_erase_mask)
    s = sp.add_parser("auto"); s.add_argument("--passes", type=int, default=2)
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
    s = sp.add_parser("fuselage-test", help="try uniform/cosine spacing and C2 on/off, keep the best")
    s.add_argument("--stations", type=int, default=31); s.set_defaults(fn=cmd_fuselage_test)
    s = sp.add_parser("fuselage-fit"); s.add_argument("--stations", type=int, default=31)
    s.add_argument("--spacing", choices=["uniform", "cosine"]); s.add_argument("--c2", choices=["on", "off"])
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
    s = sp.add_parser("restore-mask", help="put the engines back into a mask (inverse of erase-mask)")
    s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--rect", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"))
    s.add_argument("--zoom", type=float, default=0.5); s.set_defaults(fn=cmd_restore_mask)
    s = sp.add_parser("fairings-fit", help="measure the flap-track fairings from the plan view and add them")
    s.add_argument("--min-area", type=float, default=0.25); s.add_argument("--max-area", type=float, default=4.0)
    s.add_argument("--max-count", type=int, default=8)
    s.add_argument("--under", type=float, default=1.5, help="metres the fairing extends forward under the wing")
    s.set_defaults(fn=cmd_fairings_fit)
    sp.add_parser("probe-subsurf", help="list the parameters of control-surface / window sub-surfaces").set_defaults(fn=cmd_probe_subsurf)
    sp.add_parser("timing", help="time the parts of one evaluation").set_defaults(fn=cmd_timing)
    sp.add_parser("report-metrics").set_defaults(fn=cmd_report_metrics)
    sp.add_parser("inspect", help="diagnose a broken model: component sizes, parameters, calibration").set_defaults(fn=cmd_inspect)
    s = sp.add_parser("rescale", help="fix the scale of a calibrated view without redoing its mask")
    s.add_argument("--view", choices=list(VIEWS), required=True)
    s.add_argument("--length", type=float, required=True); s.set_defaults(fn=cmd_rescale)
    sp.add_parser("probe").set_defaults(fn=cmd_probe)
    s = sp.add_parser("probe-geom"); s.add_argument("--type", required=True)
    s.add_argument("--all", action="store_true"); s.set_defaults(fn=cmd_probe_geom)
    s = sp.add_parser("history"); s.add_argument("--last", type=int, help="only the last N versions")
    s.set_defaults(fn=cmd_history)
    s = sp.add_parser("diff"); s.add_argument("a", type=int, nargs="?"); s.add_argument("b", type=int, nargs="?")
    s.set_defaults(fn=cmd_diff)
    sp.add_parser("undo").set_defaults(fn=cmd_undo)
    s = sp.add_parser("checkout", help="restore version N (a whole number, e.g. `checkout 18`)")
    s.add_argument("n", type=int, help="version number from `history`"); s.set_defaults(fn=cmd_checkout)
    a = ap.parse_args(); a.fn(a)

if __name__ == "__main__":
    main()
