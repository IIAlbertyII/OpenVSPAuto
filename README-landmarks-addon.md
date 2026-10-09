# AI Aircraft Reconstruction — Automatic Landmarks Add-on

This version preserves the existing `aircraft_recon.py` architecture and commands.
It adds one command only:

    python aircraft_recon.py landmarks --view top --image drawings\top.png

Also available for `side` and `front`.

Optional:
    --mask FILE    cleaned white-on-black aircraft silhouette
    --flip         mirror the drawing horizontally

Outputs are written under the existing project/processed/ directory:

    project/processed/top_landmarks.json
    project/processed/top_landmarks.png

The automatic detector produces landmark *candidates*. It does not modify
params.json, views.json, OpenVSP geometry, or measured dimensions.

This is intentional: candidates have confidence/method metadata and must be
verified before becoming measured constraints.

The existing commands remain unchanged:
selftest, check-env, init, calibrate, build, compare, optimize, set, delta,
patch, history, undo, checkout.

Recommended workflow:

1. Keep a backup of your current aircraft_recon.py.
2. Replace it with the supplied aircraft_recon.py.
3. Run selftest.
4. Run landmarks for top/side/front.
5. Inspect the *_landmarks.png files.
6. Use cleaned masks with --mask when dimension lines/text interfere.
7. Do not convert candidates into measured dimensions automatically.
