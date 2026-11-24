"""vrma_io - read a VRM Animation (.vrma) including humanoid bones AND facial expression tracks
(VRMC_vrm_animation.expressions).

Pure Python + numpy. Works in desktop Python and browser Pyodide.
"""
# SPDX-License-Identifier: MIT
import numpy as np

import vrm_io as V

FPS = 30.0
LAST_DIAG = {}


def _qrot(q, v):
    u = np.array(q[:3])
    return v + 2 * np.cross(u, np.cross(u, v) + q[3] * v)


def _slerp(a, b, f):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + (b - a) * f
        return r / np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1 - f) * th) * a + np.sin(f * th) * b) / np.sin(th)


def _sample(times, vals, t, interp="LINEAR"):
    n_t = len(times)
    if interp == "CUBICSPLINE" and len(vals) == 3 * n_t:
        tri = vals.reshape(n_t, 3, -1)
        if t <= times[0]:
            out = tri[0, 1]
        elif t >= times[-1]:
            out = tri[-1, 1]
        else:
            i = int(np.searchsorted(times, t)) - 1
            dt = max(times[i + 1] - times[i], 1e-8)
            s = (t - times[i]) / dt
            s2 = s * s
            s3 = s2 * s
            h00 = 2.0 * s3 - 3.0 * s2 + 1.0
            h10 = s3 - 2.0 * s2 + s
            h01 = -2.0 * s3 + 3.0 * s2
            h11 = s3 - s2
            out = h00 * tri[i, 1] + h10 * dt * tri[i, 2] + h01 * tri[i + 1, 1] + h11 * dt * tri[i + 1, 0]
        if out.shape[-1] == 4:
            nrm = float(np.linalg.norm(out))
            if nrm > 1e-8:
                out = out / nrm
        return out
    if t <= times[0]:
        return vals[0]
    if t >= times[-1]:
        return vals[-1]
    i = int(np.searchsorted(times, t)) - 1
    if interp == "STEP":
        return vals[i]
    f = (t - times[i]) / max(times[i + 1] - times[i], 1e-8)
    return _slerp(vals[i], vals[i + 1], f) if vals.shape[1] == 4 else vals[i] * (1 - f) + vals[i + 1] * f


def read_vrma(data, bones):
    """Return (D (F, len(bones), 4), hips (F, 3), exprs (F, 13) or None, seconds, name)."""
    global LAST_DIAG
    j, binc = V.read_glb(data)
    ext = j.get("extensions", {}).get("VRMC_vrm_animation")
    if not ext:
        raise ValueError("not a VRM Animation (no VRMC_vrm_animation)")
    hb = {k: v["node"] for k, v in ext.get("humanoid", {}).get("humanBones", {}).items()}
    if "hips" not in hb:
        raise ValueError("VRMA: no hips bone")
    anims = j.get("animations") or []
    if not anims:
        raise ValueError("no animation in this file")

    expr_nodes = []
    raw_keys, matched_keys, unmatched_keys = [], [], []
    expr_ext = ext.get("expressions") or {}
    for gkey in ("preset", "custom"):
        grp = expr_ext.get(gkey) or {}
        if not isinstance(grp, dict):
            continue
        for ename, spec in grp.items():
            raw_keys.append(str(ename))
            if isinstance(spec, (int, np.integer)):
                n_idx = int(spec)
            elif isinstance(spec, dict) and "node" in spec:
                n_idx = int(spec["node"])
            else:
                continue
            ei = V.resolve_expr_idx(ename)
            if ei is not None:
                expr_nodes.append((ei, n_idx))
                matched_keys.append(f"{ename}->{V.EXPRESSION_NAMES[ei]}")
            else:
                unmatched_keys.append(str(ename))

    nodes = j["nodes"]
    N = len(nodes)
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    order, seen = [], set()

    def visit(i):
        if i in seen:
            return
        if i in parent:
            visit(parent[i])
        seen.add(i)
        order.append(i)

    for i in range(N):
        visit(i)
    T0 = [np.array(n.get("translation", (0, 0, 0)), float) for n in nodes]
    R0 = [np.array(n.get("rotation", (0, 0, 0, 1)), float) for n in nodes]

    def worlds(Ts, Rs):
        WR, WP = [None] * N, [None] * N
        for i in order:
            p = parent.get(i)
            if p is None:
                WR[i], WP[i] = Rs[i], Ts[i]
            else:
                WR[i] = V.qmul(WR[p], Rs[i])
                WP[i] = WP[p] + _qrot(WR[p], Ts[i])
        return WR, WP

    rWR, rWP = worlds(T0, R0)
    anim = anims[0]
    tracks, dur, interps = {}, 0.0, set()
    for ch in anim.get("channels", []):
        tgt = ch.get("target", {})
        if "node" not in tgt or "path" not in tgt:
            continue
        s = anim["samplers"][ch["sampler"]]
        interp = s.get("interpolation", "LINEAR")
        interps.add(interp)
        t = V.accessor(j, binc, s["input"]).reshape(-1).astype(float)
        v = V.accessor(j, binc, s["output"]).astype(float)
        tracks.setdefault(tgt["node"], {})[tgt["path"]] = (t, v, interp)
        dur = max(dur, float(t[-1]))

    if len(anims) > 1 and expr_nodes:
        expr_nset = {ni for _, ni in expr_nodes}
        for extra_anim in anims[1:]:
            for ch in extra_anim.get("channels", []):
                tgt = ch.get("target", {})
                ni = tgt.get("node")
                path = tgt.get("path")
                if ni in expr_nset and path == "translation" and path not in tracks.get(ni, {}):
                    s = extra_anim["samplers"][ch["sampler"]]
                    interp = s.get("interpolation", "LINEAR")
                    interps.add(interp)
                    t = V.accessor(j, binc, s["input"]).reshape(-1).astype(float)
                    v = V.accessor(j, binc, s["output"]).astype(float)
                    tracks.setdefault(ni, {})[path] = (t, v, interp)
                    dur = max(dur, float(t[-1]))

    F = max(2, int(round(dur * FPS)) + 1)
    D = np.zeros((F, len(bones), 4), np.float32)
    D[..., 3] = 1.0
    hips = np.zeros((F, 3), np.float32)
    exprs = np.zeros((F, len(V.EXPRESSION_NAMES)), np.float32) if expr_nodes else None
    hip_h = rWP[hb["hips"]][1] or 1.0
    feet = [hb[b] for b in ("leftFoot", "rightFoot", "leftToes", "rightToes") if b in hb]
    rest_foot = min(rWP[f][1] for f in feet) if feet else 0.0
    lowest = 1e9

    for k in range(F):
        t = k / FPS
        Ts, Rs = list(T0), list(R0)
        for i, tr in tracks.items():
            if "translation" in tr:
                tt, tv, ti = tr["translation"]
                Ts[i] = _sample(tt, tv, t, ti)
            if "rotation" in tr:
                rt, rv, ri = tr["rotation"]
                q = _sample(rt, rv, t, ri)
                Rs[i] = q / np.linalg.norm(q)
        WR, WP = worlds(Ts, Rs)
        for bi, b in enumerate(bones):
            n = hb.get(b)
            if n is not None:
                D[k, bi] = V.qmul(WR[n], V.qconj(rWR[n]))
        if exprs is not None:
            for ei, ni in expr_nodes:
                w = float(np.clip(Ts[ni][0], 0.0, 1.0))
                if w > exprs[k, ei]:
                    exprs[k, ei] = w
        d = WP[hb["hips"]] - rWP[hb["hips"]]
        d[0] = d[2] = 0.0
        hips[k] = d / hip_h
        if feet:
            lowest = min(lowest, min(WP[f][1] for f in feet) - rest_foot - (WP[hb["hips"]][1] - rWP[hb["hips"]][1]) + d[1])
    if feet:
        hips[:, 1] += max(0.0, -lowest) / hip_h

    max_weights = {}
    if exprs is not None:
        for ei, ename in enumerate(V.EXPRESSION_NAMES):
            mx = float(exprs[:, ei].max())
            if mx > 1e-4:
                max_weights[ename] = round(mx, 3)
        if float(exprs.max()) <= 1e-4:
            exprs = None

    LAST_DIAG = {
        "has_expr_ext": bool(expr_ext),
        "raw_keys": raw_keys,
        "matched_keys": matched_keys,
        "unmatched_keys": unmatched_keys,
        "interps": sorted(interps),
        "n_anims": len(anims),
        "max_weights": max_weights,
    }
    return D, hips, exprs, (F - 1) / FPS, anim.get("name", "")
