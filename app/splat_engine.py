"""splat_engine - Rigged 3DGS + Facial Morph Targets (GPU skinning + GPU blendshapes),
face-safe Mesh-to-Gaussian sampler, and .ply / .splat / .zip / standalone WebGL2 HTML exporters.

Pure Python + NumPy (+ stdlib). Works in desktop Pyxel 3 and in the browser (Pyodide).
"""
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
import base64
import io
import json
import math
import struct
import zipfile
import zlib

import numpy as np

import vrm_io as V

SH_C0 = 0.28209479177387814


def _norm(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _rotmat_to_quat_wxyz(R):
    """Vectorized (N, 3, 3) proper rotation matrices -> unit quaternions (N, 4) in (w, x, y, z)."""
    n = len(R)
    q = np.empty((n, 4), dtype=np.float32)
    r00, r01, r02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r10, r11, r12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r20, r21, r22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    tr = r00 + r11 + r22
    m0 = tr > 0.0
    m1 = (~m0) & (r00 >= r11) & (r00 >= r22)
    m2 = (~m0) & (~m1) & (r11 >= r22)
    m3 = (~m0) & (~m1) & (~m2)

    if m0.any():
        s = np.sqrt(np.maximum(tr[m0] + 1.0, 1e-12)) * 2.0
        q[m0, 0] = 0.25 * s
        q[m0, 1] = (r21[m0] - r12[m0]) / s
        q[m0, 2] = (r02[m0] - r20[m0]) / s
        q[m0, 3] = (r10[m0] - r01[m0]) / s
    if m1.any():
        s = np.sqrt(np.maximum(1.0 + r00[m1] - r11[m1] - r22[m1], 1e-12)) * 2.0
        q[m1, 0] = (r21[m1] - r12[m1]) / s
        q[m1, 1] = 0.25 * s
        q[m1, 2] = (r01[m1] + r10[m1]) / s
        q[m1, 3] = (r02[m1] + r20[m1]) / s
    if m2.any():
        s = np.sqrt(np.maximum(1.0 + r11[m2] - r00[m2] - r22[m2], 1e-12)) * 2.0
        q[m2, 0] = (r02[m2] - r20[m2]) / s
        q[m2, 1] = (r01[m2] + r10[m2]) / s
        q[m2, 2] = 0.25 * s
        q[m2, 3] = (r12[m2] + r21[m2]) / s
    if m3.any():
        s = np.sqrt(np.maximum(1.0 + r22[m3] - r00[m3] - r11[m3], 1e-12)) * 2.0
        q[m3, 0] = (r10[m3] - r01[m3]) / s
        q[m3, 1] = (r02[m3] + r20[m3]) / s
        q[m3, 2] = (r12[m3] + r21[m3]) / s
        q[m3, 3] = 0.25 * s

    q = _norm(q)
    flip = q[:, 0] < 0
    q[flip] = -q[flip]
    return q


def _quat_of(M):
    R = M[:3, :3] / np.maximum(np.linalg.norm(M[:3, :3], axis=0), 1e-8)
    return V.mat_to_quat(R)


def _qrot(q, v):
    u = q[:3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[3] * v)


def auto_blink_weights(t, available_exprs):
    """Smooth procedural blink pulse (~every 3.6s) when a VRM has blink morphs and motion lacks expressions."""
    w_vec = np.zeros(len(V.EXPRESSION_NAMES), dtype=np.float32)
    if not available_exprs:
        return w_vec
    ph = float(t) % 3.6
    if ph < 0.16:
        val = math.sin(math.pi * ph / 0.16)
        if 0 in available_exprs:
            w_vec[0] = val
        else:
            if 1 in available_exprs:
                w_vec[1] = val
            if 2 in available_exprs:
                w_vec[2] = val
    return w_vec


# ====================================================================== Motions
class Motions:
    """Per-humanoid-bone world rotation delta from T-pose (30 fps), hips offset, and optional expression tracks."""

    def __init__(self, path):
        d = np.load(path)
        self.bones = [str(b) for b in d["bones"]]
        self.meta = json.loads(bytes(d["meta"]).decode())
        self.by_id = {m["id"]: m for m in self.meta}
        self.D = {m["id"]: d["d_" + m["id"]].astype(np.float32) for m in self.meta}
        self.Hp = {m["id"]: d["h_" + m["id"]].astype(np.float32) for m in self.meta}
        self.Expr = {m["id"]: None for m in self.meta}

    def add(self, label, D, Hp, group="Your motions", loop=True, Expr=None, source="user", fixed_id=None, prepend=False):
        if fixed_id:
            mid = str(fixed_id)
        else:
            base = "my_" + "".join(ch for ch in label.lower().replace(" ", "_") if ch.isalnum() or ch == "_")[:40]
            if base == "my_":
                base = "my_motion"
            mid, k = base, 2
            while mid in self.by_id:
                mid, k = "%s_%d" % (base, k), k + 1
        has_expr = bool(Expr is not None and float(np.max(Expr)) > 1e-4)
        m = dict(
            id=mid,
            label=label,
            group=group,
            source=source,
            loop=bool(loop),
            has_expr=has_expr,
            frames=len(D),
            seconds=round((len(D) - 1) / 30.0, 3),
        )
        if prepend:
            self.meta.insert(0, m)
        else:
            self.meta.append(m)
        self.by_id[mid] = m
        self.D[mid] = D.astype(np.float32)
        self.Hp[mid] = Hp.astype(np.float32)
        self.Expr[mid] = Expr.astype(np.float32) if has_expr else None
        return mid

    def frame(self, mid, t):
        D, Hp, m = self.D[mid], self.Hp[mid], self.by_id[mid]
        n = len(D)
        if n <= 1:
            return D[0], Hp[0]
        f = t * 30.0
        f = f % (n - 1) if m["loop"] else min(max(0.0, f), n - 1.0001)
        i = int(f)
        u = f - i
        a, b = D[i], D[min(i + 1, n - 1)]
        b = np.where((np.sum(a * b, 1) < 0)[:, None], -b, b)
        q = a + (b - a) * u
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
        return q, Hp[i] + (Hp[min(i + 1, n - 1)] - Hp[i]) * u

    def expr_frame(self, mid, t):
        E = self.Expr.get(mid)
        m = self.by_id.get(mid)
        if E is None or m is None:
            return None
        n = len(E)
        if n <= 1:
            return E[0]
        f = t * 30.0
        f = f % (n - 1) if m["loop"] else min(max(0.0, f), n - 1.0001)
        i = int(f)
        u = f - i
        return (E[i] + (E[min(i + 1, n - 1)] - E[i]) * u).astype(np.float32)


# ====================================================================== Built-in Fallback Model
class ProceduralModel:
    """Minimal multi-box animated character when no VRM is loaded."""

    def __init__(self):
        self.version = "Builtin"
        self.title = "Retro Bot"
        self.author = "vrm-retro-4dgs"
        self.license = ["BUILTIN CC0 TEST MODEL"]
        self.warnings = []
        self.face_yaw = 0.0
        self.height = 1.65
        self.expression_names = list(V.EXPRESSION_NAMES)
        self.available_expressions = []
        self.has_expressions = False
        self.human = {
            "hips": 0, "spine": 1, "head": 2,
            "leftUpperArm": 3, "rightUpperArm": 4,
            "leftUpperLeg": 5, "rightUpperLeg": 6,
        }
        self.names = ["hips", "torso", "head", "l_arm", "r_arm", "l_leg", "r_leg"]
        self.parent = np.array([-1, 0, 1, 1, 1, 0, 0], np.int64)
        self.order = np.arange(7, dtype=np.int64)
        self.rest_t = np.array([
            [0.0, 0.85, 0.0], [0.0, 0.25, 0.0], [0.0, 0.34, 0.0],
            [0.24, 0.16, 0.0], [-0.24, 0.16, 0.0],
            [0.10, -0.35, 0.0], [-0.10, -0.35, 0.0],
        ], np.float32)
        self.rest_q = np.tile(np.array([0, 0, 0, 1], np.float32), (7, 1))
        self.rest_s = np.ones((7, 3), np.float32)
        self.rest_m = self.local_mats(self.rest_q, self.rest_t)
        self.rest_world = self.world(self.rest_m)

        pal = np.array([
            [118, 150, 222], [112, 198, 169], [233, 195, 91],
            [212, 24, 108], [238, 238, 238], [57, 92, 152],
        ], np.uint8)
        self.palette = np.vstack([pal, np.tile(pal[-1:], (239 - len(pal), 1))])
        self.rgba, self.tex_index = [], []
        for k, col in enumerate(pal):
            a = np.zeros((4, 4, 4), np.uint8)
            a[..., :3] = col
            a[..., 3] = 255
            self.rgba.append(a)
            self.tex_index.append(np.full((4, 4), 16 + k, np.uint8))

        specs = [
            (0, (0.14, 0.09, 0.10), 5), (1, (0.16, 0.22, 0.11), 0),
            (2, (0.14, 0.14, 0.14), 4), (3, (0.06, 0.24, 0.06), 1),
            (4, (0.06, 0.24, 0.06), 1), (5, (0.07, 0.28, 0.07), 2),
            (6, (0.07, 0.28, 0.07), 3),
        ]
        self.parts = []
        self.tris = 0
        for ni, (sx, sy, sz), tex in specs:
            pos, uv, idx = _box_mesh(sx, sy, sz)
            n = len(pos)
            ww = np.zeros((n, 4), np.float32)
            ww[:, 0] = 1.0
            slot = np.zeros((n, 4), np.int64)
            p1 = np.hstack([pos, np.ones((n, 1), np.float32)])
            wpos = (self.rest_world[ni].astype(np.float32) @ p1.T)[:3].T.astype(np.float32)
            self.parts.append(dict(
                pos=wpos, uv=uv, idx=idx, slot=slot, ww=ww,
                skin_nodes=np.array([ni], np.int64),
                ibm=np.linalg.inv(self.rest_world[ni]).astype(np.float32)[None],
                tex=tex, name=self.names[ni], alpha="OPAQUE", double=False, keyed=False,
                morphs={},
            ))
            self.tris += len(idx) // 3
        self.verts = sum(len(p["pos"]) for p in self.parts)

    def world(self, local):
        W = np.empty_like(local)
        for i in self.order:
            p = self.parent[i]
            W[i] = local[i] if p < 0 else W[p] @ local[i]
        return W

    def local_mats(self, q, t=None):
        t = self.rest_t if t is None else t
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        M = np.zeros((len(q), 4, 4), np.float64)
        M[:, 0, 0] = 1 - 2 * (y * y + z * z)
        M[:, 0, 1] = 2 * (x * y - z * w)
        M[:, 0, 2] = 2 * (x * z + y * w)
        M[:, 1, 0] = 2 * (x * y + z * w)
        M[:, 1, 1] = 1 - 2 * (x * x + z * z)
        M[:, 1, 2] = 2 * (y * z - x * w)
        M[:, 2, 0] = 2 * (x * z - y * w)
        M[:, 2, 1] = 2 * (y * z + x * w)
        M[:, 2, 2] = 1 - 2 * (x * x + y * y)
        M[:, :3, 3] = t
        M[:, 3, 3] = 1
        return M


def _box_mesh(hx, hy, hz):
    faces = [
        ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)),
        ((hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)),
        ((-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)),
        ((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)),
        ((hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)),
        ((-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)),
    ]
    pos, uv, idx = [], [], []
    for k, (a, b, c, d) in enumerate(faces):
        base = k * 4
        pos.extend([a, b, c, d])
        uv.extend([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return np.array(pos, np.float32), np.array(uv, np.float32), np.array(idx, np.int64)


# ====================================================================== RigState
class RigState:
    """Evaluates bone poses, skinned+morphed meshes, and deduplicated global joint matrices."""

    def __init__(self, model, motions):
        self.m = model
        self.motions = motions
        self.p1 = [np.hstack([p["pos"], np.ones((len(p["pos"]), 1), np.float32)]) for p in model.parts]
        self.rest_wq = np.array([_quat_of(M) for M in model.rest_world])
        self.bone_nodes = [model.human.get(b, -1) for b in motions.bones]
        self.node_bone = {n: k for k, n in enumerate(self.bone_nodes) if n >= 0}
        hips = model.human.get("hips")
        self.hips = hips
        self.hip_h = float(model.rest_world[hips][1, 3]) if hips is not None else 1.0
        self.flip = model.version == "VRM 0.x"
        self.has_humanoid = hips is not None
        self.yaw_sign = -1.0 if model.face_yaw == 180.0 else 1.0

        uniq = {}
        j_nodes, j_ibms, part_maps = [], [], []
        for p in model.parts:
            pmap = np.empty(len(p["skin_nodes"]), dtype=np.int32)
            for s_idx, (ni, ibm) in enumerate(zip(p["skin_nodes"], p["ibm"])):
                key = (int(ni), np.round(ibm, 4).tobytes())
                gid = uniq.get(key)
                if gid is None:
                    gid = len(j_nodes)
                    uniq[key] = gid
                    j_nodes.append(int(ni))
                    j_ibms.append(ibm.astype(np.float32))
                pmap[s_idx] = gid
            part_maps.append(pmap)
        self.joint_nodes = np.array(j_nodes, dtype=np.int64) if j_nodes else np.zeros(1, dtype=np.int64)
        self.joint_ibms = np.stack(j_ibms, axis=0) if j_ibms else np.eye(4, dtype=np.float32)[None]
        self.part_joint_maps = part_maps
        self.num_joints = len(self.joint_nodes)

    def pose_at(self, motion_id, t):
        m = self.m
        q = m.rest_q.astype(np.float64).copy()
        tr = m.rest_t.copy()
        if motion_id != "tpose" and self.has_humanoid and motion_id in self.motions.by_id:
            D, hp = self.motions.frame(motion_id, t)
            if self.flip:
                D = D * np.array([-1.0, 1.0, -1.0, 1.0], np.float32)
                hp = hp * np.array([-1.0, 1.0, -1.0], np.float32)
            Wq = {}
            one = np.array([0.0, 0.0, 0.0, 1.0])
            for i in m.order:
                p = m.parent[i]
                pw = Wq.get(p) if p >= 0 else None
                if pw is None:
                    pw = self.rest_wq[p] if p >= 0 else one
                k = self.node_bone.get(int(i))
                if k is not None:
                    w = V.qmul(D[k], self.rest_wq[i])
                    q[i] = V.qmul(V.qconj(pw), w)
                    Wq[i] = w
                else:
                    Wq[i] = V.qmul(pw, q[i])
            p = m.parent[self.hips]
            pw = Wq.get(p, one) if p >= 0 else one
            tr[self.hips] = m.rest_t[self.hips] + _qrot(V.qconj(pw), hp * self.hip_h)
        return q, tr

    def joint_matrices(self, q, tr, apply_face_yaw=True):
        """Return (J, 3, 4) float32 world skinning matrices matching global joint IDs."""
        local = self.m.local_mats(q, tr)
        world = self.m.world(local).astype(np.float32)
        mats = (world[self.joint_nodes] @ self.joint_ibms)[:, :3, :].copy()
        if apply_face_yaw and self.yaw_sign < 0.0:
            mats[:, 0, :] = -mats[:, 0, :]
            mats[:, 2, :] = -mats[:, 2, :]
        return mats

    def skin_parts(self, q, tr, apply_face_yaw=True, expr_w=None):
        """Return list of (V_i, 3) float32 vertex arrays for each mesh part, applying optional expr_w."""
        local = self.m.local_mats(q, tr)
        world = self.m.world(local).astype(np.float32)
        out = []
        for p, p1 in zip(self.m.parts, self.p1):
            mats = (world[p["skin_nodes"]] @ p["ibm"])[:, :3, :]
            p_in = p1
            morphs = p.get("morphs")
            if expr_w is not None and morphs:
                delta = None
                for ei, d_arr in morphs.items():
                    w_e = float(expr_w[ei])
                    if w_e > 1e-4:
                        add_d = w_e * d_arr
                        delta = add_d.copy() if delta is None else (delta + add_d)
                if delta is not None:
                    p_in = p1.copy()
                    p_in[:, :3] += delta
            pos = np.zeros((len(p_in), 3), np.float32)
            for k in range(4):
                w = p["ww"][:, k]
                if not w.any():
                    continue
                pos += w[:, None] * np.einsum("nij,nj->ni", mats[p["slot"][:, k]], p_in)
            if apply_face_yaw and self.yaw_sign < 0.0:
                pos[:, 0] = -pos[:, 0]
                pos[:, 2] = -pos[:, 2]
            out.append(pos)
        return out


# ====================================================================== Splat Sampler & Face-Safe Culling
@dataclass
class SplatSampler:
    count: int
    part_idx: np.ndarray     # (N,) int32
    v0: np.ndarray           # (N,) int32
    v1: np.ndarray           # (N,) int32
    v2: np.ndarray           # (N,) int32
    w0: np.ndarray           # (N,) float32
    w1: np.ndarray           # (N,) float32
    w2: np.ndarray           # (N,) float32
    rest_area: np.ndarray    # (N,) float32
    base_radius: np.ndarray  # (N,) float32
    rgb: np.ndarray          # (N, 3) uint8
    alpha: np.ndarray        # (N,) uint8
    pal_idx: np.ndarray      # (N,) uint8 (16..254)
    bind_pos: np.ndarray     # (N, 3) float32
    bind_u: np.ndarray       # (N, 3) float32
    bind_v: np.ndarray       # (N, 3) float32
    b0: np.ndarray           # (N,) uint16
    b1: np.ndarray           # (N,) uint16
    bw0: np.ndarray          # (N,) uint8
    is_face: np.ndarray      # (N,) bool
    morph_splats: int        # K (first K splats carry morph deltas)
    morph_expr_ids: list     # canonical expr indices present on sampled splats
    morph_deltas: np.ndarray # (E, K, 3) float32 in bind space
    culled_pct: float = 0.0


@dataclass
class SplatFrame:
    centers: np.ndarray    # (N, 3) float32
    u_vec: np.ndarray      # (N, 3) float32
    v_vec: np.ndarray      # (N, 3) float32
    normals: np.ndarray    # (N, 3) float32
    scales: np.ndarray     # (N, 3) float32
    quat_wxyz: np.ndarray  # (N, 4) float32


@dataclass
class BakedRigged3DGS:
    title: str
    motion_id: str
    motion_label: str
    count: int
    frames: int
    fps: float
    num_joints: int
    bbox_min: np.ndarray         # (3,) float32
    bbox_max: np.ndarray         # (3,) float32
    rgb: np.ndarray              # (N, 3) uint8
    alpha: np.ndarray            # (N,) uint8
    pal_idx: np.ndarray          # (N,) uint8
    bind_pos: np.ndarray         # (N, 3) float32
    bind_u: np.ndarray           # (N, 3) float32
    bind_v: np.ndarray           # (N, 3) float32
    b0: np.ndarray               # (N,) uint16
    b1: np.ndarray               # (N,) uint16
    bw0: np.ndarray              # (N,) uint8
    bone_mats: np.ndarray        # (F, J, 3, 4) float32
    morph_splats: int
    morph_expr_ids: list
    morph_deltas: np.ndarray     # (M, K, 3) float32
    expr_weights: np.ndarray     # (F, M) float32
    expr_names: list
    compressed_r3dg: bytes
    license_lines: list
    warnings: list
    author: str
    version: str


FACIAL_EXTERIOR_ROLES = ("face_skin", "eye_white", "eye_iris", "eye_detail", "face_mouth")


def _part_role(part_name, mouth_exterior=False):
    """Classify VRM mesh part so exterior face/eyes/hair/clothes stay intact while interior body_skin is culled."""
    nm = (part_name or "").lower()
    if any(k in nm for k in ("mouth", "teeth", "tongue", "oral")):
        return "face_mouth" if mouth_exterior else "mouth"
    if any(k in nm for k in ("highlight", "eyelash", "lash", "eyeline", "brow")):
        return "eye_detail"
    if "iris" in nm:
        return "eye_iris"
    if any(k in nm for k in ("eyewhite", "sclera", "eye")):
        return "eye_white"
    if "face" in nm:
        return "face_skin"
    if "hair" in nm:
        return "hair"
    if any(k in nm for k in ("cloth", "tops", "bottoms", "skirt", "dress", "pants", "shirt", "jacket", "coat", "shoe", "boot", "sock", "acc", "tie", "ribbon", "hat", "glove")):
        return "cloth"
    if any(k in nm for k in ("body", "skin")):
        return "body_skin"
    return "other"


def _nearest_palette_idx(rgb, palette):
    P = palette.astype(np.float32)
    out = np.empty(len(rgb), dtype=np.uint8)
    for s in range(0, len(rgb), 1024):
        chunk = rgb[s:s + 1024].astype(np.float32)
        d = ((chunk[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        out[s:s + 1024] = np.clip(d.argmin(1) + 16, 16, 254).astype(np.uint8)
    return out


def _sample_tex_attrs(m, uv, tex_ids):
    n = len(uv)
    rgb = np.empty((n, 3), dtype=np.uint8)
    alpha = np.empty(n, dtype=np.uint8)
    pal_idx = np.empty(n, dtype=np.uint8)
    for ti in np.unique(tex_ids):
        mask = tex_ids == ti
        img = m.rgba[ti]
        idx_img = m.tex_index[ti]
        th, tw = img.shape[:2]
        uu = np.mod(uv[mask, 0], 1.0)
        vv = np.mod(uv[mask, 1], 1.0)
        ix = np.clip((uu * tw).astype(np.int32), 0, tw - 1)
        iy = np.clip((vv * th).astype(np.int32), 0, th - 1)
        px = img[iy, ix]
        rgb[mask] = px[:, :3]
        alpha[mask] = px[:, 3]
        pal_idx[mask] = idx_img[iy, ix]
    return rgb, alpha, pal_idx


def _exterior_visible_mask(rest_parts, t_part, t_v, chosen, w0, w1, w2, height):
    """15-view orthographic Z-buffer culling so occluded interior layers (under clothes/skirts) are removed."""
    m = len(chosen)
    pts = np.empty((m, 3), dtype=np.float32)
    p_idx = t_part[chosen]
    v0_i, v1_i, v2_i = t_v[chosen, 0], t_v[chosen, 1], t_v[chosen, 2]
    for pi, verts in enumerate(rest_parts):
        mask = p_idx == pi
        if not mask.any():
            continue
        pts[mask] = (
            w0[mask, None] * verts[v0_i[mask]]
            + w1[mask, None] * verts[v1_i[mask]]
            + w2[mask, None] * verts[v2_i[mask]]
        )

    c = 0.5 * (pts.min(0) + pts.max(0))
    p0 = pts - c[None, :]
    span = max(0.1, float(np.max(pts.max(0) - pts.min(0))))
    G = 160
    eps = 0.006 * max(0.3, float(height))
    visible = np.zeros(m, dtype=bool)
    views = (
        [(deg, 4.0) for deg in range(0, 360, 45)]
        + [(deg, 42.0) for deg in (0.0, 90.0, 180.0, 270.0)]
        + [(45.0, 68.0), (225.0, 68.0), (0.0, 82.0)]
    )

    for yaw_deg, pitch_deg in views:
        ya, pa = math.radians(yaw_deg), math.radians(pitch_deg)
        cy, sy = math.cos(ya), math.sin(ya)
        cp, sp = math.cos(pa), math.sin(pa)
        rx, ry, rz = cy, 0.0, -sy
        fx, fy, fz = sy * cp, sp, cy * cp
        ux, uy, uz = -sy * sp, cp, -cy * sp
        vx = p0[:, 0] * rx + p0[:, 1] * ry + p0[:, 2] * rz
        vy = p0[:, 0] * ux + p0[:, 1] * uy + p0[:, 2] * uz
        vz = -(p0[:, 0] * fx + p0[:, 1] * fy + p0[:, 2] * fz)
        gx = np.clip(((vx / span + 0.55) / 1.1 * G).astype(np.int32), 0, G - 1)
        gy = np.clip(((vy / span + 0.55) / 1.1 * G).astype(np.int32), 0, G - 1)
        bid = gy * G + gx
        zbuf = np.full(G * G, 1e9, dtype=np.float32)
        np.minimum.at(zbuf, bid, vz)
        z2 = zbuf.reshape(G, G)
        zmin = z2.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sy0, sy1 = max(0, dy), G + min(0, dy)
                dy0, dy1 = max(0, -dy), G + min(0, -dy)
                sx0, sx1 = max(0, dx), G + min(0, dx)
                dx0, dx1 = max(0, -dx), G + min(0, -dx)
                zmin[dy0:dy1, dx0:dx1] = np.minimum(zmin[dy0:dy1, dx0:dx1], z2[sy0:sy1, sx0:sx1])
        visible |= (vz <= zmin.reshape(-1)[bid] + eps)
    return visible


def _compute_bind_frames_and_bones(rig, p_idx, v0, v1, v2, w0, w1, w2, base_radius):
    """Compute raw bind-pose center/tangent basis and top-2 global bone weights per splat."""
    m = rig.m
    n = len(p_idx)
    raw0 = np.empty((n, 3), dtype=np.float32)
    raw1 = np.empty((n, 3), dtype=np.float32)
    raw2 = np.empty((n, 3), dtype=np.float32)

    all_g = np.zeros((n, 12), dtype=np.int32)
    all_w = np.zeros((n, 12), dtype=np.float32)

    for pi, p in enumerate(m.parts):
        mask = p_idx == pi
        if not mask.any():
            continue
        pos = p["pos"]
        i0, i1, i2 = v0[mask], v1[mask], v2[mask]
        raw0[mask] = pos[i0]
        raw1[mask] = pos[i1]
        raw2[mask] = pos[i2]

        pmap = rig.part_joint_maps[pi]
        gslots = pmap[p["slot"]]
        ww = p["ww"]
        all_g[mask, 0:4] = gslots[i0]
        all_g[mask, 4:8] = gslots[i1]
        all_g[mask, 8:12] = gslots[i2]
        all_w[mask, 0:4] = w0[mask, None] * ww[i0]
        all_w[mask, 4:8] = w1[mask, None] * ww[i1]
        all_w[mask, 8:12] = w2[mask, None] * ww[i2]

    bind_pos = (w0[:, None] * raw0 + w1[:, None] * raw1 + w2[:, None] * raw2).astype(np.float32)
    e1 = raw1 - raw0
    e2 = raw2 - raw0
    normal = _norm(np.cross(e1, e2))
    t1 = _norm(e1)
    t1 = t1 - np.sum(t1 * normal, axis=1, keepdims=True) * normal
    bad = np.linalg.norm(t1, axis=1) < 1e-5
    if bad.any():
        alt = np.where(
            np.abs(normal[bad, 0:1]) < 0.9,
            np.array([1.0, 0.0, 0.0], np.float32),
            np.array([0.0, 1.0, 0.0], np.float32),
        )
        t1[bad] = np.cross(normal[bad], alt)
    t1 = _norm(t1)
    t2 = _norm(np.cross(normal, t1))
    bind_u = (t1 * base_radius[:, None]).astype(np.float32)
    bind_v = (t2 * base_radius[:, None]).astype(np.float32)

    has_w = all_w > 1e-6
    rows = np.arange(n)
    best0_slot = np.argmax(all_w, axis=1)
    b0 = all_g[rows, best0_slot].astype(np.uint16)
    w_top0 = np.sum(np.where((all_g == b0[:, None]) & has_w, all_w, 0.0), axis=1)

    rem_w = np.where((all_g != b0[:, None]) & has_w, all_w, 0.0)
    best1_slot = np.argmax(rem_w, axis=1)
    b1 = all_g[rows, best1_slot].astype(np.uint16)
    w_top1 = np.sum(np.where((all_g == b1[:, None]) & has_w, rem_w, 0.0), axis=1)

    no_second = w_top1 <= 1e-6
    b1[no_second] = b0[no_second]
    tot = np.maximum(w_top0 + w_top1, 1e-6)
    bw0_f = np.where(no_second, 1.0, w_top0 / tot)
    bw0 = np.clip(np.round(bw0_f * 255.0), 0, 255).astype(np.uint8)
    return bind_pos, bind_u, bind_v, b0, b1, bw0


def _allocate_facial_budgets(facial_infos, count):
    """Allocate dedicated high-res splat counts per exterior facial part (capped at 26% of count)."""
    if not facial_infos:
        return {}
    by_role = {r: [] for r in FACIAL_EXTERIOR_ROLES}
    for info in facial_infos:
        by_role[info["role"]].append(info)

    budgets = {}

    for role, frac in (("face_skin", 0.11), ("eye_white", 0.020), ("eye_iris", 0.030), ("face_mouth", 0.015)):
        items = by_role[role]
        if not items:
            continue
        role_tot = max(len(items) * 32, int(round(count * frac)))
        tot_a = sum(x["area"] for x in items)
        for x in items:
            budgets[x["pi"]] = max(16, int(round(role_tot * x["area"] / max(tot_a, 1e-10))))

    det_items = by_role["eye_detail"]
    if det_items:
        det_cap = max(len(det_items) * 120, int(round(count * 0.060)))
        floor_each = max(120, int(count * 0.008))
        sqrt_a = [math.sqrt(max(x["area"], 1e-10)) for x in det_items]
        sum_sq = max(sum(sqrt_a), 1e-10)
        raw_alloc = [max(floor_each, int(round(det_cap * s / sum_sq))) for s in sqrt_a]
        tot_det = sum(raw_alloc)
        if tot_det > det_cap:
            scale = det_cap / float(tot_det)
            raw_alloc = [max(48, int(round(k * scale))) for k in raw_alloc]
        for x, k in zip(det_items, raw_alloc):
            budgets[x["pi"]] = k

    max_facial = int(count * 0.26)
    tot_facial = sum(budgets.values())
    if tot_facial > max_facial and tot_facial > 0:
        scale = max_facial / float(tot_facial)
        for pi in list(budgets.keys()):
            budgets[pi] = max(16, int(round(budgets[pi] * scale)))
    return budgets


def build_sampler(rig, count, seed=42, cull_interior=True):
    """Build face-safe surface sampler with exterior body/skirt culling and per-splat morph deltas."""
    m = rig.m
    rest_parts = rig.skin_parts(m.rest_q.astype(np.float64), m.rest_t, apply_face_yaw=True)
    head_idx = m.human.get("head")
    head_y_cut = float(m.rest_world[head_idx][1, 3] - 0.08 * m.height) if head_idx is not None else 1e9
    mouth_has_expr = bool(
        getattr(m, "has_expressions", False)
        and (
            any(ei in m.available_expressions for ei in (3, 4, 5, 6, 7))
            or any(
                "mouth" in (p.get("name", "").lower()) and bool(p.get("morphs"))
                for p in m.parts
            )
        )
    )

    facial_infos = []
    gen_part, gen_v, gen_area, gen_uv, gen_tex, gen_w, gen_roles = [], [], [], [], [], [], []
    has_body_skin = False

    for pi, (p, pos) in enumerate(zip(m.parts, rest_parts)):
        tri = p["idx"].reshape(-1, 3)
        if len(tri) == 0:
            continue
        a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        keep = area > 1e-10
        if not keep.any():
            continue
        tri = tri[keep]
        area_k = area[keep].astype(np.float64)
        uv_k = p["uv"][tri].astype(np.float32)
        nm = p.get("name", "")
        role = _part_role(nm, mouth_exterior=mouth_has_expr)

        if role in FACIAL_EXTERIOR_ROLES:
            facial_infos.append(dict(
                pi=pi, role=role, part=p, tri=tri.astype(np.int32),
                area_arr=area_k, area=float(area_k.sum()), uv=uv_k,
            ))
        else:
            if role == "body_skin":
                has_body_skin = True
            if role == "hair":
                w_base = 2.2 if "hairback" in nm.lower() else 1.8
            elif role == "cloth":
                w_base = 1.1
            else:
                w_base = 1.0
            cy = (a[keep, 1] + b[keep, 1] + c[keep, 1]) / 3.0
            w_tri = np.where((w_base == 1.0) & (cy >= head_y_cut), 2.5, w_base).astype(np.float64)
            gen_part.append(np.full(len(tri), pi, np.int32))
            gen_v.append(tri.astype(np.int32))
            gen_area.append(area_k)
            gen_uv.append(uv_k)
            gen_tex.append(np.full(len(tri), p["tex"], np.int32))
            gen_w.append(w_tri)
            gen_roles.append((pi, role))

    if not facial_infos and not gen_area:
        raise ValueError("Model has no valid triangles to sample")

    if not gen_area:
        for info in facial_infos:
            tri = info["tri"]
            gen_part.append(np.full(len(tri), info["pi"], np.int32))
            gen_v.append(tri)
            gen_area.append(info["area_arr"])
            gen_uv.append(info["uv"])
            gen_tex.append(np.full(len(tri), info["part"]["tex"], np.int32))
            gen_w.append(np.ones(len(tri), np.float64))
            gen_roles.append((info["pi"], "other"))
        facial_infos = []

    cull_roles = ("body_skin", "mouth") if has_body_skin else ("other", "mouth")
    part_is_cullable = np.zeros(len(m.parts), dtype=bool)
    for pi_g, role_g in gen_roles:
        if role_g in cull_roles:
            part_is_cullable[pi_g] = True

    rng = np.random.default_rng(seed)
    budgets = _allocate_facial_budgets(facial_infos, count)
    facial_total = sum(budgets.values())
    body_count = max(1, count - facial_total)

    chunks = []
    # ---- 1. General body / hair / clothes pool (clothes/hair occlude body_skin & closed mouth)
    t_part = np.concatenate(gen_part)
    t_v = np.concatenate(gen_v, axis=0)
    t_area = np.concatenate(gen_area)
    t_uv = np.concatenate(gen_uv, axis=0)
    t_tex = np.concatenate(gen_tex)
    t_w = np.concatenate(gen_w)

    total_area = float(t_area.sum())
    weighted = t_area * t_w
    mean_w = float(weighted.sum()) / max(total_area, 1e-8)
    probs = weighted / float(weighted.sum())

    n_cand = max(int(body_count * (3.0 if cull_interior else 1.4)) + 512, 56000 if cull_interior else body_count)
    chosen = rng.choice(len(t_area), size=n_cand, replace=True, p=probs)
    u1 = rng.random(n_cand, dtype=np.float32)
    u2 = rng.random(n_cand, dtype=np.float32)
    r1 = np.sqrt(u1)
    w0 = (1.0 - r1).astype(np.float32)
    w1 = (r1 * (1.0 - u2)).astype(np.float32)
    w2 = (r1 * u2).astype(np.float32)

    uv = w0[:, None] * t_uv[chosen, 0] + w1[:, None] * t_uv[chosen, 1] + w2[:, None] * t_uv[chosen, 2]
    rgb, alpha, pal_idx = _sample_tex_attrs(m, uv, t_tex[chosen])

    opaque_mask = alpha >= 128
    culled_pct = 0.0
    vis_ratio = 1.0
    if cull_interior and opaque_mask.any():
        vis_mask = _exterior_visible_mask(rest_parts, t_part, t_v, chosen, w0, w1, w2, m.height)
        vis_mask |= ~part_is_cullable[t_part[chosen]]
        valid_mask = opaque_mask & vis_mask
        vis_ratio = float(valid_mask.sum()) / max(1.0, float(opaque_mask.sum()))
        culled_pct = max(0.0, (1.0 - vis_ratio) * 100.0)
        valid = np.flatnonzero(valid_mask)
        if len(valid) < max(32, body_count // 8):
            valid = np.flatnonzero(opaque_mask)
            vis_ratio = 1.0
            culled_pct = 0.0
    else:
        valid = np.flatnonzero(opaque_mask)

    if len(valid) == 0:
        valid = np.arange(n_cand)
    if len(valid) >= body_count:
        sel = valid[:body_count]
    else:
        rep = int(math.ceil(body_count / len(valid)))
        sel = np.tile(valid, rep)[:body_count]

    chosen_b = chosen[sel]
    eff_area = max(total_area * vis_ratio, 1e-4)
    w_sel = t_w[chosen_b].astype(np.float32)
    r_body = math.sqrt(eff_area * mean_w / max(1, body_count) / math.pi) * 2.15
    r_arr = np.clip(r_body / np.sqrt(np.maximum(w_sel, 0.5)), 0.0032, 0.068).astype(np.float32)

    chunks.append(dict(
        part_idx=t_part[chosen_b],
        v0=t_v[chosen_b, 0],
        v1=t_v[chosen_b, 1],
        v2=t_v[chosen_b, 2],
        w0=w0[sel],
        w1=w1[sel],
        w2=w2[sel],
        rest_area=t_area[chosen_b].astype(np.float32),
        base_radius=r_arr,
        rgb=rgb[sel],
        alpha=np.maximum(alpha[sel], 240).astype(np.uint8),
        pal_idx=pal_idx[sel],
        is_face=np.zeros(body_count, dtype=bool),
    ))

    # ---- 2. Dedicated facial exterior parts (NEVER culled by bangs/hat shadow)
    radius_rules = {
        "face_skin": (1.65, 0.0030, 0.0170),
        "eye_white": (1.50, 0.0016, 0.0090),
        "eye_iris": (1.42, 0.00090, 0.0050),
        "eye_detail": (1.20, 0.00060, 0.0034),
        "face_mouth": (1.15, 0.00080, 0.0040),
    }

    for info in facial_infos:
        k_part = budgets.get(info["pi"], 0)
        if k_part <= 0:
            continue
        tri = info["tri"]
        area_arr = info["area_arr"]
        uv_arr = info["uv"]
        p = info["part"]
        role = info["role"]

        p_prob = area_arr / max(float(area_arr.sum()), 1e-12)
        n_try = max(k_part * 10, 4096)
        f_ch = rng.choice(len(area_arr), size=n_try, replace=True, p=p_prob)
        fu1 = rng.random(n_try, dtype=np.float32)
        fu2 = rng.random(n_try, dtype=np.float32)
        fr1 = np.sqrt(fu1)
        fw0 = (1.0 - fr1).astype(np.float32)
        fw1 = (fr1 * (1.0 - fu2)).astype(np.float32)
        fw2 = (fr1 * fu2).astype(np.float32)

        f_uv = fw0[:, None] * uv_arr[f_ch, 0] + fw1[:, None] * uv_arr[f_ch, 1] + fw2[:, None] * uv_arr[f_ch, 2]
        f_tex = np.full(n_try, p["tex"], np.int32)
        f_rgb, f_alpha, f_pal = _sample_tex_attrs(m, f_uv, f_tex)

        is_blend = p["alpha"] == "BLEND"
        valid_mask = (f_alpha >= 48) if is_blend else (f_alpha >= 128)
        f_valid = np.flatnonzero(valid_mask)
        op_frac = max(0.05, float(valid_mask.mean()))
        if len(f_valid) == 0:
            f_valid = np.arange(n_try)
            op_frac = 1.0
        if len(f_valid) >= k_part:
            f_sel = f_valid[:k_part]
        else:
            f_rep = int(math.ceil(k_part / len(f_valid)))
            f_sel = np.tile(f_valid, f_rep)[:k_part]

        f_ch_s = f_ch[f_sel]
        alpha_s = f_alpha[f_sel]
        if is_blend:
            alpha_s = np.clip(np.round(alpha_s.astype(np.float32) * 1.15), 64, 255).astype(np.uint8)
        else:
            alpha_s = np.maximum(alpha_s, 240).astype(np.uint8)

        eff_part_area = info["area"] * op_frac
        d_avg = math.sqrt(max(eff_part_area, 1e-10) / max(1, k_part))
        mul, rmin, rmax = radius_rules[role]
        r_part = float(np.clip(d_avg * mul, rmin, rmax))

        chunks.append(dict(
            part_idx=np.full(k_part, info["pi"], np.int32),
            v0=tri[f_ch_s, 0],
            v1=tri[f_ch_s, 1],
            v2=tri[f_ch_s, 2],
            w0=fw0[f_sel],
            w1=fw1[f_sel],
            w2=fw2[f_sel],
            rest_area=area_arr[f_ch_s].astype(np.float32),
            base_radius=np.full(k_part, r_part, np.float32),
            rgb=f_rgb[f_sel],
            alpha=alpha_s,
            pal_idx=f_pal[f_sel],
            is_face=np.ones(k_part, dtype=bool),
        ))

    p_idx = np.concatenate([c["part_idx"] for c in chunks])
    v0_s = np.concatenate([c["v0"] for c in chunks])
    v1_s = np.concatenate([c["v1"] for c in chunks])
    v2_s = np.concatenate([c["v2"] for c in chunks])
    w0_s = np.concatenate([c["w0"] for c in chunks])
    w1_s = np.concatenate([c["w1"] for c in chunks])
    w2_s = np.concatenate([c["w2"] for c in chunks])
    rest_area = np.concatenate([c["rest_area"] for c in chunks])
    base_radius = np.concatenate([c["base_radius"] for c in chunks])
    rgb_s = np.concatenate([c["rgb"] for c in chunks], axis=0)
    alpha_s = np.concatenate([c["alpha"] for c in chunks])
    pal_s = np.concatenate([c["pal_idx"] for c in chunks])
    is_face_s = np.concatenate([c["is_face"] for c in chunks])

    bad_pal = pal_s >= 255
    if bad_pal.any():
        pal_s[bad_pal] = _nearest_palette_idx(rgb_s[bad_pal], m.palette)
    pal_s = np.clip(pal_s, 16, 254).astype(np.uint8)

    # Interpolate bind-space morph deltas & pack all morph-affected splats contiguously at indices 0..K-1
    expr_ids = list(getattr(m, "available_expressions", []) or [])
    n_tot = len(p_idx)
    if expr_ids:
        raw_morphs = np.zeros((len(expr_ids), n_tot, 3), dtype=np.float32)
        ei_to_slot = {ei: si for si, ei in enumerate(expr_ids)}
        for pi, p in enumerate(m.parts):
            pm = p.get("morphs")
            if not pm:
                continue
            mask = p_idx == pi
            if not mask.any():
                continue
            i0, i1, i2 = v0_s[mask], v1_s[mask], v2_s[mask]
            mw0, mw1, mw2 = w0_s[mask, None], w1_s[mask, None], w2_s[mask, None]
            for ei, d_arr in pm.items():
                si = ei_to_slot.get(ei)
                if si is not None:
                    raw_morphs[si, mask] = mw0 * d_arr[i0] + mw1 * d_arr[i1] + mw2 * d_arr[i2]
        morph_mask = np.any(np.abs(raw_morphs) > 1e-6, axis=(0, 2))
        morph_k = int(morph_mask.sum())
        if morph_k > 0:
            perm = np.concatenate([np.flatnonzero(morph_mask), np.flatnonzero(~morph_mask)])
            p_idx = p_idx[perm]
            v0_s, v1_s, v2_s = v0_s[perm], v1_s[perm], v2_s[perm]
            w0_s, w1_s, w2_s = w0_s[perm], w1_s[perm], w2_s[perm]
            rest_area = rest_area[perm]
            base_radius = base_radius[perm]
            rgb_s, alpha_s, pal_s = rgb_s[perm], alpha_s[perm], pal_s[perm]
            is_face_s = is_face_s[perm]
            morph_deltas = np.ascontiguousarray(raw_morphs[:, perm[:morph_k], :], dtype=np.float32)
        else:
            expr_ids = []
            morph_deltas = np.zeros((0, 0, 3), dtype=np.float32)
    else:
        morph_k = 0
        morph_deltas = np.zeros((0, 0, 3), dtype=np.float32)

    bind_pos, bind_u, bind_v, b0, b1, bw0 = _compute_bind_frames_and_bones(
        rig, p_idx, v0_s, v1_s, v2_s, w0_s, w1_s, w2_s, base_radius
    )

    return SplatSampler(
        count=n_tot,
        part_idx=p_idx,
        v0=v0_s,
        v1=v1_s,
        v2=v2_s,
        w0=w0_s,
        w1=w1_s,
        w2=w2_s,
        rest_area=rest_area,
        base_radius=base_radius,
        rgb=rgb_s,
        alpha=alpha_s,
        pal_idx=pal_s,
        bind_pos=bind_pos,
        bind_u=bind_u,
        bind_v=bind_v,
        b0=b0,
        b1=b1,
        bw0=bw0,
        is_face=is_face_s,
        morph_splats=morph_k,
        morph_expr_ids=expr_ids,
        morph_deltas=morph_deltas,
        culled_pct=culled_pct,
    )


def select_preview_indices(sampler, cap=6000):
    """Stratified preview selection preserving up to 30% face/eye splats for crisp Pyxel preview."""
    n = sampler.count
    if n <= cap:
        return np.arange(n, dtype=np.int64), np.ones((n, 1), dtype=np.float32)

    face_i = np.flatnonzero(sampler.is_face)
    body_i = np.flatnonzero(~sampler.is_face)
    if len(face_i) == 0 or len(body_i) == 0:
        idx = np.linspace(0, n - 1, cap, dtype=np.int64)
        boost = float(math.sqrt(max(1.0, n / max(1, cap))))
        return idx, np.full((len(idx), 1), boost, dtype=np.float32)

    n_face = min(len(face_i), int(cap * 0.30))
    n_body = max(1, cap - n_face)
    pf = face_i[np.linspace(0, len(face_i) - 1, n_face, dtype=np.int64)]
    pb = body_i[np.linspace(0, len(body_i) - 1, min(len(body_i), n_body), dtype=np.int64)]

    face_boost = min(1.6, float(math.sqrt(max(1.0, len(face_i) / max(1, len(pf))))))
    body_boost = float(math.sqrt(max(1.0, len(body_i) / max(1, len(pb)))))
    p_idx = np.concatenate([pb, pf])
    p_boost = np.concatenate([
        np.full(len(pb), body_boost, dtype=np.float32),
        np.full(len(pf), face_boost, dtype=np.float32),
    ])[:, None]
    return p_idx, p_boost


def eval_splat_frame(skinned_parts, sampler, full_rot=True, subset=None):
    """Evaluate deformed surfel Gaussians (all N splats or a fast preview subset)."""
    if subset is None:
        p_idx, v0, v1, v2 = sampler.part_idx, sampler.v0, sampler.v1, sampler.v2
        w0, w1, w2 = sampler.w0, sampler.w1, sampler.w2
        rest_area, base_radius = sampler.rest_area, sampler.base_radius
    else:
        p_idx, v0, v1, v2 = sampler.part_idx[subset], sampler.v0[subset], sampler.v1[subset], sampler.v2[subset]
        w0, w1, w2 = sampler.w0[subset], sampler.w1[subset], sampler.w2[subset]
        rest_area, base_radius = sampler.rest_area[subset], sampler.base_radius[subset]

    n = len(p_idx)
    p0 = np.empty((n, 3), dtype=np.float32)
    p1 = np.empty((n, 3), dtype=np.float32)
    p2 = np.empty((n, 3), dtype=np.float32)

    for pi in range(len(skinned_parts)):
        mask = p_idx == pi
        if not mask.any():
            continue
        verts = skinned_parts[pi]
        p0[mask] = verts[v0[mask]]
        p1[mask] = verts[v1[mask]]
        p2[mask] = verts[v2[mask]]

    centers = (w0[:, None] * p0 + w1[:, None] * p1 + w2[:, None] * p2).astype(np.float32)
    e1 = p1 - p0
    e2 = p2 - p0
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(area2[:, None], 1e-8)

    deg = area2 <= 1e-8
    if deg.any():
        normal[deg] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    t1 = _norm(e1)
    t1 = t1 - np.sum(t1 * normal, axis=1, keepdims=True) * normal
    bad_t1 = np.linalg.norm(t1, axis=1) < 1e-5
    if bad_t1.any():
        alt = np.where(
            np.abs(normal[bad_t1, 0:1]) < 0.9,
            np.array([1.0, 0.0, 0.0], np.float32),
            np.array([0.0, 1.0, 0.0], np.float32),
        )
        t1[bad_t1] = np.cross(normal[bad_t1], alt)
    t1 = _norm(t1)
    t2 = _norm(np.cross(normal, t1))

    cur_area = 0.5 * area2
    stretch = np.clip(np.sqrt(np.maximum(cur_area, 1e-10) / np.maximum(rest_area, 1e-10)), 0.5, 2.0)
    r = (base_radius * stretch).astype(np.float32)

    u_vec = (t1 * r[:, None]).astype(np.float32)
    v_vec = (t2 * r[:, None]).astype(np.float32)
    scales = np.stack([r, r, (r * 0.22).astype(np.float32)], axis=1).astype(np.float32)

    if full_rot:
        R = np.stack([t1, t2, normal], axis=2)
        quat_wxyz = _rotmat_to_quat_wxyz(R)
    else:
        quat_wxyz = np.empty((0, 4), dtype=np.float32)

    return SplatFrame(
        centers=centers,
        u_vec=u_vec,
        v_vec=v_vec,
        normals=normal.astype(np.float32),
        scales=scales,
        quat_wxyz=quat_wxyz,
    )


def decode_rigged_frame(baked, k):
    """Evaluate all N Gaussians at frame k via expression morphs + 2-bone LBS."""
    k = int(k) % baked.frames
    mats = baked.bone_mats[k]
    w0 = baked.bw0.astype(np.float32) / 255.0
    M = w0[:, None, None] * mats[baked.b0] + (1.0 - w0[:, None, None]) * mats[baked.b1]
    R = M[:, :, :3]

    if baked.morph_splats > 0 and len(baked.expr_names) > 0:
        bp = baked.bind_pos.copy()
        ew = baked.expr_weights[k]
        bp[:baked.morph_splats] += np.tensordot(ew, baked.morph_deltas, axes=(0, 0))
    else:
        bp = baked.bind_pos

    centers = (np.einsum("nij,nj->ni", R, bp) + M[:, :, 3]).astype(np.float32)
    u_raw = np.einsum("nij,nj->ni", R, baked.bind_u)
    v_raw = np.einsum("nij,nj->ni", R, baked.bind_v)

    sx = np.maximum(np.linalg.norm(u_raw, axis=1), 1e-6)
    sy = np.maximum(np.linalg.norm(v_raw, axis=1), 1e-6)
    t1 = u_raw / sx[:, None]
    nrm = _norm(np.cross(t1, v_raw))
    t2 = _norm(np.cross(nrm, t1))
    quat_wxyz = _rotmat_to_quat_wxyz(np.stack([t1, t2, nrm], axis=2))
    scales = np.stack([sx, sy, (sx * 0.22).astype(np.float32)], axis=1).astype(np.float32)

    return SplatFrame(
        centers=centers,
        u_vec=(t1 * sx[:, None]).astype(np.float32),
        v_vec=(t2 * sy[:, None]).astype(np.float32),
        normals=nrm.astype(np.float32),
        scales=scales,
        quat_wxyz=quat_wxyz,
    )


def decode_rigged_preview(baked, k, subset, boost):
    """Fast 2-bone LBS + morph preview decode for Pyxel subset P <= 6000 splats."""
    k = int(k) % baked.frames
    mats = baked.bone_mats[k]
    w0 = baked.bw0[subset].astype(np.float32) / 255.0
    M = w0[:, None, None] * mats[baked.b0[subset]] + (1.0 - w0[:, None, None]) * mats[baked.b1[subset]]
    R = M[:, :, :3]

    bp_sub = baked.bind_pos[subset]
    if baked.morph_splats > 0 and len(baked.expr_names) > 0:
        m_sub = subset < baked.morph_splats
        if m_sub.any():
            bp_sub = bp_sub.copy()
            ew = baked.expr_weights[k]
            bp_sub[m_sub] += np.tensordot(ew, baked.morph_deltas[:, subset[m_sub], :], axes=(0, 0))

    centers = (np.einsum("nij,nj->ni", R, bp_sub) + M[:, :, 3]).astype(np.float32)
    u_vec = (np.einsum("nij,nj->ni", R, baked.bind_u[subset]) * boost).astype(np.float32)
    v_vec = (np.einsum("nij,nj->ni", R, baked.bind_v[subset]) * boost).astype(np.float32)
    return centers, u_vec, v_vec


def bake_rigged_generator(rig, sampler, motion_id, fps=24, max_frames=900, preview_cap=6000, auto_blink=True, expr_override=None):
    """Chunked generator baking bone matrices (F, J, 3, 4) and facial expression weights (F, M)."""
    m_meta = rig.motions.by_id.get(motion_id)
    if m_meta is None:
        sec = 1.0
        loop = True
        n_frames = max(2, min(max_frames, int(fps)))
    else:
        sec = float(m_meta.get("seconds", 1.0))
        loop = bool(m_meta.get("loop", True))
        n_frames = max(2, min(max_frames, int(round(sec * fps))))

    actual_fps = ((n_frames / sec) if loop else (max(1, n_frames - 1) / sec)) if sec > 1e-4 else float(fps)

    n = sampler.count
    J = rig.num_joints
    bone_mats = np.empty((n_frames, J, 3, 4), dtype=np.float32)
    ew_full = np.zeros((n_frames, len(V.EXPRESSION_NAMES)), dtype=np.float32)

    bb_idx = np.linspace(0, n - 1, min(n, 512), dtype=np.int64)
    bb_pos = sampler.bind_pos[bb_idx]
    bb_b0 = sampler.b0[bb_idx]
    bb_b1 = sampler.b1[bb_idx]
    bb_w0 = (sampler.bw0[bb_idx].astype(np.float32) / 255.0)[:, None, None]

    bmin = np.full(3, 1e9, dtype=np.float32)
    bmax = np.full(3, -1e9, dtype=np.float32)
    bb_step = max(1, n_frames // 20)
    yield_chunk = max(1, n_frames // 25)
    avail_exprs = list(getattr(rig.m, "available_expressions", []) or [])

    for k in range(n_frames):
        t = (sec * k / n_frames) if loop else (sec * k / max(1, n_frames - 1))
        q, tr = rig.pose_at(motion_id, t)
        mats = rig.joint_matrices(q, tr, apply_face_yaw=True)
        bone_mats[k] = mats

        if expr_override is not None and 0 <= int(expr_override) < len(V.EXPRESSION_NAMES):
            ew = np.zeros(len(V.EXPRESSION_NAMES), dtype=np.float32)
            ew[int(expr_override)] = 1.0
        else:
            ew = rig.motions.expr_frame(motion_id, t)
            if ew is None and auto_blink and avail_exprs and motion_id != "tpose":
                ew = auto_blink_weights(t, avail_exprs)
        if ew is not None:
            ew_full[k] = ew

        if k == 0 or k == n_frames - 1 or (k % bb_step == 0):
            M = bb_w0 * mats[bb_b0] + (1.0 - bb_w0) * mats[bb_b1]
            pts = np.einsum("nij,nj->ni", M[:, :, :3], bb_pos) + M[:, :, 3]
            bmin = np.minimum(bmin, pts.min(0))
            bmax = np.maximum(bmax, pts.max(0))
        if ((k + 1) % yield_chunk == 0) and (k + 1 < n_frames):
            yield k + 1, n_frames, None

    # Compact to only expressions active in this motion AND present on sampled splats (max 16)
    active_slots = []
    if sampler.morph_splats > 0 and sampler.morph_expr_ids:
        for s_idx, ei in enumerate(sampler.morph_expr_ids):
            if float(ew_full[:, ei].max()) > 1e-3:
                active_slots.append((s_idx, ei))
    active_slots = active_slots[:16]

    if active_slots:
        s_idx_list = [s for s, _ in active_slots]
        e_idx_list = [ei for _, ei in active_slots]
        b_deltas = np.ascontiguousarray(sampler.morph_deltas[s_idx_list], dtype=np.float32)
        b_weights = np.ascontiguousarray(ew_full[:, e_idx_list], dtype=np.float32)
        b_expr_ids = e_idx_list
        b_expr_names = [V.EXPRESSION_NAMES[ei] for ei in e_idx_list]
        b_morph_splats = int(sampler.morph_splats)
    else:
        b_deltas = np.zeros((0, 0, 3), dtype=np.float32)
        b_weights = np.zeros((n_frames, 0), dtype=np.float32)
        b_expr_ids = []
        b_expr_names = []
        b_morph_splats = 0

    label = "T-pose" if motion_id == "tpose" else (m_meta.get("label", motion_id) if m_meta else motion_id)
    baked = BakedRigged3DGS(
        title=rig.m.title,
        motion_id=motion_id,
        motion_label=label,
        count=n,
        frames=n_frames,
        fps=float(actual_fps),
        num_joints=J,
        bbox_min=bmin,
        bbox_max=bmax,
        rgb=sampler.rgb.copy(),
        alpha=sampler.alpha.copy(),
        pal_idx=sampler.pal_idx.copy(),
        bind_pos=sampler.bind_pos.copy(),
        bind_u=sampler.bind_u.copy(),
        bind_v=sampler.bind_v.copy(),
        b0=sampler.b0.copy(),
        b1=sampler.b1.copy(),
        bw0=sampler.bw0.copy(),
        bone_mats=bone_mats,
        morph_splats=b_morph_splats,
        morph_expr_ids=b_expr_ids,
        morph_deltas=b_deltas,
        expr_weights=b_weights,
        expr_names=b_expr_names,
        compressed_r3dg=b"",
        license_lines=list(rig.m.license),
        warnings=list(rig.m.warnings),
        author=rig.m.author,
        version=rig.m.version,
    )
    baked.compressed_r3dg = export_r3dg_bin(baked)
    yield n_frames, n_frames, baked


# ====================================================================== Exporters
def _to_ydown(centers, quat_wxyz):
    """Rotate Y-up (VRM/Pyxel) Gaussians by Rx(180 deg) to Y-down (COLMAP / 3DGS external viewers)."""
    pos = centers.astype(np.float32).copy()
    pos[:, 1] = -pos[:, 1]
    pos[:, 2] = -pos[:, 2]
    q = np.stack([
        -quat_wxyz[:, 1],
        quat_wxyz[:, 0],
        -quat_wxyz[:, 3],
        quat_wxyz[:, 2],
    ], axis=1).astype(np.float32)
    flip = q[:, 0] < 0
    q[flip] = -q[flip]
    return pos, q


def export_ply_bytes(sf, rgb, alpha, y_down=True):
    """Standard 3DGS binary little-endian .ply (68 bytes per Gaussian; Y-down by default)."""
    pos, rot = _to_ydown(sf.centers, sf.quat_wxyz) if y_down else (sf.centers, sf.quat_wxyz)
    n = len(pos)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    ).encode("ascii")

    arr = np.empty((n, 17), dtype="<f4")
    arr[:, 0:3] = pos
    arr[:, 3:6] = 0.0
    arr[:, 6:9] = ((rgb.astype(np.float32) / 255.0) - 0.5) / SH_C0
    a = np.clip(alpha.astype(np.float32) / 255.0, 0.02, 0.98)
    arr[:, 9] = np.log(a / (1.0 - a))
    arr[:, 10:13] = np.log(np.maximum(sf.scales, 1e-6))
    arr[:, 13:17] = rot
    return header + arr.tobytes()


def export_splat_bytes(sf, rgb, alpha, y_down=True):
    """Antimatter15 .splat format (32 bytes per Gaussian; Y-down by default)."""
    pos, rot = _to_ydown(sf.centers, sf.quat_wxyz) if y_down else (sf.centers, sf.quat_wxyz)
    n = len(pos)
    dt = np.dtype([
        ("pos", "<f4", 3),
        ("scale", "<f4", 3),
        ("rgba", "u1", 4),
        ("rot", "u1", 4),
    ])
    arr = np.empty(n, dtype=dt)
    arr["pos"] = pos
    arr["scale"] = sf.scales
    arr["rgba"][:, :3] = rgb
    arr["rgba"][:, 3] = alpha
    arr["rot"] = np.clip(np.round(rot * 128.0 + 128.0), 0, 255).astype(np.uint8)
    return arr.tobytes()


def export_r3dg_bin(baked):
    """Compact Rigged 3DGS v2 binary: b'R3DG' + u32 hdr_len + JSON + zlib(splats + bones + quantized morphs)."""
    M = len(baked.expr_names)
    K = int(baked.morph_splats) if M > 0 else 0
    morph_scale = (max(float(np.abs(baked.morph_deltas).max()), 1e-5) / 32767.0) if (M > 0 and K > 0) else 1.0

    hdr = json.dumps({
        "version": 2,
        "format": "rigged-3dgs",
        "title": baked.title,
        "motion": baked.motion_label,
        "count": baked.count,
        "frames": baked.frames,
        "fps": round(float(baked.fps), 4),
        "joints": baked.num_joints,
        "morph_count": M,
        "morph_splats": K,
        "morph_scale": morph_scale,
        "expr_names": list(baked.expr_names),
        "bbox_min": [round(float(x), 5) for x in baked.bbox_min],
        "bbox_max": [round(float(x), 5) for x in baked.bbox_max],
    }, separators=(",", ":")).encode("utf-8")

    raw_parts = [
        np.ascontiguousarray(baked.rgb, np.uint8).tobytes(),
        np.ascontiguousarray(baked.alpha, np.uint8).tobytes(),
        np.ascontiguousarray(baked.bind_pos, "<f4").tobytes(),
        np.ascontiguousarray(baked.bind_u, "<f4").tobytes(),
        np.ascontiguousarray(baked.bind_v, "<f4").tobytes(),
        np.ascontiguousarray(baked.b0, "<u2").tobytes(),
        np.ascontiguousarray(baked.b1, "<u2").tobytes(),
        np.ascontiguousarray(baked.bw0, np.uint8).tobytes(),
        np.ascontiguousarray(baked.bone_mats, "<f4").tobytes(),
    ]
    if M > 0 and K > 0:
        q_deltas = np.clip(np.round(baked.morph_deltas / morph_scale), -32767, 32767).astype("<i2")
        raw_parts.append(np.ascontiguousarray(q_deltas, "<i2").tobytes())
        raw_parts.append(np.ascontiguousarray(baked.expr_weights, "<f4").tobytes())

    comp = zlib.compress(b"".join(raw_parts), level=6)
    return b"R3DG" + struct.pack("<I", len(hdr)) + hdr + comp


def export_4dgs_zip_bytes(baked):
    """Export ZIP with sequence.r3dg, manifest.json, per-frame Y-down .splat sequence (capped at 120), and README.txt."""
    max_splat_frames = 120
    splat_indices = (
        np.linspace(0, baked.frames - 1, max_splat_frames, dtype=np.int64).tolist()
        if baked.frames > max_splat_frames
        else list(range(baked.frames))
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("sequence.r3dg", baked.compressed_r3dg)
        manifest = {
            "format": "vrm-retro-r3dg-v2",
            "title": baked.title,
            "author": baked.author,
            "motion": baked.motion_label,
            "count": baked.count,
            "frames": baked.frames,
            "fps": round(float(baked.fps), 4),
            "joints": baked.num_joints,
            "expressions": list(baked.expr_names),
            "sequence_file": "sequence.r3dg",
            "sequence_bytes": len(baked.compressed_r3dg),
            "splat_frames": [f"frames/frame_{k:04d}.splat" for k in splat_indices],
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for k in splat_indices:
            sf = decode_rigged_frame(baked, k)
            zf.writestr(f"frames/frame_{k:04d}.splat", export_splat_bytes(sf, baked.rgb, baked.alpha, y_down=True))
        readme = [
            "Rigged 3DGS / 4DGS Export from VRM RETRO 4DGS STUDIO",
            f"Model: {baked.title} by {baked.author} ({baked.version})",
            f"Motion: {baked.motion_label} ({baked.frames} frames @ {baked.fps:.2f} fps, {baked.count} Gaussians, {baked.num_joints} bones, {len(baked.expr_names)} expressions)",
            "",
            "Coordinate conventions:",
            "  frames/*.splat  Standard 3DGS Y-down (upright in 3dgsviewers.com, SuperSplat, Antimatter15)",
            "  sequence.r3dg   GPU Rigged 3DGS v2 binary (bones + expression morphs)",
            "",
            "== License ==",
            *[f"  {line}" for line in baked.license_lines],
            *[f"  WARNING: {w}" for w in baked.warnings],
        ]
        zf.writestr("README.txt", "\n".join(readme) + "\n")
    return buf.getvalue()


def _esc_html(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


R3DG_PLAYER_JS = r"""
function createR3DGPlayer(opts) {
  const { canvas, videoEl, photoEl, hudEl, statusEl, scrubEl, fnumEl } = opts;
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: true, premultipliedAlpha: true });
  if (!gl) return null;

  const TEX_W = 1024;
  const SORT_BINS = 4096;
  const MAX_MORPH = 16;
  const vsSplat = `#version 300 es
  in vec2 aCorner;
  in float aSplatId;
  uniform sampler2D uTexPos;
  uniform sampler2D uTexU;
  uniform sampler2D uTexV;
  uniform sampler2D uTexCol;
  uniform sampler2D uTexBones;
  uniform sampler2D uTexMorph;
  uniform int uMorphCount;
  uniform int uMorphSplats;
  uniform int uMorphRows;
  uniform float uExprW[16];
  uniform mat4 uView;
  uniform vec2 uFocal;
  uniform vec2 uScreen;
  uniform float uScale;
  uniform float uDeskY;
  out vec2 vLocal;
  out vec4 vColor;
  mat3x4 loadBone(int j) {
    vec4 r0 = texelFetch(uTexBones, ivec2(0, j), 0);
    vec4 r1 = texelFetch(uTexBones, ivec2(1, j), 0);
    vec4 r2 = texelFetch(uTexBones, ivec2(2, j), 0);
    return mat3x4(r0, r1, r2);
  }
  void main() {
    int id = int(aSplatId + 0.5);
    ivec2 tc = ivec2(id & 1023, id >> 10);
    vec4 pW = texelFetch(uTexPos, tc, 0);
    vec4 uJ0 = texelFetch(uTexU, tc, 0);
    vec4 vJ1 = texelFetch(uTexV, tc, 0);
    vec4 col = texelFetch(uTexCol, tc, 0);

    vec3 bPos = pW.xyz;
    if (uMorphCount > 0 && id < uMorphSplats) {
      int mx = id & 1023;
      int myBase = id >> 10;
      for (int e = 0; e < 16; e++) {
        if (e >= uMorphCount) break;
        float ew = uExprW[e];
        if (ew > 0.001) {
          bPos += ew * texelFetch(uTexMorph, ivec2(mx, e * uMorphRows + myBase), 0).xyz;
        }
      }
    }

    int j0 = int(uJ0.w + 0.5);
    int j1 = int(vJ1.w + 0.5);
    float w0 = pW.w;
    mat3x4 m0 = loadBone(j0);
    mat3x4 m1 = (j1 == j0) ? m0 : loadBone(j1);
    vec4 r0 = m0[0] * w0 + m1[0] * (1.0 - w0);
    vec4 r1 = m0[1] * w0 + m1[1] * (1.0 - w0);
    vec4 r2 = m0[2] * w0 + m1[2] * (1.0 - w0);

    vec3 wPos = vec3(dot(r0.xyz, bPos) + r0.w, dot(r1.xyz, bPos) + r1.w, dot(r2.xyz, bPos) + r2.w);
    wPos.y += uDeskY;
    vec4 vc = uView * vec4(wPos, 1.0);
    if (vc.z > -0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

    vec3 wu = vec3(dot(r0.xyz, uJ0.xyz), dot(r1.xyz, uJ0.xyz), dot(r2.xyz, uJ0.xyz));
    vec3 wv = vec3(dot(r0.xyz, vJ1.xyz), dot(r1.xyz, vJ1.xyz), dot(r2.xyz, vJ1.xyz));
    vec3 wn = normalize(cross(wu, wv) + vec3(1e-7)) * (0.30 * length(wu));

    vec2 u2 = vec2(dot(vec3(uView[0].x, uView[1].x, uView[2].x), wu), dot(vec3(uView[0].y, uView[1].y, uView[2].y), wu));
    vec2 v2 = vec2(dot(vec3(uView[0].x, uView[1].x, uView[2].x), wv), dot(vec3(uView[0].y, uView[1].y, uView[2].y), wv));
    vec2 n2 = vec2(dot(vec3(uView[0].x, uView[1].x, uView[2].x), wn), dot(vec3(uView[0].y, uView[1].y, uView[2].y), wn));

    float ca = u2.x*u2.x + v2.x*v2.x + n2.x*n2.x;
    float cb = u2.x*u2.y + v2.x*v2.y + n2.x*n2.y;
    float cc = u2.y*u2.y + v2.y*v2.y + n2.y*n2.y;
    float mid = 0.5 * (ca + cc);
    float halfD = length(vec2(0.5 * (ca - cc), cb));
    float l1 = sqrt(max(1e-10, mid + halfD));
    float l2 = sqrt(max(1e-10, mid - halfD));
    float ang = (abs(cb) > 1e-12 || abs(ca - cc) > 1e-12) ? 0.5 * atan(2.0 * cb, ca - cc) : 0.0;
    vec2 ex = vec2(cos(ang), sin(ang));
    vec2 covA = ex * l1 * uScale;
    vec2 covB = vec2(-ex.y, ex.x) * l2 * uScale;

    float invZ = 1.0 / (-vc.z);
    vec2 ndcCenter = vec2(vc.x * uFocal.x, vc.y * uFocal.y) * invZ / (uScreen * 0.5);
    vec2 pxOff = (covA * aCorner.x + covB * aCorner.y) * uFocal * invZ;
    gl_Position = vec4(ndcCenter + pxOff / (uScreen * 0.5), 0.0, 1.0);
    vLocal = aCorner * 2.0;
    vColor = col;
  }`;

  const fsSplat = `#version 300 es
  precision highp float;
  in vec2 vLocal;
  in vec4 vColor;
  out vec4 outCol;
  void main() {
    float r2 = dot(vLocal, vLocal);
    if (r2 > 4.0) discard;
    float g = exp(-0.5 * r2) * vColor.a;
    outCol = vec4(vColor.rgb * g, g);
  }`;

  const vsShadow = `#version 300 es
  in vec2 aCorner;
  uniform mat4 uView;
  uniform vec2 uFocal;
  uniform vec2 uScreen;
  uniform float uDeskY;
  uniform float uRadius;
  out vec2 vXZ;
  void main() {
    vec3 wp = vec3(aCorner.x * uRadius, uDeskY, aCorner.y * uRadius);
    vec4 vc = uView * vec4(wp, 1.0);
    if (vc.z > -0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    float invZ = 1.0 / (-vc.z);
    gl_Position = vec4(vc.x * uFocal.x * invZ / (uScreen.x * 0.5), vc.y * uFocal.y * invZ / (uScreen.y * 0.5), 0.0, 1.0);
    vXZ = aCorner;
  }`;

  const fsShadow = `#version 300 es
  precision highp float;
  in vec2 vXZ;
  out vec4 outCol;
  void main() {
    float r2 = dot(vXZ, vXZ);
    if (r2 > 1.0) discard;
    float a = exp(-2.4 * r2) * 0.42;
    outCol = vec4(0.0, 0.0, 0.0, a);
  }`;

  function mkProg(vs, fs) {
    const comp = (t, s) => {
      const sh = gl.createShader(t);
      gl.shaderSource(sh, s);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      return sh;
    };
    const p = gl.createProgram();
    gl.attachShader(p, comp(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, comp(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    return p;
  }

  const prog = mkProg(vsSplat, fsSplat);
  const shProg = mkProg(vsShadow, fsShadow);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const qBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, qBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]), gl.STATIC_DRAW);
  const locC = gl.getAttribLocation(prog, "aCorner");
  gl.enableVertexAttribArray(locC);
  gl.vertexAttribPointer(locC, 2, gl.FLOAT, false, 0, 0);

  const idBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, idBuf);
  const locId = gl.getAttribLocation(prog, "aSplatId");
  gl.enableVertexAttribArray(locId);
  gl.vertexAttribPointer(locId, 1, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(locId, 1);

  const shVao = gl.createVertexArray();
  gl.bindVertexArray(shVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, qBuf);
  const locSC = gl.getAttribLocation(shProg, "aCorner");
  gl.enableVertexAttribArray(locSC);
  gl.vertexAttribPointer(locSC, 2, gl.FLOAT, false, 0, 0);

  function mkTex(unit) {
    const t = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  const texPos = mkTex(0), texU = mkTex(1), texV = mkTex(2), texCol = mkTex(3), texBones = mkTex(4), texMorph = mkTex(5);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4));

  const state = {
    data: null, playing: true, frame: 0.0, scale: 1.25, dotMode: false,
    yaw: 0.0, pitch: 0.14, dist: 3.0, panX: 0.0, panY: 0.0, panZ: 0.0,
    deskMode: false, bgMode: "none", deskY: 0.0, camStream: null, photoUrl: null, active: true,
    counts: new Int32Array(SORT_BINS), offs: new Int32Array(SORT_BINS),
    exprUniform: new Float32Array(MAX_MORPH),
  };

  let drag = null;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    const mode = (e.button === 2 || e.button === 1 || (e.button === 0 && e.shiftKey)) ? "pan" : "orbit";
    drag = { x: e.clientX, y: e.clientY, mode };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    if (drag.mode === "pan") {
      const sc = (2.0 * state.dist * Math.tan((16.0 * Math.PI) / 180.0)) / Math.max(1, canvas.clientHeight);
      const ca = Math.cos(state.yaw), sa = Math.sin(state.yaw);
      const cb = Math.cos(state.pitch), sb = Math.sin(state.pitch);
      const rx = ca, rz = -sa;
      const ux = -sa * sb, uy = cb, uz = -ca * sb;
      state.panX = Math.max(-3.0, Math.min(3.0, state.panX - rx * dx * sc + ux * dy * sc));
      state.panY = Math.max(-3.0, Math.min(3.0, state.panY + uy * dy * sc));
      state.panZ = Math.max(-3.0, Math.min(3.0, state.panZ - rz * dx * sc + uz * dy * sc));
    } else {
      state.yaw -= dx * 0.008;
      state.pitch = Math.max(-0.08, Math.min(1.15, state.pitch + dy * 0.008));
    }
    drag.x = e.clientX;
    drag.y = e.clientY;
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    state.dist = Math.max(0.7, Math.min(12.0, state.dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });

  function readArr(rawBuf, off, Ctor, count) {
    const bytes = count * Ctor.BYTES_PER_ELEMENT;
    if (off % Ctor.BYTES_PER_ELEMENT === 0) return { arr: new Ctor(rawBuf, off, count), next: off + bytes };
    return { arr: new Ctor(rawBuf.slice(off, off + bytes)), next: off + bytes };
  }

  async function loadR3DGBase64(b64) {
    const binStr = atob(b64);
    const bytes = new Uint8Array(binStr.length);
    for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
    if (bytes[0] !== 0x52 || bytes[1] !== 0x33 || bytes[2] !== 0x44 || bytes[3] !== 0x47) {
      throw new Error("Invalid R3DG payload");
    }
    const hdrLen = new DataView(bytes.buffer).getUint32(4, true);
    const hdr = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + hdrLen)));
    const comp = bytes.subarray(8 + hdrLen);
    if (typeof DecompressionStream === "undefined") {
      throw new Error("DecompressionStream not supported in this browser");
    }
    const rawBuf = await new Response(new Blob([comp]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer();
    const N = hdr.count, F = hdr.frames, J = hdr.joints;
    let off = 0, r;
    r = readArr(rawBuf, off, Uint8Array, N * 3); const rgb = r.arr; off = r.next;
    r = readArr(rawBuf, off, Uint8Array, N); const alpha = r.arr; off = r.next;
    r = readArr(rawBuf, off, Float32Array, N * 3); const bindPos = r.arr; off = r.next;
    r = readArr(rawBuf, off, Float32Array, N * 3); const bindU = r.arr; off = r.next;
    r = readArr(rawBuf, off, Float32Array, N * 3); const bindV = r.arr; off = r.next;
    r = readArr(rawBuf, off, Uint16Array, N); const b0 = r.arr; off = r.next;
    r = readArr(rawBuf, off, Uint16Array, N); const b1 = r.arr; off = r.next;
    r = readArr(rawBuf, off, Uint8Array, N); const bw0 = r.arr; off = r.next;
    r = readArr(rawBuf, off, Float32Array, F * J * 12); const boneMats = r.arr; off = r.next;

    const M = Math.min(MAX_MORPH, hdr.morph_count || 0);
    const K = M > 0 ? (hdr.morph_splats || 0) : 0;
    const mScale = hdr.morph_scale || 1.0;
    let qMorph = null, exprWeights = null, morphRows = 1;
    const exprMax = {};
    if (M > 0 && K > 0) {
      r = readArr(rawBuf, off, Int16Array, M * K * 3); qMorph = r.arr; off = r.next;
      r = readArr(rawBuf, off, Float32Array, F * M); exprWeights = r.arr; off = r.next;
      morphRows = Math.ceil(K / TEX_W);
      const tMorph = new Float32Array(M * morphRows * TEX_W * 4);
      for (let e = 0; e < M; e++) {
        const srcBase = e * K * 3;
        const dstRowBase = e * morphRows * TEX_W * 4;
        for (let i = 0; i < K; i++) {
          const s3 = srcBase + i * 3;
          const d4 = dstRowBase + i * 4;
          tMorph[d4] = qMorph[s3] * mScale;
          tMorph[d4 + 1] = qMorph[s3 + 1] * mScale;
          tMorph[d4 + 2] = qMorph[s3 + 2] * mScale;
        }
        let mx = 0;
        for (let f = 0; f < F; f++) {
          const v = exprWeights[f * M + e];
          if (v > mx) mx = v;
        }
        const nm = (hdr.expr_names && hdr.expr_names[e]) || ("expr" + e);
        exprMax[nm] = Math.round(mx * 1000) / 1000;
      }
      gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, texMorph);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, M * morphRows, 0, gl.RGBA, gl.FLOAT, tMorph);
    } else {
      gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, texMorph);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4));
    }
    console.info("[R3DG Expr] WebGL2 baked loaded:", {
      morph_count: M,
      morph_splats: K,
      expr_names: hdr.expr_names || [],
      max_weights: exprMax,
    });

    const texH = Math.ceil(N / TEX_W);
    const cap = texH * TEX_W;
    const tPos = new Float32Array(cap * 4), tU = new Float32Array(cap * 4), tV = new Float32Array(cap * 4);
    const tCol = new Uint8Array(cap * 4);
    for (let i = 0; i < N; i++) {
      const i3 = i * 3, i4 = i * 4;
      tPos[i4] = bindPos[i3]; tPos[i4+1] = bindPos[i3+1]; tPos[i4+2] = bindPos[i3+2]; tPos[i4+3] = bw0[i] / 255.0;
      tU[i4] = bindU[i3]; tU[i4+1] = bindU[i3+1]; tU[i4+2] = bindU[i3+2]; tU[i4+3] = b0[i];
      tV[i4] = bindV[i3]; tV[i4+1] = bindV[i3+1]; tV[i4+2] = bindV[i3+2]; tV[i4+3] = b1[i];
      tCol[i4] = rgb[i3]; tCol[i4+1] = rgb[i3+1]; tCol[i4+2] = rgb[i3+2]; tCol[i4+3] = alpha[i];
    }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texPos);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, texH, 0, gl.RGBA, gl.FLOAT, tPos);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texU);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, texH, 0, gl.RGBA, gl.FLOAT, tU);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, texV);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, texH, 0, gl.RGBA, gl.FLOAT, tV);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, texCol);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, TEX_W, texH, 0, gl.RGBA, gl.UNSIGNED_BYTE, tCol);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, texBones);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 3, J, 0, gl.RGBA, gl.FLOAT, boneMats.subarray(0, J * 12));

    gl.bindBuffer(gl.ARRAY_BUFFER, idBuf);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4, gl.DYNAMIC_DRAW);

    const bmin = hdr.bbox_min, bmax = hdr.bbox_max;
    state.dist = Math.max(2.0, (bmax[1] - bmin[1]) * 2.1);
    state.panX = 0.0; state.panY = 0.0; state.panZ = 0.0;
    state.frame = 0.0;
    state.data = {
      hdr, N, F, FPS: hdr.fps, J, M, K, morphRows, bindPos, b0, boneMats, qMorph, mScale, exprWeights,
      interpBones: new Float32Array(J * 12),
      jointDepth: new Float32Array(J * 4),
      depths: new Float32Array(N),
      order: new Float32Array(N),
      activeM: [],
    };
    if (scrubEl) scrubEl.max = Math.max(0, F - 1);
    const fpsStr = Number(hdr.fps).toFixed(Math.abs(hdr.fps - Math.round(hdr.fps)) > 0.05 ? 1 : 0);
    const mStr = M > 0 ? ` · ${M} expr` : "";
    if (hudEl) hudEl.textContent = `${N.toLocaleString()} Rigged Gaussians · ${F}F @ ${fpsStr}fps (${J} bones${mStr}) · L-drag: orbit · R-drag: move · Wheel: zoom`;
  }

  function clearDeskBg() {
    if (state.camStream) { state.camStream.getTracks().forEach((t) => t.stop()); state.camStream = null; }
    if (videoEl) { videoEl.srcObject = null; videoEl.style.display = "none"; }
    if (photoEl) { photoEl.removeAttribute("src"); photoEl.style.display = "none"; }
    if (state.photoUrl) { URL.revokeObjectURL(state.photoUrl); state.photoUrl = null; }
    state.bgMode = "none";
    state.deskMode = false;
  }

  function setDeskPhoto(file) {
    if (!file || !photoEl) return false;
    clearDeskBg();
    const url = URL.createObjectURL(file);
    state.photoUrl = url;
    photoEl.src = url;
    photoEl.style.display = "block";
    state.bgMode = "photo";
    state.deskMode = true;
    return true;
  }

  async function toggleDeskAR(forceOff) {
    if (forceOff || state.bgMode === "camera") {
      clearDeskBg();
      return { ok: true, on: false };
    }
    const inIframe = (() => { try { return window.self !== window.top; } catch (_) { return true; } })();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const reason = inIframe ? "iframe" : "no_api";
      console.error("[R3DG Camera BG] mediaDevices.getUserMedia unavailable", { isSecureContext: window.isSecureContext, inIframe, href: location.href });
      if (statusEl) { statusEl.textContent = "Camera API blocked (" + reason + ")."; statusEl.className = "err"; }
      return { ok: false, on: false, reason, message: "mediaDevices.getUserMedia unavailable" };
    }
    clearDeskBg();
    try {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
      } catch (e1) {
        console.warn("[R3DG Camera BG] environment camera failed, retrying default:", e1 && e1.name, e1 && e1.message, e1);
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      state.camStream = stream;
      videoEl.srcObject = stream;
      videoEl.style.display = "block";
      state.bgMode = "camera";
      state.deskMode = true;
      return { ok: true, on: true };
    } catch (e) {
      const reason = inIframe ? "iframe_denied" : (e && e.name ? e.name : "denied");
      const message = e && e.message ? e.message : "";
      console.error("[R3DG Camera BG] getUserMedia failed:", { name: e && e.name, message, isSecureContext: window.isSecureContext, inIframe, href: location.href }, e);
      if (navigator.mediaDevices.enumerateDevices) {
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          console.info("[R3DG Camera BG] videoinput devices:", devices.filter((d) => d.kind === "videoinput"));
        } catch (ed) {
          console.warn("[R3DG Camera BG] enumerateDevices failed:", ed);
        }
      }
      if (statusEl) { statusEl.textContent = "Camera blocked (" + reason + "). Try Photo BG or Single .HTML."; statusEl.className = "err"; }
      return { ok: false, on: false, reason, message };
    }
  }

  function renderFrame(dt) {
    const d = state.data;
    if (!d) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);

    if (state.playing) state.frame = (state.frame + dt * d.FPS) % d.F;
    const fWrap = ((state.frame % d.F) + d.F) % d.F;
    const f0 = Math.floor(fWrap) % d.F;
    const f1 = (f0 + 1) % d.F;
    const alphaT = fWrap - Math.floor(fWrap);
    if (scrubEl) { scrubEl.disabled = false; scrubEl.max = Math.max(0, d.F - 1); scrubEl.value = f0; }
    if (fnumEl) fnumEl.textContent = `${f0 + 1}/${d.F}`;

    const J12 = d.J * 12, off0 = f0 * J12, off1 = f1 * J12, ib = d.interpBones, bm = d.boneMats;
    const invA = 1.0 - alphaT;
    for (let k = 0; k < J12; k++) ib[k] = bm[off0 + k] * invA + bm[off1 + k] * alphaT;

    const eu = state.exprUniform;
    eu.fill(0);
    if (d.M > 0 && d.exprWeights) {
      const e0 = f0 * d.M, e1 = f1 * d.M, ew = d.exprWeights;
      for (let e = 0; e < d.M; e++) eu[e] = ew[e0 + e] * invA + ew[e1 + e] * alphaT;
    }

    const bmin = d.hdr.bbox_min, bmax = d.hdr.bbox_max;
    const cx = state.panX, cy = (bmin[1] + bmax[1]) * 0.5 + state.deskY + state.panY, cz = state.panZ;
    const ca = Math.cos(state.yaw), sa = Math.sin(state.yaw), cb = Math.cos(state.pitch), sb = Math.sin(state.pitch);
    const ex = cx + sa * cb * state.dist, ey = cy + sb * state.dist, ez = cz + ca * cb * state.dist;
    let fx = cx - ex, fy = cy - ey, fz = cz - ez;
    const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
    let rx = -fz, ry = 0, rz = fx;
    const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; rz /= rl;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    const V = new Float32Array([
      rx, ux, -fx, 0, ry, uy, -fy, 0, rz, uz, -fz, 0,
      -(rx*ex+ry*ey+rz*ez), -(ux*ex+uy*ey+uz*ez), (fx*ex+fy*ey+fz*ez), 1
    ]);

    const v2 = V[2], v6 = V[6], v10 = V[10], v14 = V[14], jd = d.jointDepth;
    for (let j = 0; j < d.J; j++) {
      const m = j * 12, o = j * 4;
      jd[o]   = v2 * ib[m]   + v6 * ib[m+4] + v10 * ib[m+8];
      jd[o+1] = v2 * ib[m+1] + v6 * ib[m+5] + v10 * ib[m+9];
      jd[o+2] = v2 * ib[m+2] + v6 * ib[m+6] + v10 * ib[m+10];
      jd[o+3] = v2 * (ib[m+3]) + v6 * (ib[m+7] + state.deskY) + v10 * ib[m+11] + v14;
    }

    const activeM = d.activeM;
    activeM.length = 0;
    for (let e = 0; e < d.M; e++) {
      if (eu[e] > 0.001) activeM.push(e);
    }

    const N = d.N, bp = d.bindPos, jb = d.b0, depths = d.depths, order = d.order;
    const qm = d.qMorph, msc = d.mScale, K = d.K, nAct = activeM.length;
    let minZ = 1e9, maxZ = -1e9;
    if (qm && nAct > 0 && K > 0) {
      const lim = Math.min(N, K);
      const stride = K * 3;
      for (let i = 0; i < lim; i++) {
        const p3 = i * 3, jo = jb[i] * 4;
        let px = bp[p3], py = bp[p3+1], pz = bp[p3+2];
        for (let a = 0; a < nAct; a++) {
          const e = activeM[a], w = eu[e] * msc, mo = e * stride + p3;
          px += qm[mo] * w;
          py += qm[mo + 1] * w;
          pz += qm[mo + 2] * w;
        }
        const vz = jd[jo]*px + jd[jo+1]*py + jd[jo+2]*pz + jd[jo+3];
        depths[i] = vz;
        if (vz < minZ) minZ = vz;
        if (vz > maxZ) maxZ = vz;
      }
      for (let i = lim; i < N; i++) {
        const p3 = i * 3, jo = jb[i] * 4;
        const vz = jd[jo]*bp[p3] + jd[jo+1]*bp[p3+1] + jd[jo+2]*bp[p3+2] + jd[jo+3];
        depths[i] = vz;
        if (vz < minZ) minZ = vz;
        if (vz > maxZ) maxZ = vz;
      }
    } else {
      for (let i = 0; i < N; i++) {
        const p3 = i * 3, jo = jb[i] * 4;
        const vz = jd[jo]*bp[p3] + jd[jo+1]*bp[p3+1] + jd[jo+2]*bp[p3+2] + jd[jo+3];
        depths[i] = vz;
        if (vz < minZ) minZ = vz;
        if (vz > maxZ) maxZ = vz;
      }
    }
    const counts = state.counts, offs = state.offs;
    counts.fill(0);
    const invZ = (SORT_BINS - 1) / Math.max(1e-4, maxZ - minZ);
    for (let i = 0; i < N; i++) counts[Math.max(0, Math.min(SORT_BINS - 1, ((depths[i] - minZ) * invZ) | 0))]++;
    let acc = 0;
    for (let b = 0; b < SORT_BINS; b++) { offs[b] = acc; acc += counts[b]; }
    for (let i = 0; i < N; i++) order[offs[Math.max(0, Math.min(SORT_BINS - 1, ((depths[i] - minZ) * invZ) | 0))]++] = i;

    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, texBones);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 3, d.J, gl.RGBA, gl.FLOAT, ib);
    gl.bindBuffer(gl.ARRAY_BUFFER, idBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, order);

    const focalY = h / (2 * Math.tan((32 * Math.PI) / 360));
    if (state.deskMode) gl.clearColor(0.0, 0.0, 0.0, 0.0);
    else gl.clearColor(0.067, 0.078, 0.125, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    if (state.deskMode) {
      gl.useProgram(shProg);
      gl.bindVertexArray(shVao);
      gl.uniformMatrix4fv(gl.getUniformLocation(shProg, "uView"), false, V);
      gl.uniform2f(gl.getUniformLocation(shProg, "uFocal"), focalY, focalY);
      gl.uniform2f(gl.getUniformLocation(shProg, "uScreen"), w, h);
      gl.uniform1f(gl.getUniformLocation(shProg, "uDeskY"), state.deskY + bmin[1]);
      gl.uniform1f(gl.getUniformLocation(shProg, "uRadius"), 0.42);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.uniform1i(gl.getUniformLocation(prog, "uTexPos"), 0);
    gl.uniform1i(gl.getUniformLocation(prog, "uTexU"), 1);
    gl.uniform1i(gl.getUniformLocation(prog, "uTexV"), 2);
    gl.uniform1i(gl.getUniformLocation(prog, "uTexCol"), 3);
    gl.uniform1i(gl.getUniformLocation(prog, "uTexBones"), 4);
    gl.uniform1i(gl.getUniformLocation(prog, "uTexMorph"), 5);
    gl.uniform1i(gl.getUniformLocation(prog, "uMorphCount"), d.M);
    gl.uniform1i(gl.getUniformLocation(prog, "uMorphSplats"), d.K);
    gl.uniform1i(gl.getUniformLocation(prog, "uMorphRows"), d.morphRows);
    const locExprW = gl.getUniformLocation(prog, "uExprW[0]") || gl.getUniformLocation(prog, "uExprW");
    if (locExprW !== null) gl.uniform1fv(locExprW, eu);
    gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uView"), false, V);
    gl.uniform2f(gl.getUniformLocation(prog, "uFocal"), focalY, focalY);
    gl.uniform2f(gl.getUniformLocation(prog, "uScreen"), w, h);
    gl.uniform1f(gl.getUniformLocation(prog, "uScale"), state.scale * (state.dotMode ? 0.38 : 1.0));
    gl.uniform1f(gl.getUniformLocation(prog, "uDeskY"), state.deskY);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, N);
  }

  let lastTs = performance.now();
  function loop(now) {
    const dt = Math.min(0.1, (now - lastTs) * 0.001);
    lastTs = now;
    if (state.active && state.data) renderFrame(dt);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  return { state, loadR3DGBase64, toggleDeskAR, setDeskPhoto, clearDeskBg };
}
"""


def export_standalone_html_bytes(baked):
    """Build a self-contained zero-CDN WebGL2 Rigged 3DGS + Facial Expressions HTML viewer."""
    payload_b64 = base64.b64encode(baked.compressed_r3dg).decode("ascii")
    title = _esc_html(baked.title)
    motion = _esc_html(baked.motion_label)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} - {motion} (VRM RETRO 4DGS STUDIO)</title>
<style>
  :root {{ --bg:#111420; --ink:#e8ecf8; --dim:#8f9ac0; --acc:#70c6a9; }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; height:100%; overflow:hidden; background:var(--bg); color:var(--ink); font:13px/1.4 system-ui,sans-serif; }}
  #cam,#photo {{ position:fixed; inset:0; width:100%; height:100%; object-fit:cover; display:none; z-index:0; }}
  canvas {{ position:relative; z-index:1; display:block; width:100%; height:100%; cursor:grab; }}
  canvas:active {{ cursor:grabbing; }}
  #hud {{ position:fixed; bottom:10px; left:10px; z-index:2; background:rgba(20,25,42,.82); backdrop-filter:blur(6px);
          border:1px solid #2e385c; border-radius:8px; padding:7px 11px; max-width:420px; pointer-events:none; }}
  #hud h1 {{ margin:0 0 2px; font-size:14px; }}
  #hud .sub {{ color:var(--dim); font-size:12px; }}
  #hud .err {{ color:#ff9798; }}
  #bar {{ position:fixed; left:10px; right:10px; top:10px; z-index:2; background:rgba(20,25,42,.86); backdrop-filter:blur(6px);
          border:1px solid #2e385c; border-radius:8px; padding:6px 10px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  button, label.pbtn {{ background:var(--acc); color:#0c1618; border:0; border-radius:5px; padding:5px 11px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:4px; }}
  button.sec, label.pbtn.sec {{ background:#2f3a60; color:var(--ink); }}
  button.on {{ outline:2px solid var(--acc); background:#254952; color:#fff; }}
  label.pbtn input {{ display:none; }}
  label {{ display:flex; align-items:center; gap:6px; font-size:12px; color:var(--dim); }}
  #scrub {{ flex:1; min-width:110px; accent-color:var(--acc); }}
  input[type=range] {{ accent-color:var(--acc); }}
  #scale {{ width:82px; }}
  #desky {{ width:76px; }}
  #err {{ position:fixed; inset:0; z-index:5; display:none; align-items:center; justify-content:center; padding:24px; text-align:center; background:var(--bg); }}
</style>
</head>
<body>
<video id="cam" autoplay playsinline muted></video>
<img id="photo" alt="">
<canvas id="c"></canvas>
<div id="hud">
  <h1>{title} — {motion}</h1>
  <div class="sub" id="meta">Loading Rigged 3DGS...</div>
  <div class="sub" id="stat"></div>
</div>
<div id="bar">
  <button id="play" type="button">Pause</button>
  <input id="scrub" type="range" min="0" max="1" value="0" step="1">
  <span id="fnum" style="min-width:54px;text-align:right;font-variant-numeric:tabular-nums">0/0</span>
  <label>Scale <input id="scale" type="range" min="0.4" max="2.5" step="0.05" value="1.25"></label>
  <button id="style" class="sec" type="button">Style: Gaussian</button>
  <button id="ar" class="sec" type="button">Camera BG</button>
  <label class="pbtn sec">Photo BG<input id="photo_in" type="file" accept="image/*"></label>
  <button id="clr_bg" class="sec" type="button" style="display:none">Clear BG</button>
  <label id="desky_wrap" style="display:none">Height Y <input id="desky" type="range" min="-0.6" max="0.6" step="0.02" value="0.0"></label>
  <button id="reset" class="sec" type="button">Reset</button>
</div>
<div id="err"></div>
<script>
{R3DG_PLAYER_JS}
const PAYLOAD_B64 = "{payload_b64}";
(async () => {{
  const errBox = document.getElementById("err");
  const statEl = document.getElementById("stat");
  function fail(msg) {{ errBox.style.display = "flex"; errBox.textContent = msg; }}
  const player = createR3DGPlayer({{
    canvas: document.getElementById("c"),
    videoEl: document.getElementById("cam"),
    photoEl: document.getElementById("photo"),
    hudEl: document.getElementById("meta"),
    statusEl: statEl,
    scrubEl: document.getElementById("scrub"),
    fnumEl: document.getElementById("fnum"),
  }});
  if (!player) return fail("WebGL2 is required to view this file.");
  try {{
    await player.loadR3DGBase64(PAYLOAD_B64);
  }} catch (e) {{
    return fail(e.message || String(e));
  }}
  const playBtn = document.getElementById("play");
  const arBtn = document.getElementById("ar");
  const clrBtn = document.getElementById("clr_bg");
  const deskyWrap = document.getElementById("desky_wrap");
  const styleBtn = document.getElementById("style");
  function syncBgUi() {{
    arBtn.classList.toggle("on", player.state.bgMode === "camera");
    clrBtn.style.display = player.state.deskMode ? "inline-flex" : "none";
    deskyWrap.style.display = player.state.deskMode ? "flex" : "none";
  }}
  playBtn.addEventListener("click", () => {{
    player.state.playing = !player.state.playing;
    playBtn.textContent = player.state.playing ? "Pause" : "Play";
  }});
  document.getElementById("scrub").addEventListener("input", (e) => {{
    player.state.frame = +e.target.value;
    player.state.playing = false;
    playBtn.textContent = "Play";
  }});
  document.getElementById("scale").addEventListener("input", (e) => {{ player.state.scale = +e.target.value; }});
  document.getElementById("desky").addEventListener("input", (e) => {{ player.state.deskY = +e.target.value; }});
  styleBtn.addEventListener("click", () => {{
    player.state.dotMode = !player.state.dotMode;
    styleBtn.textContent = player.state.dotMode ? "Style: Dots" : "Style: Gaussian";
    styleBtn.classList.toggle("on", player.state.dotMode);
  }});
  document.getElementById("reset").addEventListener("click", () => {{
    player.state.yaw = 0; player.state.pitch = 0.14;
    player.state.panX = 0; player.state.panY = 0; player.state.panZ = 0;
    const h = player.state.data.hdr;
    player.state.dist = Math.max(2.0, (h.bbox_max[1] - h.bbox_min[1]) * 2.1);
  }});
  arBtn.addEventListener("click", async () => {{
    const res = await player.toggleDeskAR();
    if (res && res.ok && res.on) {{ statEl.textContent = ""; statEl.className = "sub"; }}
    syncBgUi();
  }});
  document.getElementById("photo_in").addEventListener("change", (e) => {{
    if (e.target.files[0]) {{
      player.setDeskPhoto(e.target.files[0]);
      statEl.textContent = "";
      statEl.className = "sub";
      syncBgUi();
    }}
    e.target.value = "";
  }});
  clrBtn.addEventListener("click", () => {{ player.clearDeskBg(); syncBgUi(); }});
  window.addEventListener("keydown", (e) => {{ if (e.code === "Space") {{ e.preventDefault(); playBtn.click(); }} }});
}})();
</script>
</body>
</html>
"""
    return html.encode("utf-8")
