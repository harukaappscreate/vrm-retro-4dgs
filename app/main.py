"""VRM RETRO 4DGS STUDIO - HTML+Pyxel 3 studio with VRMA blink/lip-sync morphs, face-safe sampling & Camera/Photo BG.

Browser (itch.io): full-canvas Pyxel viewport controlled by web/index.html studio sidebar.
Desktop fallback : python app/main.py [model.vrm]
"""
# SPDX-License-Identifier: MIT
import base64
import json
import math
import os
import sys
import tempfile

import numpy as np
import pyxel
from pyxel.cube import Camera, Mat4, Node, Primitive, Shading, Vec3

import fbx_io
import splat_engine as SE
import vrma_io
import vrm_io as V

W, H = 320, 240
UI = [
    0x000000, 0x161A2C, 0x262D4D, 0x19959C,
    0x8B4852, 0x395C98, 0xA9C1FF, 0xEEEEEE,
    0xD4186C, 0xD38441, 0xE9C35B, 0x70C6A9,
    0x7696DE, 0x8F9AC0, 0xFF9798, 0xEDC7B0,
]
BG = 1
KEY = 255
VIEW_MODES = ["4DGS", "SPLIT", "MESH", "DOTS"]
BRUSH_MODES = ["DITHER", "SOLID"]
SPLAT_COUNTS = [8000, 15000, 30000, 50000]
BAKE_FPS_OPTIONS = [15, 24, 30]
PREVIEW_CAP = 6000
INBOX = "/tmp/inbox"
TMP = tempfile.mkdtemp(prefix="vrm4dgs_")
HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.join(os.path.dirname(HERE), "exports")

try:
    import js
except ImportError:
    js = None


def build_gaussian_atlases():
    """Create 128x128 256-color retro Gaussian atlases (Bayer dithered + solid disc)."""
    bayer4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ], dtype=np.float32) / 16.0

    yy, xx = np.mgrid[0:8, 0:8]
    dist = np.sqrt((xx - 3.5) ** 2 + (yy - 3.5) ** 2)
    gauss = np.exp(-0.42 * (dist ** 2))
    mask_dither = (dist <= 3.7) & (gauss > (bayer4[yy % 4, xx % 4] * 0.72 + 0.08))
    mask_solid = dist <= 3.35

    def make_img(mask):
        im = pyxel.Image(128, 128)
        im.cls(KEY)
        for c in range(255):
            ox, oy = (c % 16) * 8, (c // 16) * 8
            for dy in range(8):
                for dx in range(8):
                    if mask[dy, dx]:
                        im.pset(ox + dx, oy + dy, c)
        return im

    return make_img(mask_dither), make_img(mask_solid)


def png_from_index(idx, palette, path):
    from PIL import Image
    cols = np.zeros((256, 3), np.uint8)
    cols[:16] = [((c >> 16) & 255, (c >> 8) & 255, c & 255) for c in UI]
    cols[16:16 + len(palette)] = palette
    cols[KEY] = (255, 0, 255)
    Image.fromarray(cols[idx]).save(path)


def _quad_texcoords(pal_indices):
    cx = (pal_indices % 16).astype(np.float32) / 16.0
    cy = (pal_indices // 16).astype(np.float32) / 16.0
    s = 1.0 / 16.0
    eps = 0.001
    return np.stack([
        np.stack([cx + eps, cy + eps], axis=1),
        np.stack([cx + s - eps, cy + eps], axis=1),
        np.stack([cx + s - eps, cy + s - eps], axis=1),
        np.stack([cx + eps, cy + s - eps], axis=1),
    ], axis=1).reshape(-1, 2)


class MeshFigureNode(Node):
    """Original textured VRM mesh node (for MESH and SPLIT comparison)."""

    def __init__(self, rig):
        super().__init__()
        self.rig = rig
        m = rig.m
        self.prims = []
        for p in m.parts:
            pr = Primitive(
                Primitive.MODE_TRIANGLES,
                p["pos"].reshape(-1).tolist(),
                p["idx"].astype(np.int64).tolist(),
                [],
                p["uv"].reshape(-1).tolist(),
                cull=Primitive.CULL_NONE if p["double"] else Primitive.CULL_BACK,
            )
            pr.compute_normals()
            self.prims.append(pr)
        self.textures = []
        for k, idx in enumerate(m.tex_index):
            path = os.path.join(TMP, f"tex_{k}.png")
            png_from_index(idx, m.palette, path)
            self.textures.append(pyxel.Image.from_image(path))
        self.draw_order = sorted(range(len(m.parts)), key=lambda i: {"OPAQUE": 0, "MASK": 1}.get(m.parts[i]["alpha"], 2))
        self.identity = Mat4()

    def sync(self, skinned_parts):
        for pr, pos in zip(self.prims, skinned_parts):
            pr.positions[:] = pos.reshape(-1).tolist()
            pr.compute_normals()

    def on_draw(self):
        for i in self.draw_order:
            p = self.rig.m.parts[i]
            if p["keyed"]:
                self.prim(self.identity, self.prims[i], self.textures[p["tex"]], colkey=KEY)
            else:
                self.prim(self.identity, self.prims[i], self.textures[p["tex"]])


class SplatCloudNode(Node):
    """Single-draw-call Retro Gaussian Splatting / colored micro-dot renderer."""

    def __init__(self, pal_idx, atlas_dither, atlas_solid):
        super().__init__()
        self.atlas_dither = atlas_dither
        self.atlas_solid = atlas_solid
        self.use_dither = True
        self.dots_mode = False
        self.identity = Mat4()
        self.n = len(pal_idx)

        base = np.arange(self.n, dtype=np.int64) * 4
        quad_idx = np.stack([base, base + 1, base + 2, base, base + 2, base + 3], axis=1).reshape(-1).tolist()
        quad_uv = _quad_texcoords(pal_idx).reshape(-1).tolist()
        self.quad_prim = Primitive(
            Primitive.MODE_TRIANGLES,
            [0.0] * (self.n * 12),
            quad_idx,
            [],
            quad_uv,
            cull=Primitive.CULL_NONE,
        )

    def sync(self, centers, u_vec, v_vec, splat_scale):
        mul = splat_scale * (0.42 if self.dots_mode else 1.0)
        u = u_vec * mul
        v = v_vec * mul
        pos = np.stack([centers - u - v, centers + u - v, centers + u + v, centers - u + v], axis=1)
        self.quad_prim.positions[:] = pos.reshape(-1).tolist()

    def on_draw(self):
        atlas = self.atlas_solid if (self.dots_mode or not self.use_dither) else self.atlas_dither
        self.prim(self.identity, self.quad_prim, atlas, colkey=KEY)


class Bridge:
    @staticmethod
    def commands():
        cmds = list(getattr(pyxel, "_vrm4d", None) or [])
        if cmds:
            pyxel._vrm4d = []
        return cmds

    @staticmethod
    def send(fn, data):
        if js is None:
            return
        try:
            getattr(js, fn)(json.dumps(data))
        except Exception:
            pass

    @staticmethod
    def send_str(fn, val):
        if js is None:
            return
        try:
            getattr(js, fn)(val)
        except Exception:
            pass


class App:
    def __init__(self, path=None):
        pyxel.init(W, H, title="VRM RETRO 4DGS STUDIO", fps=30, capture_sec=6)
        pyxel.mouse(True)
        pyxel.colors[:] = UI + [0x808080] * (255 - 16) + [0xFF00FF]
        self.atlas_dither, self.atlas_solid = build_gaussian_atlases()
        self.compact_ui = (js is not None)

        self.motions = SE.Motions(os.path.join(HERE, "motions.npz"))
        self.motion_expr_diags = {}
        demo_vrma = os.path.join(HERE, "expr_demo.vrma")
        if os.path.exists(demo_vrma):
            try:
                with open(demo_vrma, "rb") as f:
                    _dd, _dh, _de, _ds, _dn = vrma_io.read_vrma(f.read(), self.motions.bones)
                self.motions.add(
                    "Wink & Lip-Sync Demo", _dd, _dh,
                    group="Expression Demo", loop=True, Expr=_de,
                    source="builtin", fixed_id="expr_demo", prepend=True,
                )
                self.motion_expr_diags["expr_demo"] = dict(getattr(vrma_io, "LAST_DIAG", {}) or {})
            except Exception:
                pass
        self.rig = None
        self.sampler = None
        self.baked = None
        self.bake_job = None
        self.bake_prog = (0, 1)
        self.pending_after_bake = None

        self.mesh_stage = Node()
        self.splat_stage = Node()
        self.mesh_node = None
        self.splat_node = None

        self.cam = Camera()
        self.cam.fov = 32.0
        self.cam.near = 0.05
        self.cam.far = 50.0
        self.cam.clear_color = BG
        self.mesh_stage.camera = self.cam
        self.splat_stage.camera = self.cam

        self.yaw, self.pitch, self.dist = 0.0, 10.0, 3.2
        self.drag = None
        self.pan = [0.0, 0.0, 0.0]
        self.pan_drag = None

        self.view_mode = 0       # 0=4DGS, 1=SPLIT, 2=MESH, 3=DOTS
        self.brush_idx = 0       # 0=DITHER, 1=SOLID
        self.cull_interior = True
        self.auto_blink = True
        self.expr_test = ""
        self.count_idx = 2       # default 30000 splats (Rigged 3DGS)
        self.fps_idx = 1         # default 24 fps
        self.splat_scale = 1.25
        self.motion_ids = [m["id"] for m in self.motions.meta]
        self.motion_idx = self.motion_ids.index("walk") if "walk" in self.motion_ids else 0

        self.playing = True
        self.t = 0.0
        self.bake_frame = 0
        self.last_sent_frame = -1
        self.show_help = False
        self.note, self.note_t = "", 0.0

        os.makedirs(INBOX, exist_ok=True)
        if js is None:
            os.makedirs(EXPORTS, exist_ok=True)
        self.seen = set(os.listdir(INBOX))
        self.used = set()

        self.send_motions(select=self.current_motion_id())

        candidates = [p for p in [
            path,
            os.path.join(HERE, "sample.vrm"),
            os.path.join(os.path.dirname(HERE), "web", "sample.vrm"),
        ] if p and os.path.exists(p)]
        if candidates:
            self.load_model(candidates[0])
        else:
            self._install_model(SE.ProceduralModel())

        pyxel.run(self.update, self.draw)

    @property
    def view_top(self):
        return 0 if self.compact_ui else 14

    @property
    def view_bot(self):
        return H if self.compact_ui else H - 16

    def send_motions(self, select=None):
        Bridge.send("v4dMotions", {
            "motions": [
                dict(id=m["id"], label=m["label"], group=m.get("group", "Misc"),
                     seconds=m.get("seconds", 1.0), user=m.get("source") == "user",
                     has_expr=bool(m.get("has_expr", False)))
                for m in self.motions.meta
            ],
            "select": select,
        })

    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                data = f.read()
            model = V.VrmModel(data)
            self._install_model(model)
            Bridge.send("v4dExprLog", {"kind": "vrm", "title": model.title, "diag": getattr(model, "expr_diag", {})})
            e_cnt = len(model.available_expressions)
            tag = f" ({e_cnt} EXPR)" if e_cnt > 0 else ""
            self.flash(f"LOADED: {model.title[:20].upper()}{tag}")
        except Exception as e:
            self.flash(f"ERR: {e}"[:44].upper(), err=True)
        finally:
            if path.startswith(INBOX):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _send_info(self):
        if self.rig is None or self.sampler is None:
            return
        m = self.rig.m
        avail = getattr(m, "available_expressions", []) or []
        names = getattr(m, "expression_names", V.EXPRESSION_NAMES)
        Bridge.send("v4dInfo", {
            "title": m.title,
            "author": m.author,
            "version": m.version,
            "tris": int(m.tris),
            "splats": self.sampler.count,
            "face_splats": int(self.sampler.is_face.sum()),
            "culled_pct": round(self.sampler.culled_pct, 1),
            "expr_count": len(avail),
            "expr_names": [names[e] for e in avail],
            "expr_diag": getattr(m, "expr_diag", None),
            "license": m.license,
            "warnings": m.warnings,
        })

    def _install_model(self, model):
        cols = UI + [int(r) << 16 | int(g) << 8 | int(b) for r, g, b in model.palette]
        cols += [0x808080] * (256 - len(cols))
        cols[KEY] = 0xFF00FF
        pyxel.colors[:] = cols[:256]

        sh = Shading(list(pyxel.colors))
        sh.direction = Vec3(-0.4, -0.8, -0.5)
        self.mesh_stage.shading = sh

        self.rig = SE.RigState(model, self.motions)
        self.baked = None
        self.bake_job = None
        self.dist = max(2.2, model.height * 2.15)
        self.yaw, self.pitch = 0.0, 10.0
        self.pan = [0.0, 0.0, 0.0]

        if self.mesh_node:
            self.mesh_stage.remove_child(self.mesh_node)
        self.mesh_node = MeshFigureNode(self.rig)
        self.mesh_stage.add_child(self.mesh_node)
        self.rebuild_sampler()

    def rebuild_sampler(self):
        if self.rig is None:
            return
        n = SPLAT_COUNTS[self.count_idx]
        self.sampler = SE.build_sampler(self.rig, n, cull_interior=self.cull_interior)
        self.baked = None
        self.last_sent_frame = -1
        p_idx, p_boost = SE.select_preview_indices(self.sampler, PREVIEW_CAP)
        self.preview_idx = p_idx
        self.preview_boost = p_boost
        if self.splat_node:
            self.splat_stage.remove_child(self.splat_node)
        self.splat_node = SplatCloudNode(self.sampler.pal_idx[p_idx], self.atlas_dither, self.atlas_solid)
        self.splat_node.use_dither = (self.brush_idx == 0)
        self.splat_node.dots_mode = (self.view_mode == 3)
        self.splat_stage.add_child(self.splat_node)

        face_cnt = int(self.sampler.is_face.sum())
        e_cnt = len(self.sampler.morph_expr_ids)
        self._send_info()
        Bridge.send("v4dStatus", {
            "text": f"Ready: {n:,} splats (face {face_cnt:,} · {e_cnt} morphs · culled {self.sampler.culled_pct:.0f}%)",
            "baked": False,
        })

    def load_motion_file(self, path):
        name = os.path.basename(path)
        if path.startswith(INBOX):
            name = name.split("_", 1)[-1]
        stem = os.path.splitext(name)[0].replace("_", " ").strip() or "motion"
        try:
            with open(path, "rb") as f:
                data = f.read()
            if name.lower().endswith(".vrma"):
                D, Hp, Expr, _sec, _nm = vrma_io.read_vrma(data, self.motions.bones)
                diag = dict(getattr(vrma_io, "LAST_DIAG", {}) or {})
                mid = self.motions.add(stem, D, Hp, "Your motions", loop=True, Expr=Expr)
                self.motion_expr_diags[mid] = diag
                n_expr = int((Expr.max(axis=0) > 1e-4).sum()) if Expr is not None else 0
                Bridge.send("v4dExprLog", {"kind": "vrma", "name": stem, "diag": diag})
            else:
                ff = fbx_io.FbxFile(data)
                human, _ = fbx_io.humanoid(ff.bones)
                takes = [k for k, t in enumerate(ff.takes) if t[2] > 0]
                if not takes:
                    raise ValueError("no animation in FBX")
                D, Hp = fbx_io.to_viewer(ff.motion(takes[0]), self.motions.bones, human)
                mid = self.motions.add(stem, D, Hp, "Your motions", loop=True)
                n_expr = 0
            self.motion_ids = [m["id"] for m in self.motions.meta]
            self.motion_idx = self.motion_ids.index(mid)
            self.baked = None
            self.t = 0.0
            self.send_motions(select=mid)
            etag = f" (+{n_expr} EXPR)" if n_expr > 0 else ""
            self.flash(f"MOTION: {stem[:18].upper()}{etag}")
        except Exception as e:
            self.flash(f"MOT ERR: {e}"[:42].upper(), err=True)
        finally:
            if path.startswith(INBOX):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def current_motion_id(self):
        return self.motion_ids[self.motion_idx % len(self.motion_ids)]

    def start_bake(self, after=None):
        if self.rig is None or self.sampler is None or self.bake_job is not None:
            return
        self.pending_after_bake = after
        mid = self.current_motion_id()
        fps = BAKE_FPS_OPTIONS[self.fps_idx]
        expr_override = V.resolve_expr_idx(self.expr_test) if self.expr_test else None
        self.bake_job = SE.bake_rigged_generator(
            self.rig, self.sampler, mid, fps=fps, max_frames=900,
            preview_cap=PREVIEW_CAP, auto_blink=self.auto_blink, expr_override=expr_override,
        )
        self.bake_prog = (0, 1)
        Bridge.send("v4dProgress", {"busy": True, "pct": 0, "text": f"Baking {mid} @ {fps}fps..."})

    def _live_expr_weights(self, mid, t):
        if self.expr_test:
            ei = V.resolve_expr_idx(self.expr_test)
            if ei is not None:
                ew = np.zeros(len(V.EXPRESSION_NAMES), dtype=np.float32)
                ew[ei] = 1.0
                return ew
        ew = self.motions.expr_frame(mid, t)
        if ew is None and self.auto_blink and getattr(self.rig.m, "has_expressions", False) and mid != "tpose":
            ew = SE.auto_blink_weights(t, self.rig.m.available_expressions)
        return ew

    def _current_full_splat_frame(self):
        if self.baked is not None:
            return SE.decode_rigged_frame(self.baked, int(self.bake_frame) % self.baked.frames)
        mid = self.current_motion_id()
        ew = self._live_expr_weights(mid, self.t)
        q, tr = self.rig.pose_at(mid, self.t)
        skinned = self.rig.skin_parts(q, tr, apply_face_yaw=True, expr_w=ew)
        return SE.eval_splat_frame(skinned, self.sampler, full_rot=True)

    def do_export(self, kind):
        if self.rig is None or self.sampler is None or self.bake_job is not None:
            return
        if kind in ("zip", "html") and self.baked is None:
            self.start_bake(after=f"export:{kind}")
            return
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{self.rig.m.title}_{self.current_motion_id()}")[:40]
        if kind == "ply":
            Bridge.send("v4dProgress", {"busy": True, "pct": 70, "text": "Writing .PLY..."})
            sf = self._current_full_splat_frame()
            data = SE.export_ply_bytes(sf, self.sampler.rgb, self.sampler.alpha, y_down=True)
            fname = f"{stem}.ply"
        elif kind == "splat":
            Bridge.send("v4dProgress", {"busy": True, "pct": 70, "text": "Writing .SPLAT..."})
            sf = self._current_full_splat_frame()
            data = SE.export_splat_bytes(sf, self.sampler.rgb, self.sampler.alpha, y_down=True)
            fname = f"{stem}.splat"
        elif kind == "zip":
            Bridge.send("v4dProgress", {"busy": True, "pct": 94, "text": "Packaging 4DGS ZIP..."})
            data = SE.export_4dgs_zip_bytes(self.baked)
            fname = f"{stem}_4dgs.zip"
        elif kind == "html":
            Bridge.send("v4dProgress", {"busy": True, "pct": 94, "text": "Building standalone HTML..."})
            data = SE.export_standalone_html_bytes(self.baked)
            fname = f"{stem}_4dgs.html"
        else:
            return

        if js is not None:
            tmp_path = os.path.join(TMP, fname)
            with open(tmp_path, "wb") as f:
                f.write(data)
            js._savePyxelFile(tmp_path)
            Bridge.send("v4dProgress", {"busy": False})
            self.flash(f"DOWNLOAD: {fname[:22]} ({len(data)//1024}KB)")
        else:
            out_path = os.path.join(EXPORTS, fname)
            with open(out_path, "wb") as f:
                f.write(data)
            Bridge.send("v4dProgress", {"busy": False})
            self.flash(f"SAVED exports/{fname[:18]} ({len(data)//1024}KB)")

    def flash(self, text, err=False):
        self.note = text
        self.note_t = 3.0
        mb = round(len(self.baked.compressed_r3dg) / (1024 * 1024), 2) if self.baked is not None else 0.0
        Bridge.send("v4dStatus", {
            "text": text,
            "error": err,
            "baked": self.baked is not None,
            "frames": self.baked.frames if self.baked is not None else 0,
            "fps": round(float(self.baked.fps), 2) if self.baked is not None else BAKE_FPS_OPTIONS[self.fps_idx],
            "mb": mb,
        })

    def _new_file(self):
        for f in list(getattr(pyxel, "dropped_files", None) or []) + list(getattr(pyxel, "_dropped_files", None) or []):
            if f not in self.used:
                self.used.add(f)
                return f
        try:
            for f in sorted(os.listdir(INBOX)):
                if f not in self.seen:
                    self.seen.add(f)
                    return os.path.join(INBOX, f)
        except OSError:
            pass
        return None

    def _command(self, cmd):
        if cmd == "view:reset":
            self.yaw, self.pitch = 0.0, 10.0
            self.dist = max(2.2, self.rig.m.height * 2.15) if self.rig else 3.2
            self.pan = [0.0, 0.0, 0.0]
        elif cmd == "bake":
            self.start_bake()
        elif cmd == "play:toggle":
            self.playing = not self.playing
        elif cmd.startswith("frame:"):
            if self.baked is not None:
                self.playing = False
                self.bake_frame = int(cmd.split(":", 1)[1]) % self.baked.frames
        elif cmd.startswith("motion:"):
            mid = cmd.split(":", 1)[1]
            if mid in self.motion_ids:
                self.motion_idx = self.motion_ids.index(mid)
                self.baked = None
                self.t = 0.0
                if mid in self.motion_expr_diags:
                    Bridge.send("v4dExprLog", {"kind": "vrma", "name": mid, "diag": self.motion_expr_diags[mid]})
                else:
                    Bridge.send("v4dExprLog", {"kind": "vrma", "name": mid, "diag": {"has_expr_ext": False, "max_weights": {}}})
                Bridge.send("v4dStatus", {"text": f"Motion: {mid} (Bake needed for WebGL2)", "baked": False})
        elif cmd.startswith("count:"):
            val = int(cmd.split(":", 1)[1])
            if val in SPLAT_COUNTS:
                self.count_idx = SPLAT_COUNTS.index(val)
                self.rebuild_sampler()
        elif cmd.startswith("fps:"):
            val = int(cmd.split(":", 1)[1])
            if val in BAKE_FPS_OPTIONS:
                self.fps_idx = BAKE_FPS_OPTIONS.index(val)
                self.baked = None
                Bridge.send("v4dStatus", {"text": f"Bake FPS: {val}fps (Re-bake needed)", "baked": False})
        elif cmd.startswith("autoblink:"):
            self.auto_blink = (cmd.split(":", 1)[1] != "0")
            self.baked = None
            Bridge.send("v4dStatus", {"text": f"Auto Blink: {'On' if self.auto_blink else 'Off'}", "baked": False})
        elif cmd.startswith("exprtest:"):
            self.expr_test = cmd.split(":", 1)[1].strip()
            self.baked = None
            Bridge.send("v4dStatus", {"text": f"Expr Test: {self.expr_test or 'Motion / Auto'} (Re-bake for WebGL2)", "baked": False})
        elif cmd.startswith("scale:"):
            self.splat_scale = max(0.4, min(2.5, float(cmd.split(":", 1)[1])))
        elif cmd.startswith("cull:"):
            self.cull_interior = (cmd.split(":", 1)[1] != "0")
            self.rebuild_sampler()
        elif cmd.startswith("mode:"):
            m = cmd.split(":", 1)[1]
            if m in VIEW_MODES:
                self.view_mode = VIEW_MODES.index(m)
                if self.splat_node:
                    self.splat_node.dots_mode = (self.view_mode == 3)
        elif cmd.startswith("brush:"):
            b = cmd.split(":", 1)[1]
            if b in BRUSH_MODES:
                self.brush_idx = BRUSH_MODES.index(b)
                if self.splat_node:
                    self.splat_node.use_dither = (self.brush_idx == 0)
        elif cmd.startswith("export:"):
            self.do_export(cmd.split(":", 1)[1])
        elif cmd == "preview:webgl":
            if self.baked is None:
                self.start_bake(after="preview:webgl")
            else:
                Bridge.send("v4dWebGLReady", {"ready": True})
                self.flash(f"WEBGL2 READY ({self.baked.frames}F)")

    def _handle_clicks(self):
        if self.compact_ui or not pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            return False
        mx, my = pyxel.mouse_x, pyxel.mouse_y
        if my < self.view_top:
            if 4 <= mx < 46:
                self._cycle_view_mode()
            elif 49 <= mx < 94:
                self._toggle_brush()
            elif 97 <= mx < 145:
                self.count_idx = (self.count_idx + 1) % len(SPLAT_COUNTS)
                self.rebuild_sampler()
            elif 148 <= mx < 186:
                self.fps_idx = (self.fps_idx + 1) % len(BAKE_FPS_OPTIONS)
                self.baked = None
            elif 189 <= mx < 282:
                self._step_motion(1)
            elif 284 <= mx < 316:
                self.start_bake()
            return True
        if my >= self.view_bot:
            if 4 <= mx < 36:
                self.playing = not self.playing
            elif 40 <= mx < 54:
                self.splat_scale = max(0.4, round(self.splat_scale - 0.15, 2))
            elif 92 <= mx < 106:
                self.splat_scale = min(2.5, round(self.splat_scale + 0.15, 2))
            elif 196 <= mx < 224:
                self.do_export("ply")
            elif 227 <= mx < 255:
                self.do_export("splat")
            elif 258 <= mx < 286:
                self.do_export("zip")
            elif 289 <= mx < 317:
                self.do_export("html")
            return True
        return False

    def _cycle_view_mode(self):
        self.view_mode = (self.view_mode + 1) % len(VIEW_MODES)
        if self.splat_node:
            self.splat_node.dots_mode = (self.view_mode == 3)

    def _toggle_brush(self):
        self.brush_idx = 1 - self.brush_idx
        if self.splat_node:
            self.splat_node.use_dither = (self.brush_idx == 0)

    def _step_motion(self, delta):
        self.motion_idx = (self.motion_idx + delta) % len(self.motion_ids)
        self.baked = None
        self.t = 0.0
        Bridge.send("v4dMotion", {"id": self.current_motion_id()})

    def update(self):
        f = self._new_file()
        if f:
            if f.lower().endswith((".vrma", ".fbx")):
                self.load_motion_file(f)
            else:
                self.load_model(f)
            return

        for cmd in Bridge.commands():
            self._command(cmd)

        if self.bake_job is not None:
            try:
                done, total, res = next(self.bake_job)
                self.bake_prog = (done, total)
                pct = int(done * (90 if self.pending_after_bake else 100) / max(1, total))
                Bridge.send("v4dProgress", {"busy": True, "pct": pct, "text": f"Baking 4DGS: {done} / {total} frames"})
                if res is not None:
                    self.baked = res
                    self.bake_job = None
                    self.bake_frame = 0
                    mb = len(res.compressed_r3dg) / (1024 * 1024)
                    Bridge.send_str("v4dBakedPayload", base64.b64encode(res.compressed_r3dg).decode("ascii"))
                    after = self.pending_after_bake
                    self.pending_after_bake = None
                    if after == "export:zip":
                        self.do_export("zip")
                    elif after == "export:html":
                        self.do_export("html")
                    elif after == "preview:webgl":
                        Bridge.send("v4dProgress", {"busy": False})
                        Bridge.send("v4dWebGLReady", {"ready": True})
                        self.flash(f"WEBGL2 READY ({res.frames}F, {mb:.1f}MB)")
                    else:
                        Bridge.send("v4dProgress", {"busy": False})
                        self.flash(f"BAKED {res.frames}F ({mb:.1f}MB)")
            except StopIteration:
                self.bake_job = None
                self.pending_after_bake = None
                Bridge.send("v4dProgress", {"busy": False})
            except Exception as e:
                self.bake_job = None
                self.pending_after_bake = None
                Bridge.send("v4dProgress", {"busy": False})
                self.flash(f"BAKE ERR: {e}"[:40].upper(), err=True)
            return

        if not self._handle_clicks():
            if pyxel.btn(pyxel.MOUSE_BUTTON_LEFT) and self.view_top <= pyxel.mouse_y < self.view_bot:
                if self.drag:
                    self.yaw -= (pyxel.mouse_x - self.drag[0]) * 1.1
                    self.pitch = max(-5.0, min(65.0, self.pitch + (pyxel.mouse_y - self.drag[1]) * 0.6))
                self.drag = (pyxel.mouse_x, pyxel.mouse_y)
            else:
                self.drag = None

            if (pyxel.btn(pyxel.MOUSE_BUTTON_RIGHT) or pyxel.btn(pyxel.MOUSE_BUTTON_MIDDLE)) and self.view_top <= pyxel.mouse_y < self.view_bot:
                if self.pan_drag:
                    dx = pyxel.mouse_x - self.pan_drag[0]
                    dy = pyxel.mouse_y - self.pan_drag[1]
                    a, b = math.radians(self.yaw), math.radians(self.pitch)
                    sc = 2.0 * self.dist * math.tan(math.radians(16.0)) / max(1, self.view_bot - self.view_top)
                    rx, rz = math.cos(a), -math.sin(a)
                    ux, uy, uz = -math.sin(a) * math.sin(b), math.cos(b), -math.cos(a) * math.sin(b)
                    self.pan[0] = max(-3.0, min(3.0, self.pan[0] - rx * dx * sc + ux * dy * sc))
                    self.pan[1] = max(-3.0, min(3.0, self.pan[1] + uy * dy * sc))
                    self.pan[2] = max(-3.0, min(3.0, self.pan[2] - rz * dx * sc + uz * dy * sc))
                self.pan_drag = (pyxel.mouse_x, pyxel.mouse_y)
            else:
                self.pan_drag = None

        if pyxel.mouse_wheel:
            self.dist = max(0.8, min(12.0, self.dist * (0.9 ** pyxel.mouse_wheel)))
        if pyxel.btn(pyxel.KEY_LEFT):
            self.yaw -= 3.0
        if pyxel.btn(pyxel.KEY_RIGHT):
            self.yaw += 3.0
        if pyxel.btn(pyxel.KEY_UP):
            self.dist = max(0.8, self.dist * 0.97)
        if pyxel.btn(pyxel.KEY_DOWN):
            self.dist = min(12.0, self.dist * 1.03)

        if pyxel.btnp(pyxel.KEY_SPACE):
            self.playing = not self.playing
        if pyxel.btnp(pyxel.KEY_V):
            self._cycle_view_mode()
        if pyxel.btnp(pyxel.KEY_G):
            self._toggle_brush()
        if pyxel.btnp(pyxel.KEY_B):
            self.start_bake()
        if pyxel.btnp(pyxel.KEY_M):
            self._step_motion(1)
        if pyxel.btnp(pyxel.KEY_N):
            self._step_motion(-1)
        if pyxel.btnp(pyxel.KEY_C):
            self.count_idx = (self.count_idx + 1) % len(SPLAT_COUNTS)
            self.rebuild_sampler()
        if pyxel.btnp(pyxel.KEY_F):
            self.fps_idx = (self.fps_idx + 1) % len(BAKE_FPS_OPTIONS)
            self.baked = None
        if pyxel.btnp(pyxel.KEY_LEFTBRACKET):
            self.splat_scale = max(0.4, round(self.splat_scale - 0.15, 2))
        if pyxel.btnp(pyxel.KEY_RIGHTBRACKET):
            self.splat_scale = min(2.5, round(self.splat_scale + 0.15, 2))
        if pyxel.btnp(pyxel.KEY_R):
            self._command("view:reset")
        if pyxel.btnp(pyxel.KEY_H):
            self.show_help = not self.show_help
        if pyxel.btnp(pyxel.KEY_COMMA):
            self.playing = False
            if self.baked is not None:
                self.bake_frame = (self.bake_frame - 1) % self.baked.frames
            else:
                self.t = max(0.0, self.t - 1.0 / 30.0)
        if pyxel.btnp(pyxel.KEY_PERIOD):
            self.playing = False
            if self.baked is not None:
                self.bake_frame = (self.bake_frame + 1) % self.baked.frames
            else:
                self.t += 1.0 / 30.0
        if not self.compact_ui:
            if pyxel.btnp(pyxel.KEY_1):
                self.do_export("ply")
            if pyxel.btnp(pyxel.KEY_2):
                self.do_export("splat")
            if pyxel.btnp(pyxel.KEY_3):
                self.do_export("zip")
            if pyxel.btnp(pyxel.KEY_4):
                self.do_export("html")

        if self.playing:
            if self.baked is not None:
                self.bake_frame = (self.bake_frame + self.baked.fps / 30.0) % self.baked.frames
            else:
                self.t += 1.0 / 30.0

        if self.baked is not None:
            kf = int(self.bake_frame) % self.baked.frames
            if kf != self.last_sent_frame:
                self.last_sent_frame = kf
                Bridge.send("v4dFrame", {"frame": kf, "frames": self.baked.frames, "playing": self.playing})

    def _sync_scene_for_draw(self):
        if self.rig is None or self.sampler is None:
            return
        if self.baked is not None:
            k = int(self.bake_frame) % self.baked.frames
            if self.view_mode in (0, 1, 3):
                c, u, v = SE.decode_rigged_preview(self.baked, k, self.preview_idx, self.preview_boost)
                self.splat_node.sync(c, u, v, self.splat_scale)
            if self.view_mode in (1, 2):
                m_meta = self.motions.by_id.get(self.baked.motion_id)
                sec = float(m_meta.get("seconds", 1.0)) if m_meta else 1.0
                t = (sec * k / self.baked.frames) if (m_meta and m_meta.get("loop", True)) else (sec * k / max(1, self.baked.frames - 1))
                ew = None
                if self.baked.expr_weights.shape[1] > 0:
                    ew = np.zeros(len(V.EXPRESSION_NAMES), dtype=np.float32)
                    for slot_i, ei in enumerate(self.baked.morph_expr_ids):
                        ew[ei] = self.baked.expr_weights[k, slot_i]
                q, tr = self.rig.pose_at(self.baked.motion_id, t)
                self.mesh_node.sync(self.rig.skin_parts(q, tr, apply_face_yaw=True, expr_w=ew))
        else:
            mid = self.current_motion_id()
            ew = self._live_expr_weights(mid, self.t)
            q, tr = self.rig.pose_at(mid, self.t)
            skinned = self.rig.skin_parts(q, tr, apply_face_yaw=True, expr_w=ew)
            if self.view_mode in (1, 2):
                self.mesh_node.sync(skinned)
            if self.view_mode in (0, 1, 3):
                sf = SE.eval_splat_frame(skinned, self.sampler, full_rot=False, subset=self.preview_idx)
                self.splat_node.sync(
                    sf.centers,
                    sf.u_vec * self.preview_boost,
                    sf.v_vec * self.preview_boost,
                    self.splat_scale,
                )

    def _btn(self, x, y, w, h, label, bg=2, fg=7):
        pyxel.rect(x, y, w, h, bg)
        pyxel.rectb(x, y, w, h, 5)
        tx = x + max(2, (w - len(label) * 4) // 2)
        pyxel.text(tx, y + 3, label, fg)

    def draw(self):
        pyxel.cls(BG)
        self._sync_scene_for_draw()
        vt, vb = self.view_top, self.view_bot

        if self.rig is not None:
            h = self.rig.m.height
            a, b = math.radians(self.yaw), math.radians(self.pitch)
            tgt = Vec3(self.pan[0], h * 0.52 + self.pan[1], self.pan[2])
            eye = Vec3(
                tgt.x + math.sin(a) * math.cos(b) * self.dist,
                tgt.y + math.sin(b) * self.dist,
                tgt.z + math.cos(a) * math.cos(b) * self.dist,
            )
            self.cam.transform = Mat4.look_at(eye, tgt, Vec3.UP)

            if self.view_mode == 0 or self.view_mode == 3:
                self.splat_stage.draw(0, vt, W, vb - vt)
            elif self.view_mode == 2:
                self.mesh_stage.draw(0, vt, W, vb - vt)
            else:
                pyxel.clip(0, vt, W // 2, vb - vt)
                self.mesh_stage.draw(0, vt, W, vb - vt)
                pyxel.clip(W // 2, vt, W - W // 2, vb - vt)
                self.splat_stage.draw(0, vt, W, vb - vt)
                pyxel.clip()
                pyxel.line(W // 2, vt, W // 2, vb - 1, 10)
                pyxel.text(6, vt + 4, "MESH", 13)
                pyxel.text(W // 2 + 6, vt + 4, "RETRO 4DGS", 11)

        if not self.compact_ui:
            pyxel.rect(0, 0, W, vt, 2)
            self._btn(4, 1, 42, 11, VIEW_MODES[self.view_mode], bg=5, fg=7)
            self._btn(49, 1, 45, 11, BRUSH_MODES[self.brush_idx], bg=2, fg=11 if self.brush_idx == 0 else 10)
            cnt = SPLAT_COUNTS[self.count_idx]
            self._btn(97, 1, 48, 11, f"{cnt // 1000}K SPLAT", bg=2, fg=7)
            self._btn(148, 1, 38, 11, f"{BAKE_FPS_OPTIONS[self.fps_idx]}FPS", bg=2, fg=7)
            self._btn(189, 1, 92, 11, self.current_motion_id().upper()[:20], bg=2, fg=10)
            self._btn(284, 1, 32, 11, "BAKE", bg=11 if self.baked is None else 3, fg=0)

            pyxel.rect(0, vb, W, H - vb, 2)
            self._btn(4, vb + 2, 32, 12, "PAUSE" if self.playing else "PLAY", bg=5, fg=7)
            self._btn(40, vb + 2, 14, 12, "-", bg=2, fg=7)
            pyxel.text(58, vb + 5, f"SZ:{self.splat_scale:.1f}X", 7)
            self._btn(92, vb + 2, 14, 12, "+", bg=2, fg=7)
            pyxel.rect(111, vb + 6, 80, 4, 1)
            if self.baked is not None:
                frac = (int(self.bake_frame) + 1) / max(1, self.baked.frames)
                pyxel.rect(111, vb + 6, max(2, int(80 * frac)), 4, 11)
            else:
                m_meta = self.motions.by_id.get(self.current_motion_id(), {})
                dur = max(0.1, float(m_meta.get("seconds", 1.0)))
                pyxel.rect(111, vb + 6, max(2, int(80 * ((self.t % dur) / dur))), 4, 10)
            self._btn(196, vb + 2, 28, 12, "PLY", bg=5, fg=7)
            self._btn(227, vb + 2, 28, 12, "SPL", bg=5, fg=7)
            self._btn(258, vb + 2, 28, 12, "ZIP", bg=5, fg=7)
            self._btn(289, vb + 2, 28, 12, "HTM", bg=10, fg=0)

        if self.rig is not None:
            mode_tag = "BAKED 4D" if self.baked is not None else "LIVE"
            if self.view_mode == 1:
                pyxel.text(6, vt + 13, f"{mode_tag} {self.rig.m.title[:14].upper()}", 6)
                if self.baked is not None:
                    mb = len(self.baked.compressed_r3dg) / (1024 * 1024)
                    pyxel.text(W // 2 + 6, vt + 13, f"{int(self.bake_frame)+1}/{self.baked.frames}F ({mb:.1f}MB)", 6)
            else:
                info = f"{mode_tag}  {self.rig.m.title[:18].upper()}"
                if self.baked is not None:
                    mb = len(self.baked.compressed_r3dg) / (1024 * 1024)
                    info += f"  {int(self.bake_frame)+1}/{self.baked.frames}F ({mb:.1f}MB)"
                pyxel.text(4, vt + 4, info, 6)
            if not self.compact_ui:
                pyxel.text(W - 44, vt + 4, "[H] HELP", 13)

        if self.bake_job is not None:
            done, total = self.bake_prog
            pyxel.rect(48, H // 2 - 14, W - 96, 28, 1)
            pyxel.rectb(48, H // 2 - 14, W - 96, 28, 11)
            pyxel.text(56, H // 2 - 8, f"BAKING 4DGS: {done} / {total} FRAMES", 7)
            pyxel.rect(56, H // 2 + 2, W - 112, 6, 2)
            pyxel.rect(56, H // 2 + 2, int((W - 112) * done / max(1, total)), 6, 11)

        if self.note_t > 0 and not self.compact_ui:
            self.note_t -= 1.0 / 30.0
            pyxel.rect(4, vb - 12, min(W - 8, len(self.note) * 4 + 8), 10, 0)
            pyxel.text(8, vb - 10, self.note, 10)

        if self.show_help and not self.compact_ui:
            pyxel.rect(22, 28, W - 44, H - 56, 1)
            pyxel.rectb(22, 28, W - 44, H - 56, 12)
            for idx, ln in enumerate([
                "VRM RETRO 4DGS STUDIO - SHORTCUTS",
                "DROP .VRM/.GLB MODEL OR .VRMA/.FBX MOTION",
                "L-DRAG: ORBIT  R-DRAG: MOVE  WHEEL: ZOOM",
                "R: RESET  SPACE: PLAY/PAUSE  , / .: STEP",
                "M / N: NEXT/PREV MOTION (231+ CC0)",
                "V: VIEW (4DGS / SPLIT / MESH / DOTS)",
                "G: BRUSH (DITHER / SOLID)  [/]: SCALE",
                "C: SPLAT COUNT (8K..50K)   F: FPS (15/24/30)",
                "B: BAKE 4DGS   1:PLY  2:SPLAT  3:ZIP  4:HTML",
            ]):
                pyxel.text(30, 36 + idx * 14, ln, 10 if idx == 0 else 7)


ARG = next((a for a in sys.argv[1:] if a.lower().endswith((".vrm", ".glb"))), None)
if __name__ == "__main__":
    App(ARG)
