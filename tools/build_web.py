"""Build the browser version: web/vrm_retro_4dgs.pyxapp from app/, and the itch.io upload
vrm_retro_4dgs_itch.zip (index.html + .pyxapp + sample.vrm).

    python tools/build_web.py
"""
# SPDX-License-Identifier: MIT
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
WEB = ROOT / "web"
FILES = ("main.py", "splat_engine.py", "vrm_io.py", "vrma_io.py", "fbx_io.py", "motions.npz", "expr_demo.vrma")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp) / "app"
        app.mkdir()
        for f in FILES:
            shutil.copy(APP / f, app / f)
        subprocess.run(["pyxel", "package", str(app), str(app / "main.py")], cwd=tmp, check=True)
        shutil.copy(Path(tmp) / "app.pyxapp", WEB / "vrm_retro_4dgs.pyxapp")
    out = ROOT / "vrm_retro_4dgs_itch.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("index.html", "vrm_retro_4dgs.pyxapp", "sample.vrm"):
            z.write(WEB / f, f)
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
