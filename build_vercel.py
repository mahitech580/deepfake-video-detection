"""Build the compact LiteRT model and copy web assets for Vercel."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    source_model = ROOT / "deepfake_model.h5"
    if not source_model.is_file() or source_model.stat().st_size < 1_000_000:
        raise SystemExit(
            "The trained deepfake_model.h5 Git LFS object is missing. "
            "Enable Git LFS for the Vercel project and redeploy."
        )

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")

    # Keep TensorFlow confined to a temporary build-only environment. It is
    # not part of the deployed function, which uses the much smaller LiteRT runtime.
    with tempfile.TemporaryDirectory(prefix="deepfake-model-build-") as temp_dir:
        venv_dir = Path(temp_dir) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        python = venv_dir / "bin" / "python"
        subprocess.run(
            [
                str(python), "-m", "pip", "install", "--no-cache-dir",
                "-r", str(ROOT / "requirements-build.txt"),
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )
        subprocess.run(
            [str(python), str(ROOT / "convert_model.py")],
            check=True,
            cwd=ROOT,
            env=env,
        )

    model = ROOT / "deepfake_model.tflite"
    if not model.is_file() or model.stat().st_size < 1_000_000:
        raise SystemExit("Vercel build did not produce a valid LiteRT model artifact.")

    # Vercel serves public/ assets through its CDN. Keep the HTML templates in
    # templates/ for Flask, and copy only static assets (images/CSS/JS/etc).
    static_source = ROOT / "static"
    static_target = ROOT / "public" / "static"
    if static_source.is_dir():
        shutil.copytree(
            static_source,
            static_target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.html"),
        )

    print(
        f"Vercel build prepared {model.name} ({model.stat().st_size:,} bytes) "
        f"and static assets.",
        flush=True,
    )


if __name__ == "__main__":
    main()
