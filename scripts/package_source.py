"""Package working sources and every pinned submodule, excluding Git and caches."""

# ruff: noqa: INP001 -- standalone repository packaging utility

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def collect(root, prefix, files, repositories):
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    repositories.append({"path": prefix or ".", "commit": commit})
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        info, raw_path = entry.split(b"\t", 1)
        mode, sha, stage = info.decode().split()
        if stage != "0":
            raise RuntimeError("Unresolved index entry")
        relative = raw_path.decode("utf-8")
        source = root / relative
        name = f"{prefix}/{relative}" if prefix else relative
        if mode == "160000":
            if not (source / ".git").exists():
                raise RuntimeError(f"Submodule not initialized: {name}")
            actual = git(source, "rev-parse", "HEAD").decode().strip()
            if actual != sha:
                raise RuntimeError(f"Submodule moved from pinned revision: {name}")
            collect(source, name, files, repositories)
        elif source.exists() or source.is_symlink():
            files[name] = (source, int(mode, 8))


def canonical_bytes(path, mode):
    if mode == 0o120000 and path.is_symlink():
        return os.readlink(path).encode("utf-8")
    data = path.read_bytes()
    if b"\0" not in data:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            # Windows Git checkout uses CRLF. Ship source text with Linux line endings.
            data = data.replace(b"\r\n", b"\n")
    return data


def main():
    root = Path(__file__).resolve().parents[1]
    out = Path(sys.argv[1]).resolve()
    files, repositories = {}, []
    collect(root, "", files, repositories)
    for raw in git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        path = root / name
        if path.is_file():
            files[name] = (path, 0o100755 if name.endswith(".sh") else 0o100644)
    # Evidence lives under validation/artifacts, which the general-purpose
    # .gitignore excludes. Include this small, curated validation tree even
    # though build outputs elsewhere remain excluded.
    for path in (root / "validation").rglob("*"):
        if path.is_file() and not path.name.endswith((".pyc", ".pyo")):
            name = path.relative_to(root).as_posix()
            files[name] = (path, 0o100644)
    manifest = {
        "root_commit": repositories[0]["commit"],
        "repositories": repositories,
        "file_count": len(files) + ("SOURCE_MANIFEST.json" not in files),
        "git_metadata_included": False,
        "validation": "CPU, single-device NPU reference, and native Megatron PT/midtrain/SFT passed; verl/vLLM acceptance pending",
    }
    (root / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files["SOURCE_MANIFEST.json"] = (root / "SOURCE_MANIFEST.json", 0o100644)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for index, (name, (source, mode)) in enumerate(sorted(files.items())):
            info = zipfile.ZipInfo("shensi/" + name)
            info.create_system = 3
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, canonical_bytes(source, mode))
            if index % 5000 == 0:
                print(f"Packaged {index}/{len(files)} files", flush=True)
    with zipfile.ZipFile(out) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt ZIP entry: {bad}")
        if len(archive.namelist()) != len(files):
            raise RuntimeError("Archive entry count mismatch")
        for essential in (
            "ASCEND_ADAPTATION.md",
            "SOURCE_MANIFEST.json",
            "pyproject.ascend.toml",
            "src/shensi/utils/accelerator.py",
            "validation/2026-10-02/summary.json",
            "validation/2026-10-04/artifacts/ascend_actual_retry/pytest.xml",
            "validation/2026-10-04/artifacts/ascend_actual_retry/reference_fp32/report.json",
            "validation/2026-10-04/native_final/artifacts/native_210_trial4/native_report.json",
        ):
            archive.getinfo("shensi/" + essential)
    with out.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "archive": str(out),
                "bytes": out.stat().st_size,
                "entries": len(files),
                "repositories": len(repositories),
                "sha256": digest,
                "crc_verification": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
