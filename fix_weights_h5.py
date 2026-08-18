"""One-off repair for artifacts/cnn/model.weights.h5.

The file was saved on Windows by a Keras 3 build that builds internal HDF5
group paths with `os.path.join`, which yields backslash-joined names (e.g.
``layers\\dense``) instead of true nested HDF5 groups (``layers/dense``).
h5py treats that as a single flat group whose name happens to contain a
backslash character. Keras writes and reads the file with the same buggy
path-join logic, so round-tripping on Windows appears to work -- but on
Linux (Streamlit Community Cloud), `os.path.join` uses "/" and Keras looks
for a genuinely nested "layers/dense" group that doesn't exist, so
`model.load_weights()` silently finds nothing and raises.

This script copies every flat "a\\b" group into a real nested "a/b" group,
producing a file that loads identically on Windows and Linux.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import h5py

SRC = Path("artifacts/cnn/model.weights.h5")
BACKUP = Path("artifacts/cnn/model.weights.h5.bak")


def copy_recursive(src: h5py.Group, dst: h5py.Group) -> None:
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            sub = dst.require_group(key)
            sub.attrs.update(item.attrs)
            copy_recursive(item, sub)
        else:  # Dataset
            dst.create_dataset(key, data=item[()])
    dst.attrs.update({k: v for k, v in src.attrs.items() if k not in dst.attrs})


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found")

    shutil.copy2(SRC, BACKUP)
    tmp = SRC.with_suffix(".h5.tmp")

    with h5py.File(SRC, "r") as fsrc, h5py.File(tmp, "w") as fdst:
        fdst.attrs.update(fsrc.attrs)
        for key, item in fsrc.items():
            # Backslash-joined top-level names -> real nested groups.
            dest_path = "/".join(key.split("\\"))
            if isinstance(item, h5py.Group):
                dst_group = fdst.require_group(dest_path)
                dst_group.attrs.update(item.attrs)
                copy_recursive(item, dst_group)
            else:
                fdst.create_dataset(dest_path, data=item[()])

    tmp.replace(SRC)
    print(f"Repaired {SRC} (backup at {BACKUP}).")

    # Verify: no more backslash-named groups anywhere.
    with h5py.File(SRC, "r") as f:
        bad = []

        def check(name: str) -> None:
            if "\\" in name.split("/")[-1]:
                bad.append(name)

        f.visit(check)
        if bad:
            raise SystemExit(f"Repair failed, still found backslash names: {bad}")
        print("Verified: no backslash-joined group names remain.")
        print("Top-level groups:", list(f.keys()))
        print("layers/ subgroups:", list(f["layers"].keys()))


if __name__ == "__main__":
    main()
