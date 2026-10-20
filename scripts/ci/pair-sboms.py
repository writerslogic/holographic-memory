# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect release artefacts into one directory with one CycloneDX SBOM per artefact.

usage: pair-sboms.py <downloaded-artifacts-dir> <out-dir>

<downloaded-artifacts-dir> holds one sub-directory per uploaded workflow artefact. Every
distributable (wheel, sdist, npm tarball, crate) is copied to <out-dir>/<name> and its SBOM
to <out-dir>/<name>.cdx.json. The release is refused if a distributable has no SBOM or an
SBOM has no distributable. Prints the distributable names as a JSON array (for a matrix).
"""
import json
import pathlib
import re
import shutil
import sys

DISTRIBUTABLE = (".whl", ".tar.gz", ".tgz", ".crate")


def sbom_for(dist: pathlib.Path, sboms: list[pathlib.Path], alone: bool) -> pathlib.Path | None:
    stem_of = lambda p: p.name.removesuffix(".cdx.json")
    if alone and len(sboms) == 1:
        return sboms[0]
    name = dist.name
    for sbom in sboms:
        stem = stem_of(sbom)
        root = re.fullmatch(r"(.+)-root-(\d.*)", stem)
        if root and name == f"{root[1]}-{root[2]}.tgz":
            return sbom
        if name.startswith(stem + "-") and name[len(stem) + 1].isdigit():
            return sbom
    return None


def main(argv: list[str]) -> int:
    source, out = pathlib.Path(argv[0]), pathlib.Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    names: list[str] = []
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        files = sorted(p for p in directory.iterdir() if p.is_file())
        sboms = [p for p in files if p.name.endswith(".cdx.json")]
        dists = [p for p in files if p.name.endswith(DISTRIBUTABLE)]
        used: set[pathlib.Path] = set()
        for dist in dists:
            sbom = sbom_for(dist, sboms, len(dists) == 1)
            if sbom is None:
                problems.append(f"{directory.name}: no SBOM for {dist.name}")
                continue
            used.add(sbom)
            shutil.copyfile(dist, out / dist.name)
            shutil.copyfile(sbom, out / f"{dist.name}.cdx.json")
            names.append(dist.name)
        for sbom in sboms:
            if sbom not in used:
                problems.append(f"{directory.name}: SBOM {sbom.name} matches no artefact")
        if not dists:
            problems.append(f"{directory.name}: no distributable found")
    if len(names) != len(set(names)):
        problems.append("duplicate artefact names across artifacts")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(json.dumps(sorted(names)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
