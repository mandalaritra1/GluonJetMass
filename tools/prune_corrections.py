"""Prune correctionFiles/ down to what the code actually loads.

This is rule-based, not heuristic: the keep-set is reconstructed from the JEC tag
mapping in python/corrections.py (GetJetCorrections) plus the JER tags, the two
jet algorithms the code uses (AK8PFPuppi, AK4PFPuppi), and the small set of
ancillary files (golden JSONs, puWeights, jetvetomap, ps_weight JSONs).

Default mode is --dry-run: print to-delete and projected sizes only. Pass --apply
to actually delete.

Usage (from repo root):
    python tools/prune_corrections.py            # dry-run
    python tools/prune_corrections.py --apply    # actually delete
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORR = REPO / "correctionFiles"

# ----------------------------------------------------------------------------
# Keep-set rules (mirror GetJetCorrections in python/corrections.py)
# ----------------------------------------------------------------------------
JEC_TAGS_MC = {
    "Summer19UL18_V5_MC",
    "Summer19UL17_V6_MC",
    "Summer19UL16_V7_MC",     # used for both 2016 and 2016APV MC
}
JEC_TAGS_DATA = {
    # 2018
    "Summer19UL18_RunA_V6_DATA",
    "Summer19UL18_RunB_V6_DATA",
    "Summer19UL18_RunC_V6_DATA",
    "Summer19UL18_RunD_V6_DATA",
    # 2017
    "Summer19UL17_RunB_V6_DATA",
    "Summer19UL17_RunC_V6_DATA",
    "Summer19UL17_RunD_V6_DATA",
    "Summer19UL17_RunE_V6_DATA",
    "Summer19UL17_RunF_V6_DATA",
    # 2016
    "Summer19UL16_RunFGH_V7_DATA",
    # 2016APV
    "Summer19UL16APV_RunBCD_V7_DATA",
    "Summer19UL16APV_RunEF_V7_DATA",
}
JEC_TAGS = JEC_TAGS_MC | JEC_TAGS_DATA

JER_TAGS = {
    "Summer19UL18_JRV2_MC",
    "Summer19UL17_JRV3_MC",      # JRV3 is used; JRV2 dir is dropped
    "Summer20UL16_JRV3_MC",
    "Summer20UL16APV_JRV3_MC",
}

JET_ALGOS = ("AK8PFPuppi", "AK4PFPuppi")

# JEC file pattern: <tag>_<level>_<algo>.<ext>
#   MC loads:   L1FastJet, L2Relative, L3Absolute, Uncertainty, UncertaintySources
#   DATA loads: L1FastJet, L2Relative, L3Absolute, L2L3Residual
# Extensions actually loaded: .jec.txt for JEC levels, .junc.txt for Uncertainty/UncertaintySources
JEC_LEVELS = ("L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual",
              "Uncertainty", "UncertaintySources")
JEC_KEEP_EXTS = (".jec.txt", ".junc.txt")  # plain .txt and L2Residual are dropped

# JER pattern: <tag>_PtResolution_<algo>.jr.txt and <tag>_SF_<algo>.jersf.txt
JER_KEEP_EXTS = (".jr.txt", ".jersf.txt")


def jec_keep(path: Path) -> bool:
    """True if path is a JEC file we want to keep."""
    parts = path.relative_to(CORR / "JEC").parts
    if len(parts) < 2:
        return False
    tag, fname = parts[0], parts[-1]
    if tag not in JEC_TAGS:
        return False
    # Must match one of the loaded extensions.
    if not any(fname.endswith(ext) for ext in JEC_KEEP_EXTS):
        return False
    # Filename: <tag>_<level>_<algo><ext>
    stem = fname
    for ext in JEC_KEEP_EXTS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    # stem should start with the tag and end with one of the algos
    if not stem.startswith(tag + "_"):
        return False
    if not any(stem.endswith("_" + algo) for algo in JET_ALGOS):
        return False
    # The level token in the middle:
    middle = stem[len(tag) + 1:]
    for algo in JET_ALGOS:
        suffix = "_" + algo
        if middle.endswith(suffix):
            level = middle[: -len(suffix)]
            # For unc-source files the level looks like "UncertaintySources" and
            # the actual source name is encoded as the dict key inside the file,
            # not in the filename. So matching the level prefix is sufficient.
            return level in JEC_LEVELS
    return False


def jer_keep(path: Path) -> bool:
    parts = path.relative_to(CORR / "JER").parts
    if len(parts) < 2:
        return False
    tag, fname = parts[0], parts[-1]
    if tag not in JER_TAGS:
        return False
    if not any(fname.endswith(ext) for ext in JER_KEEP_EXTS):
        return False
    # Algo token at end
    for ext in JER_KEEP_EXTS:
        if fname.endswith(ext):
            stem = fname[: -len(ext)]
            break
    return any(stem.endswith("_" + algo) for algo in JET_ALGOS)


def top_level_keep(path: Path) -> bool:
    """Small ancillary files at correctionFiles/ root we keep wholesale."""
    name = path.name
    # Golden JSONs
    if name.startswith("Cert_") and name.endswith(".txt"):
        return True
    # Trigger prescale JSONs (referenced by utils.py / triggerProcessor.py)
    if name.startswith("ps_weight") and name.endswith(".json"):
        return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="actually delete; default is dry-run")
    args = p.parse_args()

    if not CORR.is_dir():
        sys.exit(f"correctionFiles/ not found at {CORR}")

    keep, drop = [], []
    for path in CORR.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(CORR)
        first = rel.parts[0]
        # Always keep these whole subtrees:
        if first in ("puWeights", "jetvetomap", "goldenJsons", "SFs"):
            keep.append(path); continue
        if first == "JEC":
            (keep if jec_keep(path) else drop).append(path); continue
        if first == "JER":
            (keep if jer_keep(path) else drop).append(path); continue
        # Top-level loose files (Cert_*, ps_weight_*)
        if len(rel.parts) == 1:
            (keep if top_level_keep(path) else drop).append(path); continue
        # Anything else under an unknown subdir: drop (be conservative? keep)
        drop.append(path)

    def total(paths):
        return sum(pth.stat().st_size for pth in paths)

    k_sz, d_sz = total(keep), total(drop)
    print(f"keep: {len(keep):5d} files, {k_sz/1e6:8.1f} MB")
    print(f"drop: {len(drop):5d} files, {d_sz/1e6:8.1f} MB")
    print(f"total before: {(k_sz+d_sz)/1e6:8.1f} MB  ->  after: {k_sz/1e6:8.1f} MB")

    # Show a sample of drops grouped by subtree
    from collections import Counter
    by_top = Counter(str(pth.relative_to(CORR).parts[0]) for pth in drop)
    print("\ndrop-by-subtree:")
    for k, v in sorted(by_top.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v:6d} files")

    # Show a few example dropped paths (per subtree)
    print("\nexample dropped paths:")
    shown_top = set()
    for pth in drop:
        top = pth.relative_to(CORR).parts[0]
        if top in shown_top:
            continue
        shown_top.add(top)
        print(f"  [{top}] {pth.relative_to(REPO)}")
    print(f"\n(showed first per subtree out of {len(drop)} total drops)")

    if not args.apply:
        print("\n[dry-run] Nothing deleted. Re-run with --apply to delete.")
        return

    for pth in drop:
        try:
            pth.unlink()
        except OSError as e:
            print(f"WARN could not delete {pth}: {e}")
    # Remove empty directories left behind
    for d in sorted([d for d in CORR.rglob("*") if d.is_dir()], key=lambda x: -len(x.parts)):
        try:
            d.rmdir()
        except OSError:
            pass
    print(f"\nDeleted {len(drop)} files; new size: {sum(p.stat().st_size for p in CORR.rglob('*') if p.is_file())/1e6:.1f} MB")


if __name__ == "__main__":
    main()
