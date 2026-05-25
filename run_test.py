"""Run one chunk of local ROOT files through the processors under coffea 2025+.

Drop NanoAODv9 ROOT files into test_files/ and run:

    /Users/aritra/Projects/testing_ground/.venv/bin/python run_local_test.py --proc dijet

The processors infer dataset/IOV/HT-bin from `events.metadata['dataset']` and the
`/store/{mc,data}/...` LFN inside `events.metadata['filename']`. To exercise that
unchanged code path without faking metadata, this runner:

  1. Looks each local file's basename up in fileset_*.json to recover its real
     DAS dataset name + original LFN.
  2. Stages a symlink at test_files/store/{mc,data}/.../<basename>.root mirroring
     the LFN, and uses that symlink path in the fileset.
  3. Builds the fileset keyed by DAS dataset name so IOV detection works.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # so correctionFiles/... relative paths resolve
sys.path.insert(0, str(HERE))

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from python.dijetProcessor import DijetProcessor
from python.trijetProcessor import TrijetProcessor
from python.triggerProcessor import triggerProcessor


def _load_filesets() -> dict[str, tuple[str, str]]:
    """Map: basename(file).root -> (das_dataset_name, original_LFN).

    Scans both fileset_*.json (LFNs) and fileset_*_wRedirs.json (full xrootd URLs).
    """
    out: dict[str, tuple[str, str]] = {}
    for fn in glob.glob("fileset_*.json"):
        try:
            top = json.load(open(fn))
        except Exception:
            continue
        for ds_group, datasets in top.items():
            if not isinstance(datasets, dict):
                continue
            for das_name, files in datasets.items():
                if not isinstance(files, list):
                    continue
                for f in files:
                    base = os.path.basename(f)
                    # Reduce redirector-prefixed paths to /store/...
                    idx = f.find("/store/")
                    lfn = f[idx:] if idx >= 0 else f
                    # Prefer LFN-form (starts with /store/) when both exist
                    if base not in out or lfn.startswith("/store/"):
                        out[base] = (das_name, lfn)
    return out


def _stage_lfn_symlink(real_path: Path, lfn: str, root: Path) -> Path:
    """Symlink test_files/store/.../<basename> -> real_path; return the symlink."""
    assert lfn.startswith("/store/"), f"expected /store/... LFN, got: {lfn}"
    link = root / lfn.lstrip("/")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(real_path)
    return link


def build_fileset(test_dir: Path) -> tuple[dict[str, list[str]], bool]:
    real_files = sorted(p for p in test_dir.glob("*.root") if not p.is_symlink())
    if not real_files:
        sys.exit(f"No .root files found in {test_dir}. xrdcp or scp some there first.")
    lookup = _load_filesets()
    fileset: dict[str, list[str]] = {}
    saw_mc = saw_data = False
    for f in real_files:
        if f.name not in lookup:
            print(f"  ! {f.name}: not in any fileset_*.json; skipping")
            continue
        das, lfn = lookup[f.name]
        link = _stage_lfn_symlink(f, lfn, test_dir)
        fileset.setdefault(das, []).append(str(link))
        if "/store/mc/" in lfn:
            saw_mc = True
        if "/store/data/" in lfn:
            saw_data = True
    if not fileset:
        sys.exit("No files matched any entry in fileset_*.json — check basenames.")
    return fileset, saw_mc and not saw_data  # is_pure_mc


def build_fileset_from_json(path: str, dataset_filter: str | None,
                            redirector: str, n_files: int | None) -> tuple[dict[str, list[str]], bool]:
    """Build a fileset from a fileset_*.json on disk (no local staging).

    The JSON shape is {<top>: {<das_name>: [files...]}}. Files are LFNs
    (need a redirector prepended) or already-xrootd URLs (used as-is).
    dataset_filter is a case-insensitive substring matched against DAS names.
    """
    with open(path) as f:
        top = json.load(f)
    fileset: dict[str, list[str]] = {}
    saw_mc = saw_data = False
    for ds_group, datasets in top.items():
        if not isinstance(datasets, dict):
            continue
        for das_name, files in datasets.items():
            if dataset_filter and dataset_filter.lower() not in das_name.lower():
                continue
            urls = []
            for f in files:
                if f.startswith("root://"):
                    urls.append(f)
                elif f.startswith("/store/"):
                    # CMS xrootd URLs use double slash between host and /store/...
                    urls.append(redirector.rstrip("/") + "/" + f)
                else:
                    # already an absolute local path or something else; pass through
                    urls.append(f)
            if n_files:
                urls = urls[:n_files]
            if not urls:
                continue
            fileset[das_name] = urls
            # sniff /store/{mc,data}/
            joined = " ".join(urls)
            if "/store/mc/" in joined:
                saw_mc = True
            if "/store/data/" in joined:
                saw_data = True
    if not fileset:
        sys.exit(f"No datasets in {path} matched filter {dataset_filter!r}.")
    return fileset, saw_mc and not saw_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", choices=["dijet", "trijet", "trigger"], default="dijet")
    ap.add_argument("--data", action="store_true",
                    help="force data mode (do_gen=False); default auto-detects from LFN")
    ap.add_argument("--mc", action="store_true",
                    help="force MC mode (do_gen=True); default auto-detects from LFN")
    ap.add_argument("--maxchunks", type=int, default=1,
                    help="max chunks to process; <=0 means run the whole file")
    ap.add_argument("--chunksize", type=int, default=10_000)
    ap.add_argument("--test-dir", default="test_files")
    ap.add_argument("--fileset", default=None,
                    help="path to a fileset_*.json; if given, read from xrootd "
                         "instead of staging local files from --test-dir")
    ap.add_argument("--dataset", default=None,
                    help="case-insensitive substring filter on DAS dataset name "
                         "(used with --fileset)")
    ap.add_argument("--nfiles", type=int, default=1,
                    help="(with --fileset) take only the first N files per matching "
                         "dataset; 0 means all")
    ap.add_argument("--redirector", default="root://cmsxrootd.fnal.gov/",
                    help="xrootd redirector to prepend to bare /store/... LFNs")
    args = ap.parse_args()

    if args.fileset:
        fileset, pure_mc = build_fileset_from_json(
            args.fileset, args.dataset, args.redirector,
            n_files=(None if args.nfiles == 0 else args.nfiles),
        )
    else:
        test_dir = (HERE / args.test_dir).resolve()
        fileset, pure_mc = build_fileset(test_dir)

    if args.data:
        is_data = True
    elif args.mc:
        is_data = False
    else:
        is_data = not pure_mc

    # The processors have a single global do_gen flag, so when both MC and data
    # files are present we must restrict the fileset to whichever mode we're running.
    needle = "/store/data/" if is_data else "/store/mc/"
    fileset = {k: [p for p in v if needle in p] for k, v in fileset.items()}
    fileset = {k: v for k, v in fileset.items() if v}
    if not fileset:
        sys.exit(f"No {('data' if is_data else 'MC')} files in test_files/ matched any fileset_*.json.")

    print(f"running {args.proc} with data={is_data} (do_gen={not is_data})")
    print("fileset (keyed by DAS dataset name):")
    for k, v in fileset.items():
        print(f"  {k}")
        for f in v:
            print(f"    -> {f}")

    if args.proc == "dijet":
        proc = DijetProcessor(data=is_data)
    elif args.proc == "trijet":
        proc = TrijetProcessor(data=is_data)
    else:
        proc = triggerProcessor(year="2018", trigger="HLT_PFJet500", data=is_data)

    runner = processor.Runner(
        executor=processor.IterativeExecutor(workers=1, status=True),
        schema=NanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=(None if args.maxchunks <= 0 else args.maxchunks),
        skipbadfiles=False,
    )

    t0 = time.time()
    out = runner(fileset, proc, treename="Events")
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"result keys (first 20): {list(out)[:20]}")

    # True events-processed count from the input files (uproot), not the
    # processor's cutflow (which can double-count across jet-systematic loops).
    import uproot
    n_events = 0
    for _ds, files in fileset.items():
        for f in files:
            n_events += uproot.open(f)["Events"].num_entries
    # If maxchunks limits the run, the processor actually read fewer events.
    if args.maxchunks > 0:
        n_events = min(n_events, args.maxchunks * args.chunksize)
    print(f"events in input:  {n_events:,}")
    print(f"throughput:       {n_events/elapsed:,.0f} events/s")


if __name__ == "__main__":
    main()
