"""Production runner for the GluonJetMass processors under coffea 2025+.

Dispatches a processor (dijet/trijet/trigger) over a year-filtered fileset
using a chosen executor (iterative / futures / dask-lpc / dask-casa) and
pickles the result to coffeaOutput/.

Examples
--------
  # Local, single-process — quick smoke (a couple of chunks)
  python run.py --proc dijet --mc --year 2018 --mctype MG \\
      --executor iterative --maxchunks 2

  # Local, multi-core
  python run.py --proc dijet --mc --year 2018 --executor futures --workers 8

  # LPC dask via lpcjobqueue (ships correctionFiles/ + python/ to workers)
  python run.py --proc dijet --mc --year 2018 --executor dask-lpc \\
      --max-workers 100

  # coffea-casa dask
  python run.py --proc dijet --data --year 2018 --executor dask-casa

  # Run a single named dataset
  python run.py --proc trijet --mc --year 2017 --executor dask-lpc \\
      --dataset Pt_300to470
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from python.dijetProcessor import DijetProcessor
from python.trijetProcessor import TrijetProcessor
from python.triggerProcessor import triggerProcessor


# ---------------------------------------------------------------------------
# fileset selection / loading
# ---------------------------------------------------------------------------

# Substring(s) the DAS dataset name must contain to count for a given year.
# Mirrors plugins.handleData but kept here so run.py is self-contained.
ERAS_MC = {
    "2016APV": "UL16NanoAODAPV",
    "2016":    "UL16NanoAODv9",
    "2017":    "UL17NanoAODv9",
    "2018":    "UL18NanoAODv9",
}
ERAS_DATA = {
    "2016APV": "HIPM_UL2016",
    "2016":    "-UL2016",
    "2017":    "UL2017",
    "2018":    "UL2018",
}

# MC source -> (default fileset JSON, datastr tag for output filename)
MCTYPE_TO_FILESET = {
    "pythia": ("fileset_QCD_wRedirs.json",        "QCD_pythia"),
    "MG":     ("fileset_MG_pythia8_wRedirs.json", "QCD_MG"),
    "herwig": ("fileset_HERWIG_wRedirs.json",     "QCD_herwig"),
}

FULL_JET_SYSTEMATICS = [
    "nominal", "JERUp", "JERDown", "HEM",
    "JES_AbsoluteMPFBiasUp", "JES_AbsoluteMPFBiasDown",
    "JES_AbsoluteScaleUp", "JES_AbsoluteScaleDown",
    "JES_AbsoluteStatUp", "JES_AbsoluteStatDown",
    "JES_FlavorQCDUp", "JES_FlavorQCDDown",
    "JES_FragmentationUp", "JES_FragmentationDown",
    "JES_PileUpDataMCUp", "JES_PileUpDataMCDown",
    "JES_PileUpPtBBUp", "JES_PileUpPtBBDown",
    "JES_PileUpPtEC1Up", "JES_PileUpPtEC1Down",
    "JES_PileUpPtEC2Up", "JES_PileUpPtEC2Down",
    "JES_PileUpPtHFUp", "JES_PileUpPtHFDown",
    "JES_PileUpPtRefUp", "JES_PileUpPtRefDown",
    "JES_RelativeFSRUp", "JES_RelativeFSRDown",
    "JES_RelativeJEREC1Up", "JES_RelativeJEREC1Down",
    "JES_RelativeJEREC2Up", "JES_RelativeJEREC2Down",
    "JES_RelativeJERHFUp", "JES_RelativeJERHFDown",
    "JES_RelativePtBBUp", "JES_RelativePtBBDown",
    "JES_RelativePtEC1Up", "JES_RelativePtEC1Down",
    "JES_RelativePtEC2Up", "JES_RelativePtEC2Down",
    "JES_RelativePtHFUp", "JES_RelativePtHFDown",
    "JES_RelativeBalUp", "JES_RelativeBalDown",
    "JES_RelativeSampleUp", "JES_RelativeSampleDown",
    "JES_RelativeStatECUp", "JES_RelativeStatECDown",
    "JES_RelativeStatFSRUp", "JES_RelativeStatFSRDown",
    "JES_RelativeStatHFUp", "JES_RelativeStatHFDown",
    "JES_SinglePionECALUp", "JES_SinglePionECALDown",
    "JES_SinglePionHCALUp", "JES_SinglePionHCALDown",
    "JES_TimePtEtaUp", "JES_TimePtEtaDown",
    "JMRUp", "JMRDown", "JMSUp", "JMSDown",
]


def pick_fileset(args) -> tuple[str, str]:
    if args.fileset:
        return args.fileset, (args.datastr or "custom")
    if args.data:
        return "fileset_JetHT_wRedirs.json", "JetHT"
    return MCTYPE_TO_FILESET[args.mctype]


def build_fileset(json_path: str, year: str | None, is_data: bool,
                  dataset_substr: str | None, testing: bool,
                  dataset_range: list[int] | None, redirector: str) -> dict[str, list[str]]:
    """Load fileset JSON; filter by year + optional --dataset substring."""
    with open(json_path) as f:
        top = json.load(f)
    qualifier = (ERAS_DATA if is_data else ERAS_MC).get(year)
    out: dict[str, list[str]] = {}
    for _grp, datasets in top.items():
        if not isinstance(datasets, dict):
            continue
        for das, files in datasets.items():
            if qualifier and qualifier not in das:
                continue
            if dataset_substr and dataset_substr.lower() not in das.lower():
                continue
            urls = []
            for f in files:
                if f.startswith("root://"):
                    urls.append(f)
                elif f.startswith("/store/"):
                    urls.append(redirector.rstrip("/") + "/" + f)
                else:
                    urls.append(f)
            if testing:
                urls = urls[:1]
            if urls:
                out[das] = urls
    if dataset_range is not None:
        keys = list(out.keys())[dataset_range[0]:dataset_range[1]]
        out = {k: out[k] for k in keys}
    return out


# ---------------------------------------------------------------------------
# processor instantiation
# ---------------------------------------------------------------------------

def build_processor(args):
    do_minimal = not args.all_plots
    if args.proc == "dijet":
        return DijetProcessor(
            data=args.data, jet_systematics=args.jet_syst,
            jk=args.jk, jk_range=args.jk_range, do_minimal=do_minimal,
        )
    if args.proc == "trijet":
        return TrijetProcessor(
            data=args.data, jet_systematics=args.jet_syst,
            jk=args.jk, jk_range=args.jk_range, do_minimal=do_minimal,
        )
    if args.proc == "trigger":
        year = args.year if args.year != "all" else "2018"
        return triggerProcessor(year=year, trigger=(args.trigger or "HLT_PFJet500"), data=args.data)
    raise SystemExit(f"unknown --proc {args.proc!r}")


# ---------------------------------------------------------------------------
# executor / cluster lifecycle
# ---------------------------------------------------------------------------

def make_executor_and_resources(args):
    """Return (executor, teardown_callable). teardown is called in a finally
    block to close any dask cluster/client we created."""

    def register_local_directory_sys_path(client):
        from distributed.diagnostics.plugin import WorkerPlugin

        class _AddLocalDirToSysPath(WorkerPlugin):
            def setup(self, worker):
                import sys
                local_directory = str(worker.local_directory)
                if local_directory not in sys.path:
                    sys.path.insert(0, local_directory)

        client.register_plugin(_AddLocalDirToSysPath(), name="add-local-directory-to-syspath")

    if args.executor == "iterative":
        return processor.IterativeExecutor(workers=1, status=True), (lambda: None)

    if args.executor == "futures":
        return processor.FuturesExecutor(workers=args.workers, status=True, compression=None), (lambda: None)

    if args.executor == "dask-local":
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(
            n_workers=args.workers,
            threads_per_worker=1,
            memory_limit=args.dask_memory,
            dashboard_address=":8787",
        )
        client = Client(cluster)
        print("dask dashboard:", client.dashboard_link)
        ex = processor.DaskExecutor(
            client=client, status=True, retries=3, treereduction=4,
        )
        def teardown():
            client.close(); cluster.close()
        return ex, teardown

    if args.executor == "dask-lpc":
        from dask.distributed import Client
        from lpcjobqueue import LPCCondorCluster

        cluster = LPCCondorCluster(
            memory=args.dask_memory,
            transfer_input_files=["correctionFiles", "python"],
            ship_env=False,
        )
        cluster.adapt(minimum=args.min_workers, maximum=args.max_workers)
        client = Client(cluster)
        print("dask dashboard:", client.dashboard_link)
        register_local_directory_sys_path(client)
        ex = processor.DaskExecutor(
            client=client, retries=10, treereduction=4, status=args.verbose,
        )
        def teardown():
            client.close(); cluster.close()
        return ex, teardown

    if args.executor == "dask-casa":
        from dask.distributed import Client
        from distributed.diagnostics.plugin import UploadDirectory
        from coffea_casa import CoffeaCasaCluster

        cluster = CoffeaCasaCluster(memory=args.dask_memory)
        cluster.adapt(minimum=args.min_workers, maximum=args.max_workers)
        client = Client(cluster)
        print("dask dashboard:", client.dashboard_link)
        client.register_plugin(
            UploadDirectory(str(HERE / "python"), restart_workers=True, update_path=True),
            name="upload-python",
        )
        client.register_plugin(
            UploadDirectory(str(HERE / "correctionFiles"), restart_workers=True, update_path=True),
            name="upload-correction-files",
        )
        register_local_directory_sys_path(client)
        ex = processor.DaskExecutor(
            client=client, status=args.verbose, retries=10, treereduction=4,
        )
        def teardown():
            client.close(); cluster.close()
        return ex, teardown

    raise SystemExit(f"unknown --executor {args.executor!r}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def list_of_ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--proc", choices=["dijet", "trijet", "trigger"], required=True)
    ap.add_argument("--year", choices=["2016", "2016APV", "2017", "2018", "all"], default="all")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--data", action="store_true")
    g.add_argument("--mc", action="store_true", help="default if neither --data nor --mc is given")

    ap.add_argument("--mctype", choices=["pythia", "MG", "herwig"], default="MG")
    ap.add_argument("--executor",
                    choices=["iterative", "futures", "dask-local", "dask-lpc", "dask-casa"],
                    default="iterative")
    ap.add_argument("--workers", type=int, default=4, help="FuturesExecutor worker count")
    ap.add_argument("--min-workers", type=int, default=1, help="dask cluster adapt minimum")
    ap.add_argument("--max-workers", type=int, default=100, help="dask cluster adapt maximum")
    ap.add_argument("--dask-memory", default="4 GiB",
                    help="per-worker memory request for dask-lpc / dask-casa "
                         "(default '4 GiB'; 10 GiB requests rarely get granted on LPC)")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--maxchunks", type=int, default=0, help="<=0 means no limit")
    ap.add_argument("--jet-syst", nargs="+", default=["nominal", "HEM"])
    ap.add_argument("--all-jet-syst", "--allUncertaintySources",
                    dest="all_jet_syst", action="store_true",
                    help="run the full jet-systematics list used by the legacy selection scripts")
    ap.add_argument("--jk", action="store_true")
    ap.add_argument("--jk-range", type=list_of_ints, default=None,
                    help="e.g. --jk-range 0,5")
    ap.add_argument("--dataset", default=None,
                    help="case-insensitive substring filter on DAS dataset name")
    ap.add_argument("--dataset-range", type=list_of_ints, default=None,
                    help="e.g. --dataset-range 0,5 to slice matched datasets")
    ap.add_argument("--all-plots", action="store_true",
                    help="do_minimal=False (more histograms)")
    ap.add_argument("--testing", action="store_true",
                    help="keep only the first file per matched dataset")
    ap.add_argument("--fileset", default=None,
                    help="override auto-picked fileset_*.json")
    ap.add_argument("--datastr", default=None,
                    help="override datastr tag in output filename (only used with --fileset)")
    ap.add_argument("--trigger", default=None, help="for --proc trigger")
    ap.add_argument("--redirector", default="root://cmsxrootd.fnal.gov/")
    ap.add_argument("--output", default=None,
                    help="pkl output path (default: coffeaOutput/<proc>/<proc>Hists_<datastr>_<year>.pkl)")
    ap.add_argument("--skipbadfiles", action="store_true")
    ap.add_argument("--savemetrics", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.executor == "dask-casa" and args.redirector == ap.get_default("redirector"):
        args.redirector = "root://xcache/"

    # Default to MC if neither flag given, and force nominal-only for data/herwig
    if not args.data and not args.mc:
        args.mc = True
    if args.all_jet_syst:
        args.jet_syst = FULL_JET_SYSTEMATICS
    if args.data or args.mctype == "herwig":
        args.jet_syst = ["nominal"]

    fileset_json, datastr = pick_fileset(args)
    year_arg = None if args.year == "all" else args.year
    fileset = build_fileset(
        fileset_json, year_arg, args.data, args.dataset, args.testing,
        args.dataset_range, args.redirector,
    )
    if not fileset:
        raise SystemExit(
            f"empty fileset (json={fileset_json}, year={args.year}, "
            f"data={args.data}, dataset={args.dataset!r})"
        )

    print(f"running {args.proc} "
          f"({'data' if args.data else 'MC ' + args.mctype}) "
          f"on {len(fileset)} dataset(s); year={args.year} executor={args.executor}")
    if args.verbose:
        for k, v in fileset.items():
            print(f"  {k}: {len(v)} files")

    proc = build_processor(args)
    executor, teardown = make_executor_and_resources(args)
    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=args.chunksize,
        maxchunks=(None if args.maxchunks <= 0 else args.maxchunks),
        skipbadfiles=args.skipbadfiles,
        savemetrics=args.savemetrics,
        align_clusters=(args.executor == "dask-casa"),
    )

    t0 = time.time()
    try:
        out = runner(fileset, proc, treename="Events")
    finally:
        teardown()
    elapsed = time.time() - t0

    if args.savemetrics:
        out, metrics = out
        print("metrics:", metrics)
    print(f"done in {elapsed:.1f}s; result keys (first 25): {list(out)[:25]}")

    output_suffix = "_testing" if args.testing else ""
    out_path = Path(args.output) if args.output else (
        HERE / f"coffeaOutput/{args.proc}/{args.proc}Hists_{datastr}_{args.year}{output_suffix}.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
