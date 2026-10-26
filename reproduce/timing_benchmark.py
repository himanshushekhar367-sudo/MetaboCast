#!/usr/bin/env python3
"""Run time and memory of fitting and prediction on one CPU core (timing table).

Times, with BLAS and OpenMP limited to one thread:
  fit_one_section   MetaboCast() fitted on one DESIUM section (BC_515_Section_1; penalty chosen by
                    leave-one-spatial-block-out cross-validation);
  fit_seven         MetaboCast() fitted on all seven DESIUM sections (the shipped-model configuration);
  predict_query     prediction of every m/z channel for a transcriptome-only query of about 10^4 spots (the PVTT and
                    tumour sections of HCC-2, 9,356 spots) with the seven-section model.
Data loading is excluded. The machine (CPU model, logical cores, RAM), operating system and package versions are
recorded with the timings, and peak resident memory is read after each step.

Usage:
    python reproduce/timing_benchmark.py --data <DESIUM dir> --root <data root> --out reproduce/results/timing.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, gzip, importlib.metadata, json, platform, resource, sys, time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy, scipy.io, sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
from metabocast import MetaboCast                      # noqa: E402
from loso_benchmark import slim                       # noqa: E402

DESIUM = ["BC_515_Section_1", "BC_515_Section_2", "BC_525", "BC_823", "LC_091", "LC_170", "LC_276"]


def machine():
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        cpu = platform.processor()
    try:
        ram = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 1024 ** 2
    except Exception:
        ram = float("nan")
    return dict(cpu=cpu, logical_cores=os.cpu_count(), ram_gb=round(ram, 1), os=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                anndata=importlib.metadata.version("anndata"), threads=1)


def peak_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)    # ru_maxrss is in KiB on Linux


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    secs = {s: slim(ad.read_h5ad(a.data / f"{s}.h5ad")) for s in DESIUM}
    res = dict(machine=machine())
    with threadpool_limits(limits=1):
        s0 = secs["BC_515_Section_1"]
        t0 = time.perf_counter(); m1 = MetaboCast().fit([s0]); t1 = time.perf_counter()
        res["fit_one_section"] = dict(section="BC_515_Section_1", n_spots=int(s0.n_obs), n_ions=int(len(m1.mz_)),
                                      seconds=round(t1 - t0, 1), alpha=float(m1.alpha_), peak_gb=peak_gb())
        del m1
        t0 = time.perf_counter(); m7 = MetaboCast().fit([secs[s] for s in DESIUM]); t1 = time.perf_counter()
        res["fit_seven"] = dict(n_spots=int(sum(secs[s].n_obs for s in DESIUM)), seconds=round(t1 - t0, 1),
                                alpha=float(m7.alpha_), peak_gb=peak_gb())
        pid = "HCC-2"
        genes = [l.strip() for l in open(a.root / f"{pid}-expr_features.txt")]
        M = scipy.io.mmread(gzip.open(a.root / f"{pid}-expr_counts.mtx.gz")).tocsr()
        tot = np.asarray(M.sum(0)).ravel() + 1e-8
        X = M.T.tocsr().astype(np.float32)
        X = X.multiply(1.0 / tot[:, None]).multiply(1e4).tocsr(); X.data = np.log1p(X.data)
        meta = pd.read_csv(a.root / f"{pid}-expr_meta.csv", index_col=0).reindex([l.strip() for l in open(a.root / f"{pid}-expr_barcodes.txt")])
        region = meta["sample.ident"].astype(str).str.replace(pid, "", regex=False).str.lstrip("-").values
        keep = np.isin(region, ["P", "T"])
        q = ad.AnnData(X=X[keep]); q.var_names = genes; q.var_names_make_unique(); q.layers["log1p"] = q.X
        q.obsm["spatial"] = meta[["imagerow", "imagecol"]].values.astype(float)[keep]; q.obs["section"] = region[keep]
        t0 = time.perf_counter(); pred = m7.predict_metabolome(q, section_key="section"); t1 = time.perf_counter()
        res["predict_query"] = dict(query="HCC-2 PVTT and tumour sections", n_spots=int(q.n_obs), n_ions=int(pred.shape[1]),
                                    seconds=round(t1 - t0, 1), peak_gb=peak_gb())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
