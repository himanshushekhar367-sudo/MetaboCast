#!/usr/bin/env python3
"""Build every main and supplementary figure of the manuscript from reproduce/results (file names: reproduce/manifest.py).

Usage:
    python reproduce/figures/make_figures.py [--results reproduce/results] [--out figures] [--only benchmark transfer ...]
Writes <name>.png (300 dpi) and <name>.pdf per figure, source CSVs for selected panels, and
figure_summary.json with the values quoted in the figure legends.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["benchmark", "transfer", "crosscancer", "overview", "robustness", "supplementary"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results"))
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*", choices=MODULES, default=MODULES)
    args = ap.parse_args()
    os.environ["METABOCAST_RESULTS"] = os.path.abspath(args.results)
    os.environ["METABOCAST_FIGURES"] = os.path.abspath(args.out)
    sys.path.insert(0, HERE)
    import importlib
    summary = {}
    for name in args.only:
        mod = importlib.import_module(f"fig_{name}")
        summary[name] = mod.run()
        print(f"built {name} figures", flush=True)
    path = os.path.join(args.out, "figure_summary.json")
    json.dump(summary, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print("wrote", path)


if __name__ == "__main__":
    main()
