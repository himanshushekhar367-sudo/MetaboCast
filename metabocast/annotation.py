"""Putative exact-mass annotation of negative-mode MSI m/z channels.

One curated reference of common tissue metabolites and representative lipid species (84 entries) serves the package
(MetaboCast.reliability_table and the R interface) and the manuscript analyses (reproduce/annotate_ions.py and the
HCC/GBM applications). Monoisotopic masses are computed from molecular formulae, and each channel is compared with the
negative-mode adducts [M-H]-, [M+Cl]- and [M+HCOO]- within a tolerance (default 0.01 Da).

- Every candidate within tolerance is kept and listed nearest first. Exact isobars (the same ion formula from two
  metabolites) are ordered [M-H]-, [M+Cl]-, [M+HCOO]-, then by name.
- Isomers that share a formula carry one group name: Hexose covers glucose, fructose, galactose and inositol;
  Citrate/isocitrate and Leucine/Ile likewise.
- A chemical class is reported only when all candidates of a channel share it ('ambiguous' otherwise).
- Mass errors are observed minus theoretical m/z, in mDa.

These are putative exact-mass candidates (no fragmentation, isotope-pattern or standard evidence), not structural
identifications.
"""
import re

import numpy as np
import pandas as pd

ATOMIC = {"C": 12.0, "H": 1.0078250319, "O": 15.9949146221, "N": 14.0030740052, "P": 30.97376151, "S": 31.97207069,
          "Na": 22.98976928, "K": 38.96370649, "Cl": 34.96885268}
ELECTRON, PROTON = 0.00054858, 1.00727646688
TOL = 0.01

# (group name, molecular formula, chemical class)
REFERENCE = [
    ("Lactate", "C3H6O3", "carboxylic acid"), ("Pyruvate", "C3H4O3", "carboxylic acid"),
    ("Succinate", "C4H6O4", "TCA acid"), ("Fumarate", "C4H4O4", "TCA acid"), ("Malate", "C4H6O5", "TCA acid"),
    ("Citrate/isocitrate", "C6H8O7", "TCA acid"), ("aKG", "C5H6O5", "TCA acid"),
    ("Alanine", "C3H7NO2", "amino acid"), ("Serine", "C3H7NO3", "amino acid"), ("Proline", "C5H9NO2", "amino acid"),
    ("Valine", "C5H11NO2", "amino acid"), ("Threonine", "C4H9NO3", "amino acid"),
    ("Leucine/Ile", "C6H13NO2", "amino acid"), ("Asparagine", "C4H8N2O3", "amino acid"),
    ("Aspartate", "C4H7NO4", "amino acid"), ("Glutamine", "C5H10N2O3", "amino acid"),
    ("Glutamate", "C5H9NO4", "amino acid"), ("Lysine", "C6H14N2O2", "amino acid"),
    ("Arginine", "C6H14N4O2", "amino acid"), ("Histidine", "C6H9N3O2", "amino acid"),
    ("Phenylalanine", "C9H11NO2", "amino acid"), ("Tyrosine", "C9H11NO3", "amino acid"),
    ("Tryptophan", "C11H12N2O2", "amino acid"), ("Methionine", "C5H11NO2S", "amino acid"),
    ("Cysteine", "C3H7NO2S", "amino acid"), ("Taurine", "C2H7NO3S", "amino acid"),
    ("Glutathione", "C10H17N3O6S", "peptide/redox"), ("GSSG", "C20H32N6O12S2", "peptide/redox"),
    ("Hexose", "C6H12O6", "sugar"), ("Hexose-P", "C6H13O9P", "sugar phosphate"),
    ("Hexose-BP", "C6H14O12P2", "sugar phosphate"), ("Ribose-5-P", "C5H11O8P", "sugar phosphate"),
    ("Disaccharide", "C12H22O11", "sugar"), ("AMP", "C10H14N5O7P", "nucleotide"),
    ("ADP", "C10H15N5O10P2", "nucleotide"), ("ATP", "C10H16N5O13P3", "nucleotide"),
    ("GMP", "C10H14N5O8P", "nucleotide"), ("GDP", "C10H15N5O11P2", "nucleotide"),
    ("GTP", "C10H16N5O14P3", "nucleotide"), ("UMP", "C9H13N2O9P", "nucleotide"),
    ("UDP", "C9H14N2O12P2", "nucleotide"), ("NAD+", "C21H27N7O14P2", "cofactor"),
    ("NADH", "C21H29N7O14P2", "cofactor"), ("NADP", "C21H28N7O17P3", "cofactor"),
    ("FAD", "C27H33N9O15P2", "cofactor"), ("Inosine", "C10H12N4O5", "nucleoside"),
    ("Hypoxanthine", "C5H4N4O", "nucleobase"), ("Xanthine", "C5H4N4O2", "nucleobase"),
    ("Urate", "C5H4N4O3", "nucleobase"), ("Creatine", "C4H9N3O2", "other"), ("Creatinine", "C4H7N3O", "other"),
    ("Carnitine", "C7H15NO3", "acylcarnitine"), ("Acetylcarnitine", "C9H17NO4", "acylcarnitine"),
    ("Choline", "C5H13NO", "lipid headgroup"), ("Glycerophosphocholine", "C8H20NO6P", "lipid headgroup"),
    ("Ascorbate", "C6H8O6", "vitamin"), ("FA16:0", "C16H32O2", "fatty acid"), ("FA18:0", "C18H36O2", "fatty acid"),
    ("FA18:1", "C18H34O2", "fatty acid"), ("FA18:2", "C18H32O2", "fatty acid"),
    ("FA20:4(AA)", "C20H32O2", "fatty acid"), ("FA22:6(DHA)", "C22H32O2", "fatty acid"),
    ("FA20:5(EPA)", "C20H30O2", "fatty acid"), ("FA24:1", "C24H46O2", "fatty acid"),
    ("LPC16:0", "C24H50NO7P", "lysophospholipid"), ("LPC18:1", "C26H52NO7P", "lysophospholipid"),
    ("PC34:1", "C42H82NO8P", "phospholipid"), ("PC36:2", "C44H84NO8P", "phospholipid"),
    ("PC38:4", "C46H84NO8P", "phospholipid"), ("PE36:2", "C41H78NO8P", "phospholipid"),
    ("PE38:4", "C43H78NO8P", "phospholipid"), ("PI38:4", "C47H83O13P", "phospholipid"),
    ("PS36:1", "C42H80NO10P", "phospholipid"), ("PG34:1", "C40H77O10P", "phospholipid"),
    ("SM d18:1/16:0", "C39H79N2O6P", "sphingolipid"), ("Cer d18:1/16:0", "C34H67NO3", "sphingolipid"),
    ("Cholesterol", "C27H46O", "sterol"), ("TAG52:2", "C55H102O6", "glycerolipid"),
    ("Adenosine", "C10H13N5O4", "nucleoside"), ("Kynurenine", "C10H12N2O3", "amino acid"),
    ("N-acetylaspartate", "C6H9NO5", "amino acid"), ("Glycine", "C2H5NO2", "amino acid"),
    ("Sphingosine", "C18H37NO2", "sphingolipid"), ("Spermidine", "C7H19N3", "polyamine"),
]


def formula_mass(formula):
    """Monoisotopic neutral mass of a molecular formula such as 'C6H12O6'."""
    return sum(ATOMIC[e] * (int(n) if n else 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if e)


ADDUCTS = {"[M-H]-": -PROTON, "[M+Cl]-": ATOMIC["Cl"] + ELECTRON, "[M+HCOO]-": formula_mass("CH2O2") - PROTON}
_ADDUCT_ORDER = {a: i for i, a in enumerate(ADDUCTS)}


def reference_table():
    """One row per (metabolite, adduct): name, formula, class, adduct and theoretical m/z, sorted by m/z."""
    rows = [(n, f, c, a, formula_mass(f) + d) for n, f, c in REFERENCE for a, d in ADDUCTS.items()]
    return pd.DataFrame(rows, columns=["name", "formula", "class", "adduct", "mz"]).sort_values("mz").reset_index(drop=True)


def neutral_mass(name):
    """Monoisotopic neutral mass of a reference entry, by its group name."""
    for n, f, _ in REFERENCE:
        if n == name:
            return formula_mass(f)
    raise KeyError(f"{name!r} is not in the annotation reference")


def _hits(m, rmz, names, adducts, tol):
    hit = np.where(np.abs(rmz - m) <= tol)[0]
    return sorted(hit, key=lambda j: (round(abs(m - rmz[j]), 6), _ADDUCT_ORDER[adducts[j]], names[j]))


def annotate(mz, tol=TOL):
    """One row per channel (m/z rounded to 4 decimals): n_candidates, annotation (group names, nearest first),
    adduct (of the nearest candidate), candidates ('name adduct (error mDa)', nearest first), class (shared by all
    candidates, else 'ambiguous'; 'unannotated' without a candidate), candidate_classes, error_mDa (observed minus
    theoretical, nearest candidate) and ambiguous (more than one candidate)."""
    ref = reference_table()
    rmz, names, adducts, classes = ref.mz.to_numpy(), ref.name.to_numpy(), ref.adduct.to_numpy(), ref["class"].to_numpy()
    rows = []
    for m in np.round(np.atleast_1d(np.asarray(mz, float)), 4):
        hit = _hits(m, rmz, names, adducts, tol)
        if not len(hit):
            rows.append((m, 0, "", "", "", "unannotated", "", np.nan, False))
            continue
        uniq = list(dict.fromkeys(names[j] for j in hit))
        cls = list(dict.fromkeys(classes[j] for j in hit))
        rows.append((m, len(hit), "; ".join(uniq), adducts[hit[0]],
                     "; ".join(f"{names[j]} {adducts[j]} ({(m - rmz[j]) * 1000:+.1f} mDa)" for j in hit),
                     cls[0] if len(cls) == 1 else "ambiguous", "; ".join(cls),
                     round((m - rmz[hit[0]]) * 1000, 2), len(hit) > 1))
    return pd.DataFrame(rows, columns=["mz", "n_candidates", "annotation", "adduct", "candidates", "class",
                                       "candidate_classes", "error_mDa", "ambiguous"])


def candidate_labels(mz, tol=TOL):
    """'Name [adduct]; Name [adduct]' per channel, nearest first ('' without a candidate)."""
    ref = reference_table()
    rmz, names, adducts = ref.mz.to_numpy(), ref.name.to_numpy(), ref.adduct.to_numpy()
    return ["; ".join(f"{names[j]} {adducts[j]}" for j in _hits(m, rmz, names, adducts, tol))
            for m in np.round(np.atleast_1d(np.asarray(mz, float)), 4)]
