"""Mark which KRAS positions touch the RAF1 binding interface or the nucleotide pocket (for the allostery analysis).

  python src/annotate_structure.py 6VJJ.pdb data/positions.csv      # PDB 6VJJ = KRAS bound to the RAF1 RBD (as in Weng et al. Fig. 1d)

Download the file on your Mac from https://files.rcsb.org/download/6VJJ.pdb and copy it into data/.
Working definition (check it against the Methods of Weng et al.): a position is 'interface' / 'pocket' if any of its
heavy atoms is within 5 A of the partner chain / the bound nucleotide. Chain letters are detected automatically."""
import sys
import numpy as np, pandas as pd

pdb, out = sys.argv[1], sys.argv[2]
CUT = 5.0
atoms = []                      # (record, chain, resname, resnum, element, xyz)
for line in open(pdb):
    if line.startswith(("ATOM", "HETATM")):
        el = line[76:78].strip() or line[12:16].strip()[0]
        if el == "H":
            continue
        atoms.append((line[:6].strip(), line[21], line[17:20].strip(), int(line[22:26]), el,
                      np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
NUC = {"GTP", "GNP", "GDP", "GCP", "G3D", "GSP"}
nuc = [x for x in atoms if x[0] == "HETATM" and x[2] in NUC]
if not nuc:
    sys.exit("no nucleotide (GTP/GNP/GDP/GCP) found in the file")
kras_chain = nuc[0][1]
chains = {}
for x in atoms:
    if x[0] == "ATOM":
        chains.setdefault(x[1], set()).add(x[3])
partners = [c for c in chains if c != kras_chain]
print("KRAS chain:", kras_chain, "| partner chain(s):", partners)
kras = [x for x in atoms if x[0] == "ATOM" and x[1] == kras_chain]
partner = np.array([x[5] for x in atoms if x[0] == "ATOM" and x[1] in partners])
nuc_xyz = np.array([x[5] for x in nuc if x[1] == kras_chain])
rows = {}
for x in kras:
    r = rows.setdefault(x[3], dict(pos=x[3], is_interface=False, is_pocket=False))
    if len(partner) and np.min(np.linalg.norm(partner - x[5], axis=1)) < CUT:
        r["is_interface"] = True
    if np.min(np.linalg.norm(nuc_xyz - x[5], axis=1)) < CUT:
        r["is_pocket"] = True
df = pd.DataFrame(rows.values()).sort_values("pos")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} residues, {df.is_interface.sum()} interface, {df.is_pocket.sum()} pocket")
