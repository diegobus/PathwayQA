import argparse
import re
from pathlib import Path
from statistics import median
import pandas as pd

UNIPROT_RE = re.compile(r"\(UniProt:\s*([A-Z0-9]{6,10})\)")
CHEBI_RE = re.compile(r"\(ChEBI:\s*([0-9]+)\)")

def extract_ids(text: str):
    if not isinstance(text, str):
        return set(), set()
    uniprot = set(UNIPROT_RE.findall(text))
    chebi = set(CHEBI_RE.findall(text))
    return uniprot, chebi

def compute_reaction_scores(df: pd.DataFrame):
    # Extract IDs from prompt+answer
    uni_sets = []
    chebi_sets = []
    for p, a in zip(df["prompt"], df["answer"]):
        u1, c1 = extract_ids(p)
        u2, c2 = extract_ids(a)
        uni_sets.append(u1 | u2)
        chebi_sets.append(c1 | c2)

    df = df.copy()
    df["uniprot_ids"] = uni_sets
    df["chebi_ids"] = chebi_sets
    df["n_uniprot"] = df["uniprot_ids"].map(len)
    df["n_chebi"] = df["chebi_ids"].map(len)

    # Global UniProt frequencies
    freq = {}
    for s in df["uniprot_ids"]:
        for pid in s:
            freq[pid] = freq.get(pid, 0) + 1

    # Reaction-level scores
    def min_freq(s):
        if not s:
            return 0
        return min(freq[p] for p in s)

    def median_freq(s):
        if not s:
            return 0
        return int(median([freq[p] for p in s]))

    df["uniprot_min_freq"] = df["uniprot_ids"].map(min_freq)
    df["uniprot_median_freq"] = df["uniprot_ids"].map(median_freq)

    return df, freq

def suggest_thresholds(df: pd.DataFrame, score_col: str, min_size: int = 200):
    # Only consider reactions that have at least 1 protein, otherwise they're uninformative for protein-coverage splits
    sub = df[df["n_uniprot"] > 0].copy()
    values = sorted(sub[score_col].unique())

    candidates = []
    for t in values:
        under = sub[sub[score_col] <= t]
        well = sub[sub[score_col] > t]
        if len(under) >= min_size and len(well) >= min_size:
            imbalance = abs(len(under) - len(well)) / len(sub)
            candidates.append((t, len(under), len(well), imbalance))

    # Sort by (lowest imbalance, then smaller threshold)
    candidates.sort(key=lambda x: (x[3], x[0]))
    return candidates[:20]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="PathwayQA CSV with reaction_id,prompt,answer")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--score", default="uniprot_min_freq", choices=["uniprot_min_freq", "uniprot_median_freq"])
    ap.add_argument("--threshold", type=int, default=None, help="If set, write splits using this threshold")
    ap.add_argument("--min_size", type=int, default=200, help="Minimum size per split when suggesting thresholds")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if not {"reaction_id", "prompt", "answer"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: reaction_id,prompt,answer")

    scored, freq = compute_reaction_scores(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save scored table + protein frequency table
    scored_out = outdir / "pathwayqa_scored.csv"
    scored.to_csv(scored_out, index=False)

    freq_df = pd.DataFrame({"uniprot_id": list(freq.keys()), "count": list(freq.values())}).sort_values("count", ascending=False)
    freq_df.to_csv(outdir / "uniprot_frequencies.csv", index=False)

    # Print threshold suggestions
    print(f"Using score column: {args.score}")
    cands = suggest_thresholds(scored, args.score, min_size=args.min_size)
    if not cands:
        print("No thresholds found meeting min_size constraints. Try lowering --min_size or use a different score.")
    else:
        print("Top candidate thresholds (t, under_n, well_n, imbalance):")
        for t, u, w, imb in cands:
            print(f"  t={t:4d}  under={u:6d}  well={w:6d}  imbalance={imb:.3f}")

    # Write splits if requested
    if args.threshold is not None:
        sub = scored[scored["n_uniprot"] > 0].copy()
        under = sub[sub[args.score] <= args.threshold]
        well = sub[sub[args.score] > args.threshold]

        under.to_csv(outdir / f"under_observed_t{args.threshold}.csv", index=False)
        well.to_csv(outdir / f"well_observed_t{args.threshold}.csv", index=False)

        print(f"\nWrote splits to {outdir}:")
        print(f"  under_observed_t{args.threshold}.csv  (n={len(under)})")
        print(f"  well_observed_t{args.threshold}.csv   (n={len(well)})")
        # Also note how many rows had no proteins
        nop = scored[scored["n_uniprot"] == 0]
        if len(nop) > 0:
            print(f"  Note: {len(nop)} rows had 0 UniProt IDs and were excluded from splits.")

if __name__ == "__main__":
    main()