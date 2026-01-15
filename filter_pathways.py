import argparse
import json
import re
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

UNIPROT_RE = re.compile(r"\(UniProt:\s*([A-Z0-9]{6,10})\)")

REQUIRED_COLS = ["reaction_id", "prompt", "answer"]


def extract_uniprot_ids(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    return set(UNIPROT_RE.findall(text))


def compute_uniprot_sets(df: pd.DataFrame) -> pd.Series:
    """Return a Series of sets: UniProt IDs per reaction row (prompt+answer)."""
    sets = []
    for p, a in zip(df["prompt"], df["answer"]):
        sets.append(extract_uniprot_ids(p) | extract_uniprot_ids(a))
    return pd.Series(sets, index=df.index)


def compute_global_freq(uniprot_sets: pd.Series) -> Dict[str, int]:
    """freq[p] = number of reactions (rows) that contain UniProt p."""
    freq: Dict[str, int] = {}
    for s in uniprot_sets:
        for pid in s:
            freq[pid] = freq.get(pid, 0) + 1
    return freq


def purity_le_k(uniprot_set: Set[str], freq: Dict[str, int], k: int) -> float:
    """
    Fraction of proteins in the reaction with global frequency <= k.
    Returns NaN-like None if reaction has zero proteins.
    """
    if not uniprot_set:
        return None
    rare = sum(1 for p in uniprot_set if freq.get(p, 0) <= k)
    return rare / len(uniprot_set)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split PathwayQA reactions into under-/well-observed using protein-frequency purity."
    )
    ap.add_argument(
        "--csv", required=True, help="Input CSV with columns: reaction_id,prompt,answer"
    )
    ap.add_argument(
        "--outdir",
        default="data/filtered",
        help="Output directory (default: data/filtered)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=3,
        help="Protein rarity cutoff: a protein is 'rare' if it appears in <= k reactions (default: 3)",
    )
    ap.add_argument(
        "--under_purity",
        type=float,
        default=0.7,
        help="Under-observed if purity >= this (default: 0.7)",
    )
    ap.add_argument(
        "--well_purity",
        type=float,
        default=0.1,
        help="Well-observed if purity <= this (default: 0.1)",
    )
    ap.add_argument(
        "--drop_no_protein",
        action="store_true",
        help="If set, drop rows with 0 UniProt IDs from both splits (default: keep them in neither split, reported separately)",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Optional tag added to output filenames (useful for tracking runs)",
    )
    args = ap.parse_args()

    in_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    # UniProt IDs per reaction + global frequencies
    uniprot_sets = compute_uniprot_sets(df)
    freq = compute_global_freq(uniprot_sets)

    n_total = len(df)
    n_with_protein = int((uniprot_sets.map(len) > 0).sum())
    n_no_protein = n_total - n_with_protein
    n_unique_proteins = len(freq)

    # Purity per row
    purity = uniprot_sets.apply(lambda s: purity_le_k(s, freq, args.k))

    # Build masks
    has_protein = uniprot_sets.map(len) > 0
    under_mask = has_protein & purity.notna() & (purity >= args.under_purity)
    well_mask = has_protein & purity.notna() & (purity <= args.well_purity)

    # Optionally drop no-protein rows entirely (they’re already excluded from both masks)
    if args.drop_no_protein:
        df_eval = df[has_protein].copy()
        under_mask = under_mask.loc[df_eval.index]
        well_mask = well_mask.loc[df_eval.index]
    else:
        df_eval = df.copy()

    # IMPORTANT: write ONLY original columns to keep inference format identical
    under_df = df_eval.loc[under_mask, REQUIRED_COLS].copy()
    well_df = df_eval.loc[well_mask, REQUIRED_COLS].copy()

    # Filename tagging
    tag = args.tag or f"k{args.k}_u{args.under_purity:g}_w{args.well_purity:g}"
    under_path = outdir / f"reaction_prompts_answers_under_observed_{tag}.csv"
    well_path = outdir / f"reaction_prompts_answers_well_observed_{tag}.csv"

    under_df.to_csv(under_path, index=False)
    well_df.to_csv(well_path, index=False)

    # Save a small reproducibility report
    report = {
        "input_csv": str(in_path),
        "output_dir": str(outdir),
        "k": args.k,
        "under_purity": args.under_purity,
        "well_purity": args.well_purity,
        "drop_no_protein": bool(args.drop_no_protein),
        "n_total_rows": n_total,
        "n_rows_with_protein": n_with_protein,
        "n_rows_no_protein": n_no_protein,
        "n_unique_proteins": n_unique_proteins,
        "n_under_observed": int(len(under_df)),
        "n_well_observed": int(len(well_df)),
        "n_overlap_under_and_well": int((under_mask & well_mask).sum()),
        "n_unassigned_with_protein": int(
            (has_protein & ~(under_mask | well_mask)).sum()
        ),
    }
    report_path = outdir / f"split_summary_{tag}.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Clean, minimal terminal output
    print(f"Input: {in_path}")
    print(
        f"Total rows: {n_total:,} | With UniProt: {n_with_protein:,} | No UniProt: {n_no_protein:,} | Unique proteins: {n_unique_proteins:,}"
    )
    print(
        f"Split params: k={args.k}, under_purity>={args.under_purity:g}, well_purity<={args.well_purity:g}"
    )
    print(f"Under-observed: {len(under_df):,}  -> {under_path}")
    print(f"Well-observed:  {len(well_df):,}  -> {well_path}")
    print(
        f"Unassigned (with protein, in-between): {report['n_unassigned_with_protein']:,}"
    )
    print(f"Summary JSON: {report_path}")


if __name__ == "__main__":
    main()
