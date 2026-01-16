import argparse
import os
import time
import pandas as pd
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

SYSTEM_PROMPT = (
    "You give very concise answers. "
    "You only generate lists of products and nothing else. "
    "You do not repeat enzymes or reactants."
)

MAX_BATCH_SIZE = 9000
MAX_POLL_ITERATIONS = 360

def extract_text(message) -> str:
    # message.content is a list of content blocks; collect all text blocks
    out = []
    for block in message.content:
        if block.type == "text":
            out.append(block.text)
    return "".join(out).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--model", default="claude-3-5-haiku-20241022")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--poll_sec", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="Debug: only run first N rows")
    ap.add_argument("--batch_size", type=int, default=MAX_BATCH_SIZE, help=f"Max requests per batch (default: {MAX_BATCH_SIZE})")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment")

    df = pd.read_csv(args.input_csv)
    required = {"reaction_id", "prompt", "answer"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    
    original_len = len(df)
    df = df.dropna(subset=["reaction_id", "prompt"]).copy()
    df["reaction_id"] = df["reaction_id"].astype(str)
    df = df.drop_duplicates(subset=["reaction_id"], keep="first")
    if len(df) < original_len:
        print(f"Warning: Dropped {original_len - len(df)} rows (nulls/duplicates)")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    
    if len(df) > args.batch_size:
        print(f"Warning: {len(df)} rows exceeds batch limit {args.batch_size}, will process in {(len(df) + args.batch_size - 1) // args.batch_size} batches")

    client = anthropic.Anthropic(api_key=api_key)

    # Process in batches
    all_results = {}
    num_batches = (len(df) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(df))
        df_chunk = df.iloc[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1}/{num_batches}: processing {len(df_chunk)} requests")
        
        # Build batch requests
        requests = []
        for _, row in df_chunk.iterrows():
            rid = str(row["reaction_id"])
            prompt = str(row["prompt"])
            requests.append(
                Request(
                    custom_id=rid,
                    params=MessageCreateParamsNonStreaming(
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                )
            )

        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"Submitted batch: {batch_id}")

        # Poll until done with timeout
        poll_count = 0
        while poll_count < MAX_POLL_ITERATIONS:
            st = client.messages.batches.retrieve(batch_id)
            if hasattr(st, 'request_counts'):
                print(f"  Status: {st.processing_status} | Succeeded: {st.request_counts.succeeded}/{len(requests)}")
            if st.processing_status in ("ended", "completed"):
                break
            time.sleep(args.poll_sec)
            poll_count += 1
        
        if poll_count >= MAX_POLL_ITERATIONS:
            print(f"ERROR: Batch {batch_id} timed out")
            break

        # Collect results
        n_ok = n_err = 0
        for r in client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                all_results[r.custom_id] = extract_text(r.result.message)
                n_ok += 1
            else:
                all_results[r.custom_id] = ""
                n_err += 1
        
        print(f"Batch {batch_idx + 1} done: succeeded={n_ok} failed={n_err}")

    # Keep original CSV columns and add generated
    df["generated"] = df["reaction_id"].astype(str).map(all_results).fillna("")
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    total_ok = sum(1 for v in all_results.values() if v)
    print(f"\nDone. Total succeeded={total_ok} failed={len(df) - total_ok}")
    print(f"Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
