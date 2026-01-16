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
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment")

    df = pd.read_csv(args.input_csv)
    required = {"reaction_id", "prompt", "answer"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    client = anthropic.Anthropic(api_key=api_key)

    # Build batch requests
    requests = []
    for _, row in df.iterrows():
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
    print(f"Submitted Claude batch: {batch_id}  (n={len(requests)})")

    # Poll until done
    while True:
        st = client.messages.batches.retrieve(batch_id)
        if st.processing_status in ("ended", "completed"):
            break
        time.sleep(args.poll_sec)

    # Collect results
    out_map = {}
    n_ok = n_err = 0
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            out_map[r.custom_id] = extract_text(r.result.message)
            n_ok += 1
        else:
            out_map[r.custom_id] = ""
            n_err += 1

    # Keep original CSV columns and add generated
    df["generated"] = df["reaction_id"].astype(str).map(out_map).fillna("")
    df.to_csv(args.output_csv, index=False)

    print(f"Done. succeeded={n_ok} failed={n_err}")
    print(f"Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
