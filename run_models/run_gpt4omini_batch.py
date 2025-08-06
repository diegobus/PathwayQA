import argparse
import os
import pandas as pd
import json
import openai
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch entity-extraction requests to OpenAI GPT-4.1"
    )
    p.add_argument(
        "--input_csv", "-i", required=True,
        help="Path to input CSV with columns: reaction_id,prompt,answer,length,generated"
    )
    p.add_argument(
        "--output_csv", "-o", required=True,
        help="Where to write the CSV with added api_response, entity_presence, and extra_count columns"
    )
    p.add_argument(
        "--model", "-m", default="gpt-4o-mini",
        help="OpenAI Chat API model to use (e.g., gpt-4, gpt-4-turbo)"
    )
    p.add_argument(
        "--env-file", default="../.env",
        help="Path to .env file containing OPENAI_API_KEY"
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of concurrent API requests to run"
    )
    return p.parse_args()


def process_row(idx, row, model) -> tuple:
    # Prepare text
    prompt = str(row.get("prompt", ""))
    try:
        resp = openai.responses.create(
                        model=model,
                        input=prompt)
        text = resp.output[0].content[0].text
    except Exception as e:
        print(e)
        text = ""
    return idx, text


def main():
    args = parse_args()
    if Path(args.output_csv).exists():
        print(f"Output file {args.output_csv} already exists. Exiting.")
        return
    # Load API key
    load_dotenv(dotenv_path=Path(args.env_file))
    key = os.getenv("API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in env or .env file")
    openai.api_key = key

    # Load data
    print (f'Input File: {args.input_csv}')
    df = pd.read_csv(args.input_csv)
    n = len(df)
    print(f"Processing {n} rows in batches of {args.batch_size}")

    # Prepare result container
    responses = [None] * n
    
    # Submit concurrently
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(process_row, idx, row, args.model): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(futures):
            idx = futures[future]
            _, text = future.result()
            responses[idx] = text
            if idx % 100 == 0:
                print(f"Completed row {idx}")

    # Append and save
    df["generated"] = responses
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {n} rows to {args.output_csv!r}")


if __name__ == "__main__":
    main()
