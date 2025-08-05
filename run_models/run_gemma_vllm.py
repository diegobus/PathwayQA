import torch._dynamo
torch._dynamo.config.suppress_errors = True
import argparse
import pandas as pd
import os


from vllm import LLM, SamplingParams

def main(args):
    # 1) Initialize the vLLM engine
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.8,
        cpu_offload_gb=5.0,
        enforce_eager=True,
        load_format="auto",
        dtype="float16"          # e.g. "float16" or "bfloat16"
    )

    # 2) Prepare sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    # 3) Read prompts from CS[[V
    df = pd.read_csv(args.input_file, header=0, dtype=str)
    cols = set(df.keys().tolist())
    print (cols)
    #df = df.head(1)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    # 4) Run batched inference
    prompts = df["prompt"].tolist()
    outputs = llm.generate(prompts, sampling_params)

    # 5) Collect results
    results = []
    for row, output in zip(df.itertuples(index=False), outputs):
        generated = output.outputs[0].text.strip()
        #print (f'Prompt: \n {row.prompt} \n\n')
        #print (f'Generated: \n {generated} \n\n')
        
        if ('reaction_id' in cols):
            results.append({
                "reaction_id": row.reaction_id,
                "prompt":      row.prompt,
                "answer":      row.answer,
                "length":      len(row.prompt),
                "generated":   generated,
            })
        elif ('pathway_id' in cols):
            results.append({
                "pathway_id": row.pathway_id,
                "prompt":      row.prompt,
                "answer":      row.answer,
                "length":      len(row.prompt),
                "generated":   generated,
            })

    # 6) Write out
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_file, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vLLM on a CSV of prompts and dump products"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the vLLM‐compatible model directory"
    )
    parser.add_argument(
        "--input_file", "-i",
        required=True,
        help="Path to input CSV with columns: reaction_id, prompt, answer"
    )
    parser.add_argument(
        "--out_file", "-o",
        required=True,
        help="Path to write output CSV"
    )
    
    args = parser.parse_args()
    main(args)
