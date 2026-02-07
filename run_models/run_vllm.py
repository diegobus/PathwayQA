import os
import pandas as pd
import argparse
from vllm import LLM, SamplingParams


def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.outname)), exist_ok=True)

    df = pd.read_csv(args.incsv)
    df["reaction_id"] = df["reaction_id"].astype(str)

    # Resume: skip rows already in output file
    done_ids = set()
    if os.path.exists(args.outname):
        prev = pd.read_csv(args.outname)
        if "reaction_id" in prev.columns:
            done_ids = set(prev["reaction_id"].astype(str).tolist())
        print(f"Resuming: found {len(done_ids)} completed rows in {args.outname}")

    todo = df[~df["reaction_id"].isin(done_ids)].copy()
    todo.reset_index(drop=True, inplace=True)
    print(f"Total rows: {len(df)} | Remaining: {len(todo)}")

    if len(todo) == 0:
        print("Nothing to do.")
        return

    # Load model
    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
    )
    if args.downcast:
        llm_kwargs["dtype"] = "float16"
    llm = LLM(**llm_kwargs)

    # Sampling
    stop_tokens = [s for s in args.stop.split(",") if s]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=stop_tokens,
    )

    # Batched generation with checkpointing
    for start in range(0, len(todo), args.batch_size):
        batch = todo.iloc[start : start + args.batch_size]
        prompts = batch["prompt"].tolist()

        outputs = llm.generate(prompts, sampling_params)
        gens = [o.outputs[0].text for o in outputs]

        batch_out = batch[["reaction_id", "prompt", "answer"]].copy()
        batch_out["generated"] = gens

        # Append checkpoint
        write_header = not os.path.exists(args.outname) or os.path.getsize(args.outname) == 0
        batch_out.to_csv(args.outname, mode="a", header=write_header, index=False)

        done = min(start + args.batch_size, len(todo))
        print(f"Processed {done}/{len(todo)} remaining rows")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM inference on reaction prompts")
    parser.add_argument("--model", type=str, required=True, help="path to huggingface model directory")
    parser.add_argument("--incsv", type=str, required=True, help="path to input csv")
    parser.add_argument("--outname", type=str, required=True, help="path to output file")
    parser.add_argument("--downcast", action="store_true", help="use dtype=float16")
    parser.add_argument("--stop", type=str, default="END", help="comma-separated stop tokens (default: END)")
    parser.add_argument("--max_tokens", type=int, default=512, help="max tokens to generate (default: 512)")
    parser.add_argument("--max_model_len", type=int, default=4096, help="max model context length (default: 2048)")
    parser.add_argument("--gpu_mem", type=float, default=0.90, help="GPU memory utilization (default: 0.90)")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for generation (default: 64)")
    args = parser.parse_args()
    main(args)
