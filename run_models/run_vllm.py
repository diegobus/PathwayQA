import pandas as pd
import argparse
from vllm import LLM, SamplingParams


def main(args):
    df = pd.read_csv(args.incsv)
    if args.downcast:
        llm = LLM(model=args.model, dtype="float16")
    else:
        llm = LLM(model=args.model)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512, stop=[args.stop])
    outputs = llm.generate(df["prompt"].tolist(), sampling_params)
    df["generated"] = [elem.outputs[0].text for elem in outputs]
    df.to_csv(args.outname, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to huggingface model directory")
    parser.add_argument("--outname", type=str, help="path to output file")
    parser.add_argument("--downcast", action="store_true", help="flag to add dtype=float16 to llm instantiation")
    parser.add_argument("--stop", type=str, help="stop tokens to specify", default="\n")
    parser.add_argument("--incsv", type=str, help="path to input csv")
    args = parser.parse_args()
    main(args)
