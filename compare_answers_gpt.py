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
        "--model", "-m", default=" ",
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
    p.add_argument(
        "--col", type=str, default="generated",
        help="Column containing generated outputs to evaluate"
    )
    return p.parse_args()

def format_entity(ent: dict) -> str:
    """
    Format one entity dict as "Name (ID)" if an ID exists, else just "Name".
    Checks in order: uniprot, chebi, ensemble.
    """
    name = ent.get("name", "")
    # pick first non‑None, non‑empty id
    for key in ("uniprot", "chebi", "dna_id"):
        val = ent.get(key)
        if val:
            return f"{name} ({key}:{val})"
    return name
def format_outputs(outputs: list[dict]) -> list[str]:
    out = []
    for ent in outputs:
        if ent.get("complex"):
            main_fmt   = format_entity(ent)
            comps_fmt  = [format_entity(c) for c in ent.get("components", [])]

            out.append(main_fmt)
            for x in comps_fmt:
                out.append(x)
        else:
            out.append(format_entity(ent))
    return out

def make_reaction_dict():
    reaction_file = '../reactome_data/reactions.json'
    with open(reaction_file, "r") as f:
        reaction_data = json.load(f)

    reaction_outputs_dict = {}
    for reaction in reaction_data:
        reaction_id = reaction['reaction_id']
        output_list = reaction['outputs']
        output_list_format = format_outputs(output_list)
        reaction_outputs_dict[reaction_id] = output_list_format
    return reaction_outputs_dict



def get_reactants(reaction_id, reaction_outputs_dict):
    return reaction_outputs_dict[reaction_id]

def make_prompt(reference: str, generated: str, reaction_id: str, reaction_outputs_dict:dict) -> str:
    ref_output_list = get_reactants(reaction_id, reaction_outputs_dict)
    return (
        "You are given a reference answer (in the form of a list of products) and a generated answer. "
        "For each protein, chemical compound, gene product, or complex entity in the reference product list, determine whether the generated text answer contains each of the entities."
        "Be lenient in determining the presence of the entity in the generated answer. Allow for the following discrepancies:"
        "1. The subcellular localization, which is indicated in brackets (e.g. [periplasmic space], [cytosol], [nucleoplasm]), does not need to match"
        "2. The exact text used to describe the name of the entities does not need to match perfectly; instead make sure the entities match in meaning between the generated and references answer. Small differences like phosphorylated residue index or partial charge do not matter. Be lenient."
        "3. Allow for differences between mrna, gene, or product forms of the entity. Consider all three version of the same entity as a match."
        "4. Allow for mismatches between the id number, if present. If the names in the reference list and generated text have the same meaning and the id number do not match, consider that entity as a match."
        "5. If the reference compound appears anywhere in the generated answer, it counts as a match, regardless of whether or not that section of the generation appears relevant."
        "6. When considering a complex, consider the listed constituents of the complex and not simply the name of the complex. A complex is denoted as '[complex name] [subcellular localization] (complex of [constituent], [constituent], ...)'. Do not consider the complex name and do not count the constituents of the complex separately from the complex. If only a small portion of the complex's constituents are present, do not say the complex is present."
        "7. If the generated answer is blank or NaN, then return False for all reference products."
        "Again, be very lenient, and allow for matches even if they are not exactly chemically or biologically alike. \n\n"
        f"Reference answer: {ref_output_list}\n"
        f"Generated answer: {generated}\n\n"
        "Respond only with a valid JSON with the following structure (Do not include ANY explanatory text or code fences.):\n"
        "{\"Entity1\": true, \"Entity2\": false, ...},\n"
    )


def process_row(idx, row, model, reaction_outputs_dict, gen_col="generated") -> tuple:
    # Prepare text
    reaction_id = row['reaction_id']
    gen = str(row.get(gen_col, "")).replace("\n", " ")
    ref = str(row.get("answer", "")).replace("\n", " ")
    prompt = make_prompt(ref, gen,reaction_id, reaction_outputs_dict)

    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()
    if not content:
        raise ValueError(f"Empty response from LLM for reaction.")
    
    if len(content) > 7 and content[:7] == "```json" and content[-3:] == "```":
        content = content[7:-3].strip()
    try:
        return json.loads(content)
    except:
        print(content)
        print(f"error processing result: {reaction_id}")
        return None
    '''except Exception as e:
        text = ""
        presence = None
        extra = None'''
    return idx, data


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

    reaction_outputs_dict = make_reaction_dict()

    # Prepare result containers

    entity_presence = [None] * n

    # Submit concurrently
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(process_row, idx, row, args.model, reaction_outputs_dict, args.col): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(futures):
            idx = futures[future]
            presence = future.result()
            entity_presence[idx] = presence
            if idx % 1000 == 0:
                print(f"Completed row {idx}")

    # Append and save
    df["entity_presence"] = entity_presence

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {n} rows to {args.output_csv!r}")


if __name__ == "__main__":
    main()
