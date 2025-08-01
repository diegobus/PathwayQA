import argparse
import os
import pandas as pd
import ast
from pathlib import Path
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine multiple validation CSVs and compute exact_match and num_correct from entity_presence dict"
    )
    parser.add_argument(
        "--model", required=True,
        help="Input CSV files to combine (each must have an entity_presence column containing a dict)"
    )
    parser.add_argument(
        "-o", "--output_csv", 
        help="Output CSV path to write combined results"
    )
    return parser.parse_args()


def process_dict(reaction_id, model, shots, d):
    if (pd.isna(d)):
        print (f'Error found! dict na {shots} : {reaction_id}')
        with open('./validate_errors.txt', 'a') as ef:
            ef.write(f"{reaction_id},{model},{shots}\n")
        return np.nan, np.nan

    d= d.replace('false', 'False').replace('true', 'True')
    d = ast.literal_eval(d)
    '''try:
        d = ast.literal_eval(d)
    except (ValueError, SyntaxError) as e:
        print (e)
        print (reaction_id, model, shots)
        print (d)
        quit()'''
    if (len(d) == 0):
        print (f'Error found! not dict properly {reaction_id}')
        with open('./validate_errors.txt', 'a') as ef:
            ef.write(f"{reaction_id},{model},{shots}\n")
        return 0,0
    exactMatch = list(d.values()).count(True)/len(d.values())#int(all(v is True for v in d.values()))
    numCorrect = list(d.values()).count(True)
    return exactMatch, numCorrect


def main():
    args = parse_args()
    print (f'Model: {args.model}')
    output_fp = f'./validate_processed_avg/{args.model}.csv'
    '''if Path(output_fp).exists():
        print(f"Output file {output_fp} already exists. Exiting.")
        return'''

    df_two = pd.read_csv(f'./validate_new/{args.model}_validate.csv', header=0)
    df_one = pd.read_csv(f'./validate_new/{args.model}_validate_oneExample.csv', header=0)
    df_zero = pd.read_csv(f'./validate_new/{args.model}_validate_zeroShot.csv', header=0)

    final = []
    for index, row in df_two.iterrows():
        if (index%1000 == 0):
            print (f'Processing : {index}/{len(df_two)}')
        reaction_id = row['reaction_id']
            
        row_two = row

        if (reaction_id not in set(df_one['reaction_id'].tolist())):
            with open('./validate_errors.txt', 'a') as ef:
                ef.write(f"{reaction_id},{args.model},one\n")
            continue
        if (reaction_id not in set(df_zero['reaction_id'].tolist())):
            with open('./validate_errors.txt', 'a') as ef:
                ef.write(f"{reaction_id},{args.model},zero\n")
            continue
        row_one = df_one[df_one["reaction_id"] == reaction_id].iloc[0]
        row_zero = df_zero[df_zero["reaction_id"] == reaction_id].iloc[0]


        exactMatch_two, numCorrect_two = process_dict(reaction_id, args.model, 'two', row_two['entity_presence'])
        exactMatch_one, numCorrect_one = process_dict(reaction_id, args.model, 'one', row_one['entity_presence'])
        exactMatch_zero, numCorrect_zero = process_dict(reaction_id, args.model, 'zero', row_zero['entity_presence'])

        if (pd.isna(exactMatch_two) or pd.isna(exactMatch_one) or pd.isna(exactMatch_zero)):
            continue

        final.append({
            'reaction_id': reaction_id, 
            'overall_two': exactMatch_two,
            'numCorrect_two': numCorrect_two,

            'overall_one': exactMatch_one,
            'numCorrect_one': numCorrect_one,

            'overall_zero': exactMatch_zero,
            'numCorrect_zero': numCorrect_zero,

            })

    df_output = pd.DataFrame(final)
    df_output.to_csv(output_fp,index=False)
    print(f"Wrote {len(df_output)} rows to {output_fp}")


if __name__ == '__main__':
    main()