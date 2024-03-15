# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import pandas as pd
from tqdm import tqdm
#arguments to be parsed from command line

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float, help="alpha probability")
args = ap.parse_args()

def add_substrings(list1, list2):

    return [l1 + ". " + l2 if l1!="" else l2 for l1, l2 in zip(list1,list2)]

def gen_eda_df(input_df: pd.DataFrame, alpha_sr: float, alpha_ri:float, alpha_rs:float, alpha_rd:float, num_aug: int = 9, seed: int = 1) -> pd.DataFrame:

    output_df = {'text':[], 'label':[]}

    for i, line in tqdm(input_df.iterrows(), total=len(input_df["text"])):

        label = line["label"]
        sentences = line["text"].split('.')

        sentences = [sentence for sentence in sentences if len(sentence) > 1 ]

        lag = [""]*(num_aug+1)
        for sentence in sentences:
            if sentence == "" or sentence == None:
                continue
            aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug, seed=seed)
            lag = add_substrings(lag, aug_sentences)

        output_df["text"].extend(lag)
        output_df["label"].extend([label]*(len(lag)))

    output_df = pd.DataFrame(output_df)

    return output_df

#main function
if __name__ == "__main__":

    input_df = pd.read_csv(args.input)

    output_df = gen_eda_df(input_df, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)

    output_df.to_csv(args.output, index=False)