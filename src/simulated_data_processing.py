import pandas as pd
import os
import csv
import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--idir", default="/pine/scr/n/i/nickmatt/unzipped_segSite_data")
    parser.add_argument("--ifile", default="None")
    parser.add_argument("--length", default="1000")
    parser.add_argument("--ofile", default="./simulated_data.csv")
    parser.add_argument("--break_count", default="None")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ifile != "None":
        files = list(ifile)
    else:
        files = os.listdir(args.idir)

    df_full = pd.DataFrame()

    for file in files:
        data_file = os.path.join(args.idir, file)
        data = pd.read_csv(data_file)
        
        # there is probably a more efficient way to do this
        for i in range(len(data)):
            df_full = df_full.append(data.iloc[i, :].dropna()[:args.length])
            if args.break_count != "None":
                if i > args.break_count:
                    break

    df_full.dropna(inplace=True)
    df_full.to_csv(args.ofile, index=False)


if __name__ == "__main__":
    main()