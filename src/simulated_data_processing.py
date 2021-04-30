import pandas as pd
import os
import csv
import argparse

""" This file takes in a directory of unzipped data (unzipped by simulated_data_extractor.py)
    and converts it into a single .csv file that can be fed in to the GANs. Use the --length
    argument to specify how long you want your sequences for training to be. Use the 
    --break_count argument if you only want to use a certain number of samples for each file """

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--idir", default="/pine/scr/n/i/nickmatt/unzipped_segSite_data")
    parser.add_argument("--ifile", default="None")
    parser.add_argument("--length", default="1000")
    parser.add_argument("--ofile", default="simulated_data.csv")
    parser.add_argument("--break_count", default="None")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ifile != "None":
        files = [args.ifile]
    else:
        files = os.listdir(args.idir)

    df_full = pd.DataFrame()

    for file in files:
        data_file = os.path.join(args.idir, file)
        print(data_file)
        data = pd.read_csv(data_file)
        
        # there is probably a more efficient way to do this
        for i in range(len(data)):
            slice = data.iloc[i, :].dropna()
            if len(slice) > int(args.length):
                df_full = df_full.append(slice[:int(args.length)])
            if args.break_count != "None":
                if i > args.break_count:
                    break

    df_full.dropna(inplace=True)
    df_full.to_csv(args.ofile, index=False)


if __name__ == "__main__":
    main()