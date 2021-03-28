# for simulated data

import pandas as pd
import os
import gzip
import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--idir",
                        default="/proj/dschridelab/simulationStudies/posSelVsBgs/backupFromPine/posSelVsBgs/sweeps/humanEquilib/trainingData/")
    parser.add_argument("--ifile", default="None")
    parser.add_argument("--odir", default="None")

    args = parser.parse_args()

    if args.odir != "None" and not os.path.exists(args.odir):
        os.mkdir(args.odir)

    return args


def main():
    args = parse_args()

    if args.ifile != "None":
        ifiles = [os.path.join(args.idir, args.ifile)]
    else:
        ifiles = [os.path.join(args.idir, file) for file in os.listdir(args.idir) if file.endswith(".gz")]

    for file in ifiles:
        with gzip.open(file, 'rt') as f:
            file_content = f.read()
            pre = file_content.split("\n")
            data = [list(x) for x in pre if x.startswith("0") or x.startswith("1")]
            del data[0] # first one is just file metadata
            dataframe = pd.DataFrame(data)
            dataframe.to_csv(os.path.join(args.odir, file.split(".")[0] + ".csv"), index=False)


if __name__ == "__main__":
    main()