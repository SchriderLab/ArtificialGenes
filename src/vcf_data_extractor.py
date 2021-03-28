# for 1000G vcf files in the longleaf /datacommons/1000genomes directory

import pandas as pd
import os
import csv
import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ifile", default="/datacommons/1000genomes/ALL.chr14.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf")
    parser.add_argument("--popfile", default="integrated_call_samples_v3.20130502.ALL.panel", help="File mapping individuals IDs with their respective populations")
    parser.add_argument("--population", default="YRI", help="What population you want to extract")
    parser.add_argument("--break_point", default="None", help="how many samples you want to pull before ending")
    parser.add_argument("--extract_frequency", default="100000")
    parser.add_argument("--extract_length", default="1000")
    parser.add_argument("--odir", default="None")

    args = parser.parse_args()

    if args.odir != "None" and not os.path.exists(args.odir):
        os.mkdir(args.odir)

    return args

def main():

    args = parse_args()
    key_dataframe = pd.read_csv(args.key_file, delimiter="\t")
    key_dataframe.drop(columns=['Unnamed: 4', "Unnamed: 5"], inplace=True)
    key_dataframe = key_dataframe[key_dataframe['pop'] == args.population]
    yri_samples = list(key_dataframe['sample'].values)

    keep_columns = []
    data = {}
    segregating_site = []
    location = []
    getting_data = False
    sites_count = 0
    i = 0

    with open(args.ifile) as csvfile:
        file_reader = csv.reader(csvfile, delimiter='\t')
            
        # pulls out 1000 consecutive sites every 100k sites for the individuals we're interested in
        for row in file_reader:

            if not row[0].startswith('#'):
                if (i % int(args.extract_frequency) == 1 and i != 1) or getting_data == True:
                    getting_data = True
                    yri_data = [row[elem] for elem in keep_columns]
                    location.append(row[1])
                    segregating_site.append(yri_data)     
                    
                    if len(segregating_site) == int(args.extract_length):
                        data["site{}".format(sites_count)] = segregating_site.copy()

                        segregating_site = []
                        getting_data = False
                        sites_count += 1
                        if args.break_point != "None":
                            if sites_count > int(args.break_point): # for debugging/getting sample
                                break 
                i += 1
            if row[0] == ("#CHROM"):
                print(len(row))
                for j, label in enumerate(row):
                    if label in yri_samples:
                        keep_columns.append(j)

    full_dataframe_one = pd.DataFrame([])
    full_dataframe_two = pd.DataFrame([])

    for sequence in data.keys():
        data_temp = pd.DataFrame(data[sequence])
        
        sequence_df_one = pd.DataFrame([])
        sequence_df_two = pd.DataFrame([])
        
        for i, column in enumerate(data_temp.columns):
            temp = data_temp[column].str.split("|", expand=True)
            sequence_df_one[args.population+"_{}".format(i)] = temp[0]
            sequence_df_two[args.population+"_{}".format(i)] = temp[1]
            
        full_dataframe_one = pd.concat([full_dataframe_one, sequence_df_one], ignore_index=True)
        full_dataframe_two = pd.concat([full_dataframe_two, sequence_df_two], ignore_index=True)

    full_dataframe_one.to_csv(os.path.join(args.odir, "full_dataframe_one.csv"), index=False)
    full_dataframe_two.to_csv(os.path.join(args.odir, "full_dataframe_two.csv"), index=False)


if __name__ == "__main__":
    main()