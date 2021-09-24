# for 1000G vcf files in the longleaf /datacommons/1000genomes directory

import pandas as pd
import os
import csv
import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--chrom_list", default="14", help="list of chromosomes you want to grab data from")
    parser.add_argument("--popfile", default="/datacommons/1000genomes/integrated_call_samples_v3.20130502.ALL.panel", help="File mapping individuals IDs with their respective populations")
    parser.add_argument("--population", default="YRI", help="What population you want to extract")
    parser.add_argument("--break_count", default="None", help="How many samples you want to pull before ending")
    parser.add_argument("--extract_frequency", default="100000")
    parser.add_argument("--extract_length", default="1000", help="How long you want each sample to be")
    parser.add_argument("--odir", default="None")
    parser.add_argument("--consecutive_sites", action="store_true", help="Whether you want data to be adjacent")
    parser.add_argument("--threshold", default=".25", help="What minimum percentage of alleles must be variants")

    args = parser.parse_args()

    if args.odir != "None" and not os.path.exists(args.odir):
        os.mkdir(args.odir)

    return args


def has_ones(elem):
    if "1" in elem:
        return True
    else: 
        return False


def main():

    args = parse_args()
    key_dataframe = pd.read_csv(args.popfile, delimiter="\t")
    key_dataframe.drop(columns=['Unnamed: 4', "Unnamed: 5"], inplace=True)
    key_dataframe = key_dataframe[key_dataframe['pop'] == args.population]
    pop_samples = list(key_dataframe['sample'].values)
    pop_size = len(pop_samples)
    chrom_list = ["/datacommons/1000genomes/ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf".format(chrom)
        for chrom in args.chrom_list.split(",")]



    full_data = pd.DataFrame()
    location = []

    for num, chrom_file in enumerate(chrom_list):

        keep_columns = []
        data = {}
        segregating_site = []
        getting_data = False
        sites_count = 0
        i = 0

        with open(chrom_file) as csvfile:
            file_reader = csv.reader(csvfile, delimiter='\t')
                
            for line_count, row in enumerate(file_reader):

                if not row[0].startswith('#'):
                    if args.consecutive_sites: # pulling based on consecutive sites
                        if (i % int(args.extract_frequency) == 1 and i != 1) or getting_data == True:
                            getting_data = True
                            yri_data = [row[elem] for elem in keep_columns]
                            location.append((num, line_count+1, row[1]))
                            segregating_site.append(yri_data)     
                            
                            if len(segregating_site) == int(args.extract_length):
                                data["site{}".format(sites_count)] = segregating_site.copy()

                                segregating_site = []
                                getting_data = False
                                sites_count += 1
                                if args.break_count != "None":
                                    if sites_count > int(args.break_count): # for debugging/getting sample
                                        break 
                    else: # pulling based on diversity 
                        yri_data = [row[elem] for elem in keep_columns]
                        has_reference_allele = list(map(has_ones, yri_data))
                        fraction = has_reference_allele.count(True) / pop_size
                        if fraction > float(args.threshold) and fraction < 1-float(args.threshold):
                            location.append((num, line_count+1, row[1]))
                            segregating_site.append(yri_data)     

                        if len(segregating_site) == int(args.extract_length):
                            data["site{}".format(sites_count)] = segregating_site.copy()

                            segregating_site = []
                            sites_count += 1
                            if args.break_count != "None":
                                if sites_count > int(args.break_count): # for debugging/getting sample
                                    break 
                    i += 1

                if row[0] == ("#CHROM"):
                    print(len(row))
                    for j, label in enumerate(row):
                        if label in pop_samples:
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

        data_one = full_dataframe_one.T.values.reshape(-1, int(args.extract_length))
        data_two = full_dataframe_two.T.values.reshape(-1, int(args.extract_length))
        data_one_df = pd.DataFrame(data_one)
        data_two_df = pd.DataFrame(data_two)
        # pd.concat((data_one_df, data_two_df)).to_csv(os.path.join(args.odir, "full_dataframe.csv"), index=False)
        full_chrom = pd.concat((data_one_df, data_two_df))
        full_chrom["chrom"] = args.chrom_list.split(",")[num]

        full_data = pd.concat((full_data, full_chrom))

    full_data.to_csv(os.path.join(args.odir, "chrom_data.csv"), index=False)
    if args.odir != "None":
        with open(os.path.join(args.odir, "locations.txt"), "w") as f:
            f.write(str(location))
            f.close()


if __name__ == "__main__":
    main()