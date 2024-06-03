# Concatenates four CSV files into a single CSV file

import csv

def concatenate_csv_files(input_files, output_file):
    # Write the header row
    header_written = False

    # Concatenate the input files
    with open(output_file, mode='w', newline='', encoding='utf-8') as out_csv_file:
        out_csv_writer = csv.writer(out_csv_file)
        header = ['Bad Translation', 'Ground truth']
        out_csv_writer.writerow(header)
        for input_file in input_files:
            with open(input_file, mode='r', newline='', encoding='utf-8') as in_csv_file:
                in_csv_reader = csv.reader(in_csv_file)
                for row in in_csv_reader:
                    if not header_written:
                        out_csv_writer.writerow(row)
                        header_written = True
                    else:
                        out_csv_writer.writerow(row)

if __name__ == "__main__":
    input_files = [f'bad_europarl{i}.csv' for i in range(4)]
    output_file = 'bad_europarl_full.csv'
    concatenate_csv_files(input_files, output_file)