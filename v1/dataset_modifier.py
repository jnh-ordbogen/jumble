import random
import csv

def subset(lst, subset_size):
    return lst[:subset_size]

def find_subset_size(other_file_size, percentage_size=0.02):
    other_file_percent = 1.0 - percentage_size

    #print((percentage_size / other_file_percent) * other_file_size)
    #print(int((percentage_size / other_file_percent) * other_file_size))

    return int((percentage_size / other_file_percent) * other_file_size)

def csv_file_to_list_of_rows(csv_file_path):
    rows = []
    with open(csv_file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # Append only the first two columns to the list
            rows.append(row[:2])
    return rows

rows1 = csv_file_to_list_of_rows('bad_idioms.csv')
rows2 = csv_file_to_list_of_rows('bad_translations.csv')

all_rows = rows1 + rows2
print('all_rows: ', len(all_rows))
random.shuffle(all_rows)

subset_size = find_subset_size(len(all_rows))
print(subset_size)
subset_rows = subset(all_rows, subset_size)

for row in subset_rows:
    row[1] = row[0]

full_rows = all_rows + subset_rows

print(len(all_rows))
print(len(subset_rows))
print(len(full_rows))

# with open('full_dataset.csv', mode='w', newline='') as csv_file:
    # writer = csv.writer(csv_file)
    # for row in full_rows:
        # writer.writerow(row[:2])  # Write only the first two columns to the CSV file
