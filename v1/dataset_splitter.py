import csv
import random
import os

def split_data(data, train_percent, val_percent, test_percent):
    total_len = len(data)
    train_len = int(total_len * train_percent)
    val_len = int(total_len * val_percent)
    test_len = total_len - train_len - val_len
    return (
        data[:train_len],
        data[train_len:train_len + val_len],
        data[train_len + val_len:]
    )

def write_to_file(data, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        for row in data:
            file.write(row + '\n')

def main(csv_file_path, output_dir, train_percent, val_percent, test_percent):
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    random.shuffle(data)

    train_data, val_data, test_data = split_data(data, train_percent, val_percent, test_percent)
    print(f'Train: {len(train_data)} Val: {len(val_data)} Test: {len(test_data)}')
    print(f'Total: {len(train_data) + len(val_data) + len(test_data)}')
    print(f'Percentages:')
    print(f'Train: {len(train_data) / len(data)} Val: {len(val_data) / len(data)} Test: {len(test_data) / len(data)}')
    write_to_file([row[1] for row in train_data], f'{output_dir}/train.src')
    write_to_file([row[0] for row in train_data], f'{output_dir}/train.tgt')
    write_to_file([row[1] for row in val_data], f'{output_dir}/val.src')
    write_to_file([row[0] for row in val_data], f'{output_dir}/val.tgt')
    write_to_file([row[1] for row in test_data], f'{output_dir}/test.src')
    write_to_file([row[0] for row in test_data], f'{output_dir}/test.tgt')

if __name__ == '__main__':
    csv_file_path = "full_slayer_dataset.csv"
    output_dir = "datasets/slayer"
    train_percent = 0.8
    val_percent = 0.10
    test_percent = 0.10

    main(csv_file_path, output_dir, train_percent, val_percent, test_percent)
