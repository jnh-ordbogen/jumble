def read_file_lines(file_path):
    with open(file_path, mode='r', newline='') as txt_file:
        lines = txt_file.readlines()
    return [line.strip() for line in lines]

def write_lines_to_file(file_path, lines):
    with open(file_path, mode='w', newline='') as txt_file:
        for line in lines:
            txt_file.write(line + '\n')

def divide_files_into_chunks(file1_path, file2_path, num_chunks):
    # Read lines from both files
    lines_file1 = read_file_lines(file1_path)
    lines_file2 = read_file_lines(file2_path)

    # Calculate chunk size
    chunk_size = len(lines_file1) // num_chunks

    # Divide lines into chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else None
        chunk_lines_file1 = lines_file1[start_idx:end_idx]
        chunk_lines_file2 = lines_file2[start_idx:end_idx]

        # Write chunk lines to separate files
        chunk_file1_path = f"{file1_path[:-3]}{i}.da"
        chunk_file2_path = f"{file2_path[:-3]}{i}.en"
        write_lines_to_file(chunk_file1_path, chunk_lines_file1)
        write_lines_to_file(chunk_file2_path, chunk_lines_file2)
        print(f"Chunk {i} written to {chunk_file1_path} and {chunk_file2_path} with length {len(chunk_lines_file1)} and {len(chunk_lines_file2)}")

if __name__ == "__main__":
    file1_path = '../datasets/Europarl.da-en.da'
    file2_path = '../datasets/Europarl.da-en.en'
    num_chunks = 4
    divide_files_into_chunks(file1_path, file2_path, num_chunks)
