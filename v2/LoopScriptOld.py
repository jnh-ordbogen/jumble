from TranslationEvaluator import TranslationEvaluator
from tqdm import tqdm
import torch.multiprocessing as mp
from Tictoc import Tictoc

def read_file_lines(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()
    return [line.strip() for line in lines]


def process_in_chunks(evaluator, danish_file, english_file, chunk_size):
    
    danish_sentences = read_file_lines(danish_file)
    correct_sentences = read_file_lines(english_file)

    # For starting the program where we left off
    #cutoff_index = correct_sentences.index("After working in the same company for many years, it was time to go their separate ways and try something new.")
    cutoff_index = 0
    danish_sentences = danish_sentences[cutoff_index+1:]
    correct_sentences = correct_sentences[cutoff_index+1:]
    print(danish_sentences[0])
    print(correct_sentences[0])
    num_chunks = len(danish_sentences) // chunk_size + (1 if len(danish_sentences) % chunk_size else 0)

    for i in tqdm(range(num_chunks)):
        try:
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk_danish = danish_sentences[start_idx:end_idx]
            chunk_correct = correct_sentences[start_idx:end_idx]

            evaluator.process(chunk_danish, chunk_correct)
            caches = evaluator.get_num_use_cache()
            infers = evaluator.get_num_use_translator()
            cache_percentage = int(caches / (infers + caches) * 100)
            print(f'{evaluator.device} Cache percentage: {cache_percentage}%, ({caches} caches, {infers} infers)')
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            # Optionally, log the error or take other actions before continuing.
            continue

def process_chunk(chunk_num):
    device = 0
    print(f"Processing chunk {chunk_num} on device {device}")
    evaluator = TranslationEvaluator(device=device, output_path=f'bad_idioms_2.csv')
    process_in_chunks(evaluator, f'../datasets/danish_idioms.txt', f'../datasets/english_idioms.txt', 1000)

def main():
    # Start a multiprocessing context
    mp.set_start_method('spawn')

    # Define the pool of processes
    pool = mp.Pool()

    # Map the function to process each chunk to the pool
    #pool.map(process_chunk, [0,1])
    pool.map(process_chunk, [0])

    # Close the pool to release resources
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()

""" def process_in_chunks(danish_file, english_file, chunk_size=1000):
    evaluator = TranslationEvaluator(device=3, output_path='bad_europarl_no_pruning.csv')
    danish_sentences = read_file_lines(danish_file)
    correct_sentences = read_file_lines(english_file)

    # For starting the program where we left off
    #cutoff_index = correct_sentences.index("After working in the same company for many years, it was time to go their separate ways and try something new.")
    cutoff_index = 0
    danish_sentences = danish_sentences[cutoff_index+1:]
    correct_sentences = correct_sentences[cutoff_index+1:]
    print(danish_sentences[0])
    print(correct_sentences[0])
    num_chunks = len(danish_sentences) // chunk_size + (1 if len(danish_sentences) % chunk_size else 0)
    #print(num_chunks)
    for i in tqdm(range(num_chunks)):
        try:
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk_danish = danish_sentences[start_idx:end_idx]
            chunk_correct = correct_sentences[start_idx:end_idx]
            evaluator.process(chunk_danish, chunk_correct)
            #print('Chunks processed:', i+1, '/', num_chunks)
            print('Caches:', evaluator.get_num_use_cache())
            print('Infers:', evaluator.get_num_use_translator())
            print('Percentage: ', int(evaluator.get_num_use_cache() / evaluator.get_num_use_translator() * 100))
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            # Optionally, log the error or take other actions before continuing.
            continue

process_in_chunks('../datasets/Europarl.da-en.da', '../datasets/Europarl.da-en.en', 1000) """