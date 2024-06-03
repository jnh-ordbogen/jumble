from TranslationEvaluator import TranslationEvaluator
from tqdm import tqdm

def read_file_lines(file_path):
    with open(file_path, mode='r', newline='') as txt_file:
        lines = txt_file.readlines()
    return [line.strip() for line in lines]

def process_in_chunks(danish_file, english_file, chunk_size=1000):
    danish_sentences = read_file_lines(danish_file)
    correct_sentences = read_file_lines(english_file)

    # For starting the program where we left off
    cutoff_index = correct_sentences.index("After working in the same company for many years, it was time to go their separate ways and try something new.")
    #cutoff_index = 0
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

            evaluator = TranslationEvaluator(chunk_danish, chunk_correct)
            evaluator.process(is_idiom=True)
            print('Chunks processed:', i+1, '/', num_chunks)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            # Optionally, log the error or take other actions before continuing.
            continue

#process_in_chunks('datasets/Europarl.da-en.da', 'datasets/Europarl.da-en.en')
process_in_chunks('datasets/danish_idioms.txt', 'datasets/english_idioms.txt', 500)
