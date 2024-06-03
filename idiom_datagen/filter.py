import csv

def filter_and_write_csv(input_file, output_file):
  """Filters and writes a new CSV file based on specified criteria.

  Args:
      input_file (str): Path to the input CSV file.
      output_file (str): Path to the output CSV file.
  """
  try:
    with open(input_file, 'r', encoding='utf-8') as input_csv, open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
      reader = csv.DictReader(input_csv)
      writer = csv.DictWriter(output_csv, fieldnames=["combination", "translation"])
      writer.writeheader()
      for row in reader:
        if row['direction'] == "daen" and str_has_more_than_n_words(3, row['combination']):
          writer.writerow({"combination": row['combination'], "translation": row['translation']})
  except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
  except PermissionError:
    print(f"Error: Insufficient permissions to access file '{input_file}' or create '{output_file}'.")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")

def str_has_more_than_n_words(n, s):
  """Returns True if the string has more than n words, False otherwise.

  Args:
      n (int): The minimum number of words.
      s (str): The string to check.

  Returns:
      bool: True if the string has more than n words, False otherwise.
  """
  return len(s.split()) > n

# Replace with the actual paths to your files
input_file = "combinations.csv"
output_file = "filtered_combinations.csv"

filter_and_write_csv(input_file, output_file)

print(f"Successfully filtered and written data to '{output_file}'.")
