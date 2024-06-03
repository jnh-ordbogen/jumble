from gpt_generate import generate, generate_batch
import csv
import json
from tqdm import tqdm
import requests

template = """The Danish idiom/phrase/saying "{da}" has the semantically equivalent saying "{en}" in English. Please provide, in the following format, ten parallel sentences in each language that use this phrase in a wider context. Make sure the sentences are unique and creative in choice of theme, subject matter, and context, as well as length. Do not include a single character of meta/conversational text. Just give me the JSON and nothing else, such that it can be parsed from your response directly. Please do not include any line breaks within the strings themselves: {{
  "Danish": [
    sentence1,
    sentence2,
    ...
  ],
  "English": [
    sentence1,
    sentence2,
    ...
  ]
}}"""

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

def response_to_lists(response):
    try:
        json_response = json.loads(response)
        return json_response["Danish"], json_response["English"]
    except json.decoder.JSONDecodeError as e:
        print(e)
        return None, None
    except KeyError as e:
        print(e)
        return None, None

def response_is_valid(da, en):
    return da is not None and en is not None and len(da) == len(en) and len(da) == 10

def write_rows_to_csv(writer, da, en):
    for da_item, en_item in zip(da, en):
        writer.writerow([da_item.strip("'\""), en_item.strip("'\"")])

def main():
    # with open('gyldendal2.csv', 'r', encoding='utf-8') as input_csv:
    with open('output_over_5.csv', 'r', encoding='utf-8') as input_csv:
        reader = csv.DictReader(input_csv)
        requests = [template.format(da=row['l1'], en=row['l2']) for row in reader]

    batch_size = 5
    for i in tqdm(range(0, len(requests), batch_size), desc="Generating"):
        batch_requests = requests[i:i+batch_size]
        batch_responses = generate_batch(batch_requests)
        
        with open('gyldendal_idioms4.csv', 'a', encoding='utf-8', newline='') as output_csv:
            writer = csv.writer(output_csv)
            for response in batch_responses:
                da, en = response_to_lists(response)
                if response_is_valid(da, en):
                    write_rows_to_csv(writer, da, en)
                else:
                    print("Error processing response.")

if __name__ == "__main__":
    main()

# def main():
#     requests = []

#     with open('gyldendal.csv', 'r', encoding='utf-8') as input_csv:
#         reader = csv.DictReader(input_csv)
#         rows = [row for row in reader]
#         for row in rows:
#             da = row['l1']
#             en = row['l2']
#             requests.append(template.format(da=da, en=en))
#     #print(requests[0])
#     #exit()

#     with open('gyldendal_idioms1.csv', 'w', encoding='utf-8', newline='') as output_csv:
#         writer = csv.writer(output_csv)
#         writer.writerow(['Danish', 'English'])
#         for request in tqdm(requests, desc="Generating"):
#             try:
#                 response = generate(request)
#             except Exception as e:
#                 print(e)
#                 send_report("Error", str(e))
#                 continue
#             da, en = response_to_lists(response)
#             if response_is_valid(da, en):
#                 write_rows_to_csv(writer, da, en)
#             else:
#                 print("Error processing response.")

# if __name__ == "__main__":
#     main()

