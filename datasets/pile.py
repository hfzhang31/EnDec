import pyap
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
import re
import spacy
"""
python -m spacy download en_core_web_sm
"""
from tqdm import tqdm
import json

import os
import multiprocessing

"""
https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
"""

# Download NLTK data if not already downloaded
# nltk.download('punkt')
# nltk.download('names')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('averaged_perceptron_tagger')

def extract_physical_addresses(paragraph):
    addresses = pyap.parse(paragraph, country='US')  # You can specify a different country code if needed
    return [address.full_address for address in addresses]

def extract_named_entities(paragraph):
    # Tokenize the paragraph into sentences and words
    sentences = sent_tokenize(paragraph)
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    # Perform part-of-speech tagging on the tokenized sentences
    pos_tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]

    # Extract proper nouns (NNP) as human names
    human_names = set()
    for pos_tagged_sentence in pos_tagged_sentences:
        for word, pos in pos_tagged_sentence:
            if pos == 'NNP':
                human_names.add(word)

    return list(human_names)

def distill_human_name(full_names):
    human_name = re.compile(r'\b[A-Z][A-Za-z]+[,\s]+[A-Z][A-Za-z]+\b')
    human_names = []
    for full_name in full_names:
        names = re.findall(human_name, full_name)
        human_names += names
    return human_names

def extract_human_names(paragraph):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(paragraph)

    human_names = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            human_names.add(ent.text)
    single_names = []
    full_names = []
    for name in human_names:
        if not ' ' in name:
            single_names.append(name)
        else:
            full_names.append(name)
    for name_a in single_names:
        for name_b in single_names:
            if f'{name_a}, {name_b}' in paragraph:
                full_names.append(f'{name_a}, {name_b}')

    return distill_human_name(full_names)


def extract_phone_numbers(paragraph):
    phone_number = re.compile("[0-9][0-9][0-9][-.()][0-9][0-9][0-9][-.()][0-9][0-9][0-9][0-9]")
    phone_numbers = re.findall(phone_number, paragraph)
    return phone_numbers

def extract_emails(paragraph):
    email_address = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    email_addresses = re.findall(email_address, paragraph)
    return email_addresses

def extract_all(paragraph, log_json):
    try:
        email = extract_emails(paragraph)
        phone = extract_phone_numbers(paragraph)
        address = extract_physical_addresses(paragraph)
        entity = extract_human_names(paragraph) #extract_named_entities(paragraph)
        a = entity
        b = email
        c = phone
        d = address
        for name in a:
            if name in log_json:
                log_json[name]['email'].update(b)
                log_json[name]['phone'].update(c)
                log_json[name]['address'].update(d)
            else:
                log_json[name] = {'email':set(b), 'phone':set(c), 'address':set(d)}
    except Exception as e:
        raise e

def process_chunk(chunk, result_list, index):
    with tqdm(total=len(chunk), desc=f'#{index}', position=index+1) as pbar:
        for input_string in chunk:
            try:
                extract_all(input_string, result_list)
                pbar.update(1)
            except Exception as e:
                pass

def get_all_file_contents(directory):
    file_contents = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    file_contents.append(file.read())
                except UnicodeDecodeError:
                    pass

    return file_contents


if __name__ == "__main__":
    directory = "./Pile/maildir/"  # Replace with the path to your directory
    num_cpus = 40
    all_file_contents = get_all_file_contents(directory)
    print(f"extract {len(all_file_contents)} files")
    chunk_size = len(all_file_contents) // num_cpus
    logs = {}
    manager = multiprocessing.Manager()
    result_list = manager.dict()

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_cpus) as pool:
        for i in tqdm(range(0, len(all_file_contents), chunk_size), desc="Processing"):
            chunk = all_file_contents[i:i + chunk_size]
            pool.apply_async(process_chunk, args=(chunk, result_list, int(i/chunk_size)))
        pool.close()
        pool.join()
    logs = dict(result_list)
    #print(logs.keys())
    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    with open('./pile_email.json', 'w+') as logfile:
        json.dump(logs, logfile, default=set_default)
        