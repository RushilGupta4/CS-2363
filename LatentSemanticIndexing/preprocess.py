import json
from math import log
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy import linalg
import re
import os
from typing import Dict, List, Set

with open("books_raw.json", "r") as f:
    books_obj = json.load(f)

print(f"Loaded {len(books_obj)} books")


def get_term_freqs(s: str) -> Dict[str, int]:
    s = re.sub(r"[^\w\s]", "", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    terms = s.split()
    return Counter(terms)


titles: List[str] = []
seen_titles: Set[str] = set()
term_freqs: List[Dict[str, int]] = []
global_term_freqs: Dict[str, int] = {}

for book in books_obj.values():
    title = book["title"]

    if (
        not title.isascii()
        or (freqs := get_term_freqs(title)) == {}
        or title.lower() in seen_titles
    ):
        continue

    for term, freq in freqs.items():
        global_term_freqs[term] = global_term_freqs.get(term, 0) + freq

    titles.append(title)
    seen_titles.add(title.lower())
    term_freqs.append(get_term_freqs(title))


# only keep terms that appear at least 2 times
global_term_freqs = {
    term: freq for term, freq in global_term_freqs.items() if freq >= 1
}

term_to_index = {term: i for i, term in enumerate(global_term_freqs)}
num_terms, num_docs = len(global_term_freqs), len(titles)

print(f"Titles: {num_docs} | Terms: {num_terms}")

data, row_indices, col_indices = [], [], []

for doc_idx, tf in enumerate(term_freqs):
    for term, freq in tf.items():
        if term not in term_to_index:
            continue

        row_indices.append(term_to_index[term])
        col_indices.append(doc_idx)
        data.append(freq)

term_doc_matrix = csr_matrix(
    (data, (row_indices, col_indices)), shape=(num_terms, num_docs)
)

np.save("term_doc_matrix.npy", term_doc_matrix)

# Create folders
folders = ["occurrences", "normalized", "word_length", "tfidf"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)


# Function to create and save term-document matrix
def create_and_save_matrix(data, row_indices, col_indices, folder):
    term_doc_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(num_terms, num_docs)
    )
    np.save(f"{folder}/term_doc_matrix.npy", term_doc_matrix.toarray())
    print(f"Document-Term matrix shape for {folder}: {term_doc_matrix.shape}")

    term_doc_matrix = np.load(f"{folder}/term_doc_matrix.npy").astype(np.float64)
    U, S, Vt = linalg.svd(term_doc_matrix, full_matrices=False)

    # Save SVD components
    np.save(f"{folder}/U.npy", U)
    np.save(f"{folder}/S.npy", S)
    np.save(f"{folder}/Vt.npy", Vt)
    print(
        f"Saved SVD components for {folder} | U: {U.shape} | S: {S.shape} | Vt: {Vt.shape}"
    )


# Prepare data for different matrices
occurrences_data, normalized_data, word_length_data, tfidf_data = (
    [],
    [],
    [],
    [],
)
row_indices, col_indices = [], []

doc_freq = {term: 0 for term in term_to_index}
for tf in term_freqs:
    for term in tf:
        if term in term_to_index:
            doc_freq[term] += 1

idf = {term: log(num_docs / (freq + 1)) for term, freq in doc_freq.items()}

for doc_idx, tf in enumerate(term_freqs):
    total_words = sum(tf.values())
    for term, freq in tf.items():
        if term not in term_to_index:
            continue

        term_idx = term_to_index[term]
        row_indices.append(term_idx)
        col_indices.append(doc_idx)

        occurrences_data.append(freq)
        normalized_data.append(freq / total_words)
        word_length_data.append((freq / total_words) * log(len(term)))
        tfidf_data.append((freq / total_words) * idf[term])

# Create and save matrices
create_and_save_matrix(occurrences_data, row_indices, col_indices, "occurrences")
create_and_save_matrix(normalized_data, row_indices, col_indices, "normalized")
create_and_save_matrix(word_length_data, row_indices, col_indices, "word_length")
create_and_save_matrix(tfidf_data, row_indices, col_indices, "tfidf")

# Save term_to_index and titles
with open("term_to_index.json", "w") as f:
    json.dump(term_to_index, f, indent=4)

with open("titles.json", "w") as f:
    json.dump(titles, f, indent=4)

with open("doc_freq.json", "w") as f:
    json.dump(doc_freq, f)

print(f"Saved {len(titles)} books")
print(f"Document-Term matrix shape: {term_doc_matrix.shape}")
