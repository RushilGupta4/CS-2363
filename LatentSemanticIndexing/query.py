import re
from typing import Dict
import numpy as np
import json
from collections import Counter
from math import log
import pandas as pd
from numba import njit

# Load the term-document matrix and SVD components for each type
folders = ["occurrences", "normalized", "word_length", "tfidf"]
matrices = {}
svd_components = {}

for folder in folders:
    matrices[folder] = np.load(f"{folder}/term_doc_matrix.npy")
    svd_components[folder] = {
        "U": np.load(f"{folder}/U.npy"),
        "S": np.load(f"{folder}/S.npy"),
        "Vt": np.load(f"{folder}/Vt.npy"),
    }

with open("term_to_index.json", "r") as f:
    term_to_index = json.load(f)

with open("titles.json", "r") as f:
    titles = json.load(f)

# Load doc_freq
with open("doc_freq.json", "r") as f:
    doc_freq = json.load(f)


def get_term_freqs(s: str) -> Dict[str, int]:
    s = re.sub(r"[^\w\s]", "", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    terms = s.split()
    return Counter(terms)


def query_lsi(query, top_n=1, matrix_type="occurrences", dimension=None):
    # Preprocess the query
    query_tf = get_term_freqs(query)

    # Create query vector based on matrix_type
    query_vector = np.zeros(len(term_to_index))
    total_words = sum(query_tf.values())

    for term, freq in query_tf.items():
        if term in term_to_index:
            term_idx = term_to_index[term]
            if matrix_type == "occurrences":
                query_vector[term_idx] = freq
            elif matrix_type == "normalized":
                query_vector[term_idx] = freq / total_words
            elif matrix_type == "word_length":
                query_vector[term_idx] = (freq / total_words) * log(len(term))
            elif matrix_type == "tfidf":
                idf = log(len(titles) / (doc_freq.get(term, 0) + 1))
                query_vector[term_idx] = (freq / total_words) * idf

    # Get SVD components for the selected matrix type
    U = svd_components[matrix_type]["U"]
    S = svd_components[matrix_type]["S"]
    Vt = svd_components[matrix_type]["Vt"]

    if dimension is not None:
        U = U[:, :dimension]
        S = S[:dimension]
        Vt = Vt[:dimension, :]

    # Transform query vector to LSI space
    query_vector_lsi = np.dot(query_vector, U) * S

    # Precompute the transformed document vectors
    doc_vectors = np.dot(np.diag(S), Vt).T

    norms = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector_lsi)
    similarities = np.dot(doc_vectors, query_vector_lsi) / np.where(
        norms != 0, norms, 1
    )

    # Get top N results
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({"title": titles[idx], "similarity": similarities[idx]})

    return results


if __name__ == "__main__":
    queries = ["Agriculture", "Economics", "Artificial Intelligence", "Policy"]
    dimensions = [100, 500, 2500, None]  # None represents full dimension (8460)

    # Create column names for the DataFrame
    column_names = ["Query", "Matrix Type", "Rank", "Top Result"]
    results_df = []

    for query in queries:
        for matrix_type in folders:
            for dimension in dimensions:
                results = query_lsi(
                    query, matrix_type=matrix_type, top_n=1, dimension=dimension
                )
                dim_name = "8460" if dimension is None else str(dimension)
                row_data = {
                    "Query": query,
                    "Matrix Type": matrix_type,
                    "Rank": dim_name,
                    "Top Result": results[0]["title"],
                }
                results_df.append(row_data)
                print(
                    f"Query: {query}, Matrix: {matrix_type}, Dimension: {dim_name}, Results: {row_data['Top Result']}"
                )

    results_df = pd.DataFrame(results_df)
    results_df.to_csv("lsi_results.csv", index=False)
    print(results_df)
