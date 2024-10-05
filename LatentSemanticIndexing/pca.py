import numpy as np
import json
import os
import matplotlib.pyplot as plt
from pprint import pprint

N_TERMS = 20
N_TITLES = 5
SKIP = 0  # New variable to skip singular values

# Define the folders to process
folders = ["occurrences", "normalized", "word_length", "tfidf"]

# Load term_to_index and titles
with open("term_to_index.json", "r") as f:
    term_to_index = json.load(f)

with open("titles.json", "r") as f:
    titles = json.load(f)

# pprint([i for i in titles if "energy" in i])
# quit()

sampled_titles = [
    "India : the energy sector",
    "Politics and society in India",
    # "Lal Bahdur Shastri: A Life in Politics",
    "Complex Analysis",
    # "Microeconomic Analysis",
    "Agriculture in China's modern economic development",
    "The political economy of world energy : a twentieth-century perspective",
]
sampled_titles = {i: titles.index(i) for i in sampled_titles}


# Load doc_freq if needed (not used here, but loaded for completeness)
with open("doc_freq.json", "r") as f:
    doc_freq = json.load(f)

# Reverse mapping from index to term for plotting
index_to_term = {idx: term for term, idx in term_to_index.items()}

# Create the PCA folder if it doesn't exist
pca_folder = "PCA"
os.makedirs(pca_folder, exist_ok=True)

# Iterate over each folder to perform PCA and plotting
for folder in folders:
    print(f"Processing folder: {folder}")

    # Load SVD components
    U = np.load(os.path.join(folder, "U.npy"))
    S = np.load(os.path.join(folder, "S.npy"))
    Vt = np.load(os.path.join(folder, "Vt.npy"))

    # Project terms and titles into 2D using the SKIP+1 and SKIP+2 principal components
    # For terms: Use U * S
    terms_pca = U[:, SKIP : SKIP + 2] * S[SKIP : SKIP + 2]

    # For titles: Use Vt^T * S
    titles_pca = (Vt[SKIP : SKIP + 2, :].T) * S[SKIP : SKIP + 2]

    indices = np.argsort(-np.sum(U[:, SKIP : SKIP + 2] * S[SKIP : SKIP + 2], axis=1))[
        :N_TERMS
    ]
    plt.figure(figsize=(12, 6))
    plt.scatter(terms_pca[:, 0], terms_pca[:, 1], alpha=0.1, label="Terms")
    for idx in indices:
        plt.text(terms_pca[idx, 0], terms_pca[idx, 1], index_to_term[idx], fontsize=8)
    plt.title(f"PCA of Terms with Annotations - {folder.capitalize()}")
    plt.xlabel(f"PC{SKIP+1}")
    plt.ylabel(f"PC{SKIP+2}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pca_folder, f"{folder}_terms_pca_annotated.png"))
    plt.close()
    print(
        f"Saved annotated plot: {os.path.join(pca_folder, f'{folder}_terms_pca_annotated.png')}"
    )

    plt.figure(figsize=(12, 6))
    plt.scatter(
        titles_pca[:, 0], titles_pca[:, 1], alpha=0.5, color="red", label="Titles"
    )
    for title, idx in sampled_titles.items():
        plt.text(titles_pca[idx, 0], titles_pca[idx, 1], title, fontsize=8)
    plt.title(f"PCA of Titles with Annotations - {folder.capitalize()}")
    plt.xlabel(f"PC{SKIP+1}")
    plt.ylabel(f"PC{SKIP+2}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pca_folder, f"{folder}_titles_pca_annotated.png"))
    plt.close()
    print(
        f"Saved annotated plot: {os.path.join(pca_folder, f'{folder}_titles_pca_annotated.png')}"
    )

print("PCA plotting completed for all folders.")
