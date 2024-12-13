import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import List, Tuple
from sklearn.datasets import fetch_olivetti_faces


def load_images(
    folder_path: str = None, image_size: Tuple[int, int] = (64, 64)
) -> np.ndarray:
    """Load images from Olivetti faces dataset."""
    dataset = fetch_olivetti_faces(shuffle=True)
    return dataset.data


def compute_eigenfaces(
    X: np.ndarray, n_components: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenfaces using the Turk and Pentland method."""
    # Number of images (M) and dimensionality of each image (N)
    M, N = X.shape

    # Center the data by subtracting the mean face
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face

    # Compute the covariance matrix in the reduced space (M x M)
    # L = X_centered * X_centered.T
    L = np.dot(X_centered, X_centered.T)

    # Compute eigenvalues and eigenvectors of L
    # Use 'eigh' since L is symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # If n_components is specified, truncate
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    # Compute the actual eigenfaces by projecting eigenvectors of L back to the original N-dim space
    eigenfaces = np.dot(X_centered.T, eigenvectors)

    # Normalize eigenfaces (each eigenface should be unit length)
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i]) + 1e-15

    # Transpose so that eigenfaces are rows instead of columns
    eigenfaces = eigenfaces.T

    # The singular values (for explained variance) can be approximated from eigenvalues
    # eigenvalues correspond to variance in the directions of eigenfaces
    singular_values = np.sqrt(eigenvalues)

    return eigenfaces, mean_face, singular_values


def project_image(
    image: np.ndarray, eigenfaces: np.ndarray, mean_face: np.ndarray
) -> np.ndarray:
    """Project an image onto the eigenface space."""
    image_centered = image - mean_face
    # eigenfaces shape: (n_components, N), image_centered shape: (N,)
    weights = np.dot(eigenfaces, image_centered)
    return weights


def reconstruct_image(
    weights: np.ndarray, eigenfaces: np.ndarray, mean_face: np.ndarray
) -> np.ndarray:
    """Reconstruct an image from its eigenface weights."""
    # weights shape: (n_components,), eigenfaces shape: (n_components, N)
    reconstruction = np.dot(weights, eigenfaces) + mean_face
    return reconstruction


def plot_eigenfaces(
    eigenfaces: np.ndarray, image_size: Tuple[int, int], n_components: int = 10
):
    """Plot the top eigenfaces."""
    n_plot = min(n_components, eigenfaces.shape[0])
    fig, axes = plt.subplots(2, n_plot // 2, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(n_plot):
        eigenface = eigenfaces[i].reshape(image_size)
        axes[i].imshow(eigenface, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Eigenface {i+1}")

    plt.tight_layout()
    plt.savefig("eigenfaces.png")
    plt.close()


if __name__ == "__main__":
    # Example usage
    IMAGE_SIZE = (64, 64)  # Olivetti faces are 64x64
    N_COMPONENTS = 128

    # Load images
    print("Loading images...")
    X = load_images()

    # Compute eigenfaces using Turk & Pentland method
    print("Computing eigenfaces...")
    eigenfaces, mean_face, singular_values = compute_eigenfaces(X, N_COMPONENTS)

    # Plot eigenfaces
    print("Plotting eigenfaces...")
    plot_eigenfaces(eigenfaces, IMAGE_SIZE, n_components=10)

    # Example reconstruction
    if len(X) > 0:
        # Take first image as example
        test_image = X[0]

        # Project and reconstruct
        weights = project_image(test_image, eigenfaces, mean_face)
        reconstruction = reconstruct_image(weights, eigenfaces, mean_face)

        # Plot original vs reconstruction
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(test_image.reshape(IMAGE_SIZE), cmap="gray")
        ax1.set_title("Original")
        ax1.axis("off")

        ax2.imshow(reconstruction.reshape(IMAGE_SIZE), cmap="gray")
        ax2.set_title("Reconstruction")
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig("reconstruction_example.png")
        plt.close()

        # Print explained variance ratio
        explained_variance_ratio = singular_values**2 / np.sum(singular_values**2)
        print(
            f"Explained variance ratio of first {N_COMPONENTS} components: "
            f"{np.sum(explained_variance_ratio):.3f}"
        )
