import numpy as np

def PCA(data, latent_count):

    # We use covariance matrix for PCA
    cov_matrix = np.cov(data, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    eigenvalues_subset = sorted_eigenvalues[:latent_count]
    eigenvectors_subset = sorted_eigenvectors[:, :latent_count]

    # Left factor matrix: projected data onto principal components
    left_matrix = np.dot(data, eigenvectors_subset)
    # Right factor matrix: Principal components (eigenvectors)
    right_matrix = eigenvectors_subset
    
    pca_data = np.dot(data, eigenvectors_subset)

    print(f"Top-{latent_count} latent Semantics for PCA")
    for index, eigenvalue in enumerate(eigenvalues_subset):
        print(f"{index} - {eigenvalue}")
    
    return left_matrix, right_matrix, sorted_eigenvalues[:latent_count]
