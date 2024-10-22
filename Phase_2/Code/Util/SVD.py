import numpy as np

def SVD(data, latent_count):

    # Step 1: Compute D^T D
    DtD = np.dot(data.T, data)

    # Step 2: Compute eigenvalues and eigenvectors of D^T D
    eigenvalues_V, V = np.linalg.eigh(DtD)

    # Step 3: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[sorted_indices]
    V = V[:, sorted_indices]
    
    V_subset = V[:, :latent_count]

    # TODO: implement k somewhere
    # Step 4: Compute singular values (square roots of eigenvalues of D^T D)
    singular_values = np.sqrt(eigenvalues_V)

    # Step 5: Compute D D^T
    DDt = np.dot(data, data.T)

    # Step 6: Compute eigenvalues and eigenvectors of D D^T
    eigenvalues_U, U = np.linalg.eigh(DDt)

    # Step 7: Sort eigenvalues and eigenvectors of U in descending order
    sorted_indices_U = np.argsort(eigenvalues_U)[::-1]
    U = U[:, sorted_indices_U]

    # Step 8: Construct the Sigma matrix
    Sigma = np.zeros_like(data, dtype=float)
    np.fill_diagonal(Sigma, singular_values)
    
    return U, Sigma, V.T


    eigenvectors_subset = sorted_eigenvectors[:, :latent_count]

    # Left factor matrix: projected data onto principal components
    left_matrix = np.dot(data, eigenvectors_subset)
    # Right factor matrix: Principal components (eigenvectors)
    right_matrix = eigenvectors_subset
    

    pca_data = np.dot(data, eigenvectors_subset)
    
    return left_matrix, right_matrix, sorted_eigenvalues[:latent_count]
