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

    # Step 4: Compute singular values (square roots of eigenvalues of D^T D)
    singular_values = np.sqrt(eigenvalues_V)

    # Step 5: Compute D D^T
    DDt = np.dot(data, data.T)

    # Step 6: Compute eigenvalues and eigenvectors of D D^T
    eigenvalues_U, U = np.linalg.eigh(DDt)

    # Step 7: Sort eigenvalues and eigenvectors of U in descending order
    sorted_indices_U = np.argsort(eigenvalues_U)[::-1]
    eigenvalues_U = eigenvalues_U[sorted_indices_U]
    U = U[:, sorted_indices_U]

    U_subset = U[:, :latent_count]

    # Step 8: Construct the Sigma matrix
    Sigma = np.zeros_like(data, dtype=float)
    np.fill_diagonal(Sigma, singular_values[:latent_count])

    print(f"Top-{latent_count} latent Semantics for SVD")
    for index, eigenvalue in enumerate(singular_values[:latent_count]):
        print(f"{index} - {eigenvalue}")
    
    # print("U_k Matrix:")
    # print(U_subset)
    # print("\nSigma_k Matrix:")
    # print(Sigma)
    # print("\nVt_k Matrix:")
    # print(V_subset.T)

    return U_subset, Sigma, V_subset.T
