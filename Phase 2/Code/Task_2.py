from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
import numpy as np

Feature_model=input("Input the feature model:")
latent_count=input("Input the number of latent semantics:")
method=input("input 1 for PCA,2 for SVD,3 for LDA, 4 for Kmeans:")

#we need to write the code for PCA and SVD


def lda(features, labels, s):
    lda = LDA(n_components=s)
    transformed_features = lda.fit_transform(features, labels)
    return transformed_features, lda.scalings_

def kmeans(features, s):
    kmeans = KMeans(n_clusters=s)
    kmeans.fit(features)
    return kmeans.cluster_centers_, kmeans.labels_

if method == 1:
    print(f"Applying PCA with top-{s} components...")
    transformed, components = pca(features, s)
elif method == 2:
    print(f"Applying SVD with top-{s} components...")
    transformed, components = svd(features, s)
elif method == 3:
    print(f"Applying LDA with top-{s} components...")
    # Assuming we have labels for LDA (for example, frame categories, replace with actual labels)
    labels = np.random.randint(0, 3, features.shape[0])  # 3 random categories for demo
    transformed, components = lda(features, labels, s)
elif method == 4:
    print(f"Applying K-means with {s} clusters...")
    centers, labels = kmeans(features, s)
    transformed = centers
    components = None
else:
    print("Invalid method selected.")