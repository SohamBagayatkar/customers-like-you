
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MATRIX_DIR = OUTPUT_DIR / "matrix"


def load_customer_data():
    """Load training transactions, matrix, and customer IDs."""
    transactions = pd.read_csv(
        OUTPUT_DIR / "train_transactions.csv"
    )
    matrix = load_npz(
        MATRIX_DIR / "customer_product_matrix.npz"
    ).tocsr()
    with open(MATRIX_DIR / "customers.json", "r") as file:
        customers = json.load(file)
    customers = [str(customer_id) for customer_id in customers]
    if matrix.shape[0] != len(customers):
        raise ValueError(
            f"Matrix has {matrix.shape[0]} rows, "
            f"but customers.json contains {len(customers)} IDs."
        )
    return transactions, matrix, customers


def prepare_purchase_matrix(matrix):
    """Reduce quantity skew and emphasize distinctive products."""

    transformed_matrix = matrix.copy().astype(np.float64)
    # Reduce the influence of unusually large purchase quantities
    transformed_matrix.data = np.log1p(transformed_matrix.data)
    # Give more weight to products that distinguish customers
    tfidf = TfidfTransformer(
        use_idf=True,
        norm="l2"
    )
    transformed_matrix = tfidf.fit_transform(transformed_matrix)
    return transformed_matrix


def create_customer_projection(matrix):
    """Project customers into two dimensions for the map."""

    transformed_matrix = prepare_purchase_matrix(matrix)
    svd = TruncatedSVD(
        n_components=2,
        random_state=42
    )
    coordinates = svd.fit_transform(transformed_matrix)
    projection = pd.DataFrame({
        "Component 1": coordinates[:, 0],
        "Component 2": coordinates[:, 1]
    })
    return projection, svd


def cluster_customers(matrix, n_clusters=4, n_components=20):
    """Cluster customers using a higher-dimensional SVD representation."""

    transformed_matrix = prepare_purchase_matrix(matrix)
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )
    customer_factors = svd.fit_transform(transformed_matrix)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(customer_factors)

    return cluster_labels, svd, kmeans
