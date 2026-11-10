import os
import tempfile

from joblib import Memory
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Cache the merge tree so sweeping over k reuses one tree per dataset instead of
# rebuilding it for every k (results are identical, it is just much faster).
_tree_cache = Memory(os.path.join(tempfile.gettempdir(), "hac_tree_cache"), verbose=0)


def run_hac(X, k, linkage_method='ward'):
    X_input = X.toarray() if hasattr(X, 'toarray') else X
    hac = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage_method,
        memory=_tree_cache,
        compute_full_tree=True,
    )
    cluster_labels = hac.fit_predict(X_input)
    return cluster_labels
