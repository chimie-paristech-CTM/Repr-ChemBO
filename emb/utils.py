"""PCA compression and fingerprint DataFrame helpers used by benchmarking.py."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from emb.dataset import BLANK_SUBSTANCE

def pca_by_variance_ratio(X, target_variance_ratio=0.98, max_components=None):
    """
    Perform PCA to achieve a target explained variance ratio.
    
    Parameters:
    -----------
    X : numpy.Tensor
        Input data of shape (n_samples, n_features)
    target_variance_ratio : float
        Target cumulative explained variance ratio (default: 0.98)
    max_components : int, optional
        Maximum number of components to consider (default: min(n_samples, n_features))
    
    Returns: fitted PCA object

    """
    n_samples, n_features = X.shape
    
    # Set max_components if not provided
    if max_components is None:
        max_components = min(n_samples, n_features)
    
    # First, fit PCA with all components to get variance ratios
    pca_full = PCA(n_components=max_components)
    pca_full.fit(X)
    
    # Get cumulative explained variance ratio
    variance_ratio = pca_full.explained_variance_ratio_#.detach().numpy()
    cumulative_variance = variance_ratio.cumsum()
    
    # Find number of components needed to reach target
    n_components = np.sum( (cumulative_variance < target_variance_ratio) ) + 1
    n_components = int(n_components)
    
    # Refit PCA with optimal number of components
    pca_optimal = PCA(n_components=n_components)
    pca_optimal.fit(X)
    
    actual_variance = cumulative_variance[n_components - 1]
    
    return pca_optimal, n_components, actual_variance


def apply_pca_to_group(group_features, target_variance=0.98):
    """Apply PCA separately to each group of features"""
    group_features = np.array(group_features)
    # Apply PCA if more than 1 feature
    if group_features.shape[1] > 1:
        unique_rows = np.unique(group_features, axis=0)
        pca, n_comp, var = pca_by_variance_ratio(unique_rows, target_variance)
        transformed = pca.transform(group_features)
    else:
        transformed = group_features
        pca = None 

    return transformed, pca

def custom_fingerprinter(smiles_dict, fingerprinter, PCA=False, target_variance=0.98, norm=None, substance_fp=None):
    """
    Generate HSF fingerprints for a dictionary of SMILES strings.
    
    Args:
        smiles_dict: Dictionary mapping names to SMILES strings
        fingerprinter: HSFFingerprint instance (REQUIRED - must be shared across all calls)
        substance_fp: generate substance fingerprints based on BayBE (e.g., "MORDRED")
    
    Returns:
        pandas DataFrame with molecule names as index and fingerprint dimensions as columns
        (columns with constant values across all molecules are removed)
    """
    # Identify blank and real candidates independently for this component.
    blank_names = [
        name
        for name, smiles in smiles_dict.items()
        if smiles == BLANK_SUBSTANCE
    ]

    valid_smiles_dict = {
        name: smiles
        for name, smiles in smiles_dict.items()
        if smiles != BLANK_SUBSTANCE
    }
    if not valid_smiles_dict:
        raise ValueError("A molecular parameter contains only blank candidates. Its fingerprint dimension cannot be inferred, so this parameter should normally be removed from molecular_params.")
    
    # Extract names and SMILES in consistent order
    names = list(valid_smiles_dict.keys())
    smiles_list = [valid_smiles_dict[name] for name in names]
    
    if substance_fp is not None and fingerprinter is None:
        # "MORDRED" from BayBE 
        if substance_fp not in ["MORDRED", "ECFP", "RDKIT2DDESCRIPTORS"]:
            raise ValueError(f"un-supported substance fingerprint {substance_fp}.")
        from baybe.parameters import SubstanceParameter
        sp = SubstanceParameter(
            name='_',
            data=valid_smiles_dict,
            encoding=substance_fp,
            decorrelate=False,
        )
        fingerprint_array = sp.comp_df.values
        names = list(sp.comp_df.index)
        
    else:
        # Generate hidden-space fingerprints for all molecules at once (expect a list)
        fingerprint_array = fingerprinter(smiles_list)
    
    if PCA:
        fingerprint_array, _ = apply_pca_to_group(fingerprint_array, target_variance=target_variance)

    # Normalize to [0, 1]
    if norm == 'global':
        min_all = fingerprint_array.min()
        max_all = fingerprint_array.max()
        div = max_all - min_all if (max_all - min_all) != 0 else 1e-12
        fingerprint_array = (fingerprint_array - min_all) / div
    elif norm == 'local':
        mmin = fingerprint_array.min(axis=0)
        div = fingerprint_array.max(axis=0) - mmin
        fingerprint_array = fingerprint_array - mmin
        fingerprint_array = fingerprint_array / div
    else:
        pass


    # EXPLICITLY CONVERT TO FLOAT64 for numerical stability in BayBE
    # This ensures consistent precision in GP computations, PCA, and standardization
    # fingerprint_array = fingerprint_array.astype(np.float64)
    
    # Create column names for each fingerprint dimension
    n_dims = fingerprint_array.shape[1]
    column_names = [f"HSF_{i}" for i in range(n_dims)]
    
    # Create DataFrame with molecule names as index
    fingerprints_df = pd.DataFrame(
        fingerprint_array,
        index=names,
        columns=column_names
    )
    
    # Remove columns with constant values (BayBE requires varying features)
    # A column is constant if its standard deviation is zero or very close to zero
    varying_columns = fingerprints_df.std() > 1e-10
    fingerprints_df = fingerprints_df.loc[:, varying_columns]
    
    # Add blank candidates only after all transformations.
    # This is a zero vector, broadcasting to all dims.
    for blank_name in blank_names:
        fingerprints_df.loc[blank_name] = 0.0
    
    return fingerprints_df
    