from rdkit import Chem
from rdkit.Chem import MACCSkeys, DataStructs
import pandas as pd
import numpy as np

def calculate_tanimoto_similarity(fp1, fp2):
    """Calculates the Tanimoto similarity between two fingerprints using RDKit."""
    return DataStructs.FingerprintSimilarity(fp1, fp2)

def cluster_compounds(smiles_list, similarity_threshold=0.8):
    """
    Clusters compounds based on Tanimoto similarity.
    
    Args:
    - smiles_list: A list of SMILES strings representing the compounds.
    - similarity_threshold: The Tanimoto similarity threshold for clustering.
    
    Returns:
    - clusters: A list of clusters, each containing indices of similar compounds.
    """
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Ensure valid SMILES
            fingerprints.append(MACCSkeys.GenMACCSKeys(mol))
        else:
            fingerprints.append(None)  # Placeholder for invalid SMILES

    # Initialize pairwise Tanimoto similarity matrix
    similarity_matrix = np.zeros((len(fingerprints), len(fingerprints)))
    for i in range(len(fingerprints)):
        if fingerprints[i] is None:  # Skip invalid fingerprints
            continue
        for j in range(i + 1, len(fingerprints)):
            if fingerprints[j] is not None:  # Skip invalid fingerprints
                similarity = calculate_tanimoto_similarity(fingerprints[i], fingerprints[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    # Cluster based on similarity threshold
    clusters = []
    for i in range(len(fingerprints)):
        if fingerprints[i] is None:  # Skip invalid fingerprints
            continue
        found_cluster = False
        for cluster in clusters:
            if any(similarity_matrix[i, j] >= similarity_threshold for j in cluster):
                cluster.append(i)
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([i])

    return clusters

def calculate_modi(smiles_list, similarity_threshold=0.8):
    """
    Calculates the Modelability Index (MODI) for a dataset.
    
    Args:
    - smiles_list: A list of SMILES strings representing the compounds.
    - similarity_threshold: The Tanimoto similarity threshold for clustering.
    
    Returns:
    - modi: The Modelability Index for the dataset.
    """
    clusters = cluster_compounds(smiles_list, similarity_threshold)

    total_compounds = len(smiles_list)
    pure_compounds = 0

    # Check if all compounds in the cluster are considered similar
    for cluster in clusters:
        if len(cluster) > 1:  # Pure clusters have more than one member
            pure_compounds += len(cluster)

    modi = pure_compounds / total_compounds if total_compounds > 0 else 0
    return modi

def classify_compound(ic50):
    """
    Classifies a compound as 'Active', 'Semi-active', or 'Inactive' based on IC50 value.
    
    Args:
    - ic50: The IC50 value of the compound.
    
    Returns:
    - str: 'Active' if IC50 < 0.5 μM, 'Semi-active' if 0.5 μM ≤ IC50 ≤ 10 μM, 'Inactive' if IC50 > 10 μM.
    """
    try:
        if ic50 < 0.5:
            return 'Active'
        elif 0.5 <= ic50 <= 10:
            return 'Semi-active'
        else:
            return 'Inactive'
    except TypeError:
        return 'Unknown'  # Handle cases where IC50 is invalid

def predict_activity_based_on_modi(modi, cutoff=0.65):
    """
    Predicts if a dataset is likely to be active or inactive based on MODI.
    
    Args:
    - modi: The Modelability Index of the dataset.
    - cutoff: The cutoff value for determining activity.
    
    Returns:
    - str: 'Active' if the MODI is greater than the cutoff, 'Inactive' otherwise.
    """
    return 'Active' if modi > cutoff else 'Inactive'

# Read SMILES and IC50 from CSV file
file_path = 'compounds_with_predictions2.csv'  # Change to your file path
df = pd.read_csv(file_path)

# Extract SMILES and IC50 lists
smiles_list = df['smiles'].tolist()
ic50_list = df['IC50'].tolist()

# Classify compounds based on IC50 values
df['Classified Activity'] = [classify_compound(ic50) for ic50 in ic50_list]

# Calculate MODI
modi = calculate_modi(smiles_list)
print(f"MODI: {modi:.4f}")

# Predict activity based on MODI
df['Predicted Activity'] = predict_activity_based_on_modi(modi)

# Save the modified DataFrame to a new CSV file
output_file_path = 'Active_inactive.csv'
df.to_csv(output_file_path, index=False)

print(f"Predictions saved to '{output_file_path}'")
