import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

try:
    from rdkit import Chem
    from rdkit import RDPaths
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.Draw import MolDraw2DSVG
except ImportError:
    st.error("RDKit library is not installed. Please install it using 'pip install rdkit-pypi'.")

# Streamlit app
st.title("PhenAR")

st.set_page_config(
    page_title="PhenAR",
    page_icon="Logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

left_col, right_col = st.columns(2)

right_col.write("# Welcome to PhenAR")
right_col.write("v4.0")
right_col.write("Created by Srijit Seal, Shantanu Singh, and Anne Carpenter")
left_col.image("Logo.png")

# Load the dataset
data = pd.read_csv('Dummy_Dataset_of_Compounds_and_Genes.csv')

# Separate compounds and genes
compounds = data[data['Type'] == 'C']
genes = data[data['Type'] == 'G']

# Calculate Pearson correlation
feature_columns = data.columns[2:]  # Assuming features start from the third column
compounds_features = compounds[feature_columns]
genes_features = genes[feature_columns]

def calculate_pearson_matrix(df1, df2):
    correlation_matrix = np.zeros((df1.shape[0], df2.shape[0]))
    for i in range(df1.shape[0]):
        for j in range(df2.shape[0]):
            correlation_matrix[i, j] = pearsonr(df1.iloc[i], df2.iloc[j])[0]
    return correlation_matrix

compounds_pearson = calculate_pearson_matrix(compounds_features, compounds_features)
genes_pearson = calculate_pearson_matrix(genes_features, genes_features)
compounds_to_genes_pearson = calculate_pearson_matrix(compounds_features, genes_features)
genes_to_compounds_pearson = calculate_pearson_matrix(genes_features, compounds_features)

def get_top_correlations(entries, correlation_matrix, idx, n=6, threshold=0.3):
    if idx >= correlation_matrix.shape[0]:
        return pd.DataFrame(), pd.DataFrame(), [], []
    
    correlations = correlation_matrix[idx]
    
    positive_mask = (correlations > threshold) & (np.arange(len(correlations)) != idx)
    negative_mask = (correlations < -threshold) & (np.arange(len(correlations)) != idx)
    
    top_positive_indices = np.where(positive_mask)[0]
    top_negative_indices = np.where(negative_mask)[0]
    
    top_positive_values = correlations[top_positive_indices]
    top_negative_values = correlations[top_negative_indices]
    
    sorted_positive_indices = top_positive_indices[np.argsort(top_positive_values)[::-1]][:n]
    sorted_negative_indices = top_negative_indices[np.argsort(top_negative_values)][:n]
    
    top_positive_entries = entries.iloc[sorted_positive_indices]
    top_negative_entries = entries.iloc[sorted_negative_indices]
    
    top_positive_values = top_positive_values[np.argsort(top_positive_values)[::-1]][:n]
    top_negative_values = top_negative_values[np.argsort(top_negative_values)][:n]
    
    return top_positive_entries, top_negative_entries, top_positive_values, top_negative_values

def display_results(entries, values, entry_type, display_images=False):
    if entries.empty:
        st.write("None")
    else:
        entry_column = 'ID'
        num_entries = len(entries)
        num_rows = (num_entries + 2) // 3
        cols = st.columns(3)
        for i in range(num_entries):
            col = cols[i % 3]
            with col:
                if display_images and entry_type == 'compound':
                    display_smiles_structure(entries.iloc[i][entry_column], values[i])
                elif entry_type == 'gene':
                    st.write(f"[**{entries.iloc[i][entry_column]}**: {values[i]:.4f}](https://www.ncbi.nlm.nih.gov/gene/?term={entries.iloc[i][entry_column]})")

def display_smiles_structure(smiles, correlation=None):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            caption = f"{smiles}" if correlation is None else f"{smiles} ({correlation:.4f})"
            img = Draw.MolToImage(mol)
            st.image(img, width=200, caption=caption)
        else:
            st.write("Invalid SMILES string")
    except Exception as e:
        st.write(f"Error in drawing SMILES structure: {e}")

input_type = st.selectbox("Select input type:", ["compound", "gene"])
output_type = st.selectbox("Select output type:", ["compound", "gene"])

if input_type == "compound" and output_type == "compound":
    st.subheader("Compound-Compound Correlation")
    input_method = st.radio("Choose input method for the compound:", ["Type SMILES", "Select from list"])
    
    if input_method == "Type SMILES":
        entry = st.text_input("Enter SMILES:")
    else:
        entry = st.selectbox("Select a SMILES:", compounds['ID'].tolist())
    
    if entry and entry in compounds['ID'].values:
        idx = compounds[compounds['ID'] == entry].index[0]
        
        st.write(f"Selected SMILES:")
        display_smiles_structure(entry)
        
        top_positive, top_negative, top_positive_values, top_negative_values = get_top_correlations(compounds, compounds_pearson, idx)
        st.write("Top Positively Correlated Compounds:")
        display_results(top_positive, top_positive_values, 'compound', display_images=True)
        st.write("Top Negatively Correlated Compounds:")
        display_results(top_negative, top_negative_values, 'compound', display_images=True)

elif input_type == "gene" and output_type == "compound":
    st.subheader("Gene-Compound Correlation")
    st.write("First, enter the gene. Then, the correlated compounds will be shown.")
    input_method = st.radio("Choose input method for the gene:", ["Type Gene", "Select from list"])
    
    if input_method == "Type Gene":
        entry = st.text_input("Enter Gene:")
    else:
        entry = st.selectbox("Select a Gene:", genes['ID'].tolist())
    
    if entry and entry in genes['ID'].values:
        idx = genes[genes['ID'] == entry].index[0]
        pos_idx = genes.index.get_loc(idx)
        
        st.write(f"Selected Gene: {entry}")
        
        if pos_idx < genes_pearson.shape[0]:
            top_positive_compounds, top_negative_compounds, top_positive_compounds_values, top_negative_compounds_values = get_top_correlations(compounds, genes_to_compounds_pearson, pos_idx)
            st.write("Top Positively Correlated Compounds:")
            display_results(top_positive_compounds, top_positive_compounds_values, 'compound', display_images=True)
            st.write("Top Negatively Correlated Compounds:")
            display_results(top_negative_compounds, top_negative_compounds_values, 'compound', display_images=True)
        else:
            st.write("Error: Selected index is out of bounds for genes Pearson correlation matrix.")
    else:
        st.write("Invalid Gene entered. Please select from the list or enter a valid Gene.")

elif input_type == "compound" and output_type == "gene":
    st.subheader("Compound-Gene Correlation")
    st.write("First, enter the compound. Then, the correlated genes will be shown.")
    input_method = st.radio("Choose input method for the compound:", ["Type SMILES", "Select from list"])
    
    if input_method == "Type SMILES":
        entry = st.text_input("Enter SMILES:")
    else:
        entry = st.selectbox("Select a SMILES:", compounds['ID'].tolist())
    
    if entry and entry in compounds['ID'].values:
        idx = compounds[compounds['ID'] == entry].index[0]
        
        st.write(f"Selected SMILES:")
        display_smiles_structure(entry)
        
        top_positive_genes, top_negative_genes, top_positive_genes_values, top_negative_genes_values = get_top_correlations(genes, compounds_to_genes_pearson, idx)
        st.write("Top Positively Correlated Genes:")
        display_results(top_positive_genes, top_positive_genes_values, 'gene')
        st.write("Top Negatively Correlated Genes:")
        display_results(top_negative_genes, top_negative_genes_values, 'gene')
    else:
        st.write("Invalid SMILES entered. Please select from the list or enter a valid SMILES.")

elif input_type == "gene" and output_type == "gene":
    st.subheader("Gene-Gene Correlation")
    st.write("First, enter the gene. Then, the correlated genes will be shown.")
    input_method = st.radio("Choose input method for the gene:", ["Type Gene", "Select from list"])
    
    if input_method == "Type Gene":
        entry = st.text_input("Enter Gene:")
    else:
        entry = st.selectbox("Select a Gene:", genes['ID'].tolist())
    
    if entry and entry in genes['ID'].values:
        idx = genes[genes['ID'] == entry].index[0]
        pos_idx = genes.index.get_loc(idx)
        
        st.write(f"Selected Gene: {entry}")
        
        if pos_idx < genes_pearson.shape[0]:
            top_positive, top_negative, top_positive_values, top_negative_values = get_top_correlations(genes, genes_pearson, pos_idx)
            st.write("Top Positively Correlated Genes:")
            display_results(top_positive, top_positive_values, 'gene')
            st.write("Top Negatively Correlated Genes:")
            display_results(top_negative, top_negative_values, 'gene')
        else:
            st.write("Error: Selected index is out of bounds for genes Pearson correlation matrix.")
    else:
        st.write("Invalid Gene entered. Please select from the list or enter a valid Gene.")
