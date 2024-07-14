import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

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

def get_top_correlations(entries, correlation_matrix, idx, n=5, threshold=0.3):
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

# Streamlit app
st.title("Interactive Pearson Correlation")

option = st.selectbox("Select type:", ["SMILES", "Gene"])

if option == "SMILES":
    entry = st.selectbox("Select a SMILES:", compounds['ID'].tolist())
    idx = compounds[compounds['ID'] == entry].index[0]
    
    top_positive, top_negative, top_positive_values, top_negative_values = get_top_correlations(compounds, compounds_pearson, idx)
    top_positive_genes, top_negative_genes, top_positive_genes_values, top_negative_genes_values = get_top_correlations(genes, compounds_to_genes_pearson, idx)
    
    st.write(f"Selected SMILES: {entry}")
    
    st.write("Top Positively Correlated Compounds:")
    if top_positive.empty:
        st.write("None")
    else:
        for entry, value in zip(top_positive['ID'], top_positive_values):
            st.write(f"{entry}: {value:.4f}")
    
    st.write("Top Negatively Correlated Compounds:")
    if top_negative.empty:
        st.write("None")
    else:
        for entry, value in zip(top_negative['ID'], top_negative_values):
            st.write(f"{entry}: {value:.4f}")
    
    st.write("Top Positively Correlated Genes:")
    if top_positive_genes.empty:
        st.write("None")
    else:
        for entry, value in zip(top_positive_genes['ID'], top_positive_genes_values):
            st.write(f"{entry}: {value:.4f}")
    
    st.write("Top Negatively Correlated Genes:")
    if top_negative_genes.empty:
        st.write("None")
    else:
        for entry, value in zip(top_negative_genes['ID'], top_negative_genes_values):
            st.write(f"{entry}: {value:.4f}")

elif option == "Gene":
    entry = st.selectbox("Select a Gene:", genes['ID'].tolist())
    idx = genes[genes['ID'] == entry].index[0]
    
    top_positive, top_negative, top_positive_values, top_negative_values = get_top_correlations(genes, genes_pearson, idx)
    top_positive_compounds, top_negative_compounds, top_positive_compounds_values, top_negative_compounds_values = get_top_correlations(compounds, genes_to_compounds_pearson, idx)
    
    st.write(f"Selected Gene: {entry}")
    
    st.write("Top Positively Correlated Genes:")
    if top_positive.empty:
        st.write("None")
    else:
        for entry, value in zip(top_positive['ID'], top_positive_values):
            st.write(f"{entry}: {value:.4f}")
    
    st.write("Top Negatively Correlated Genes:")
    if top_negative.empty:
        st.write("None")
    else:
        for entry, value in zip(top_negative['ID'], top_negative_values):
            st.write(f"{entry}: {value:.4f}")
    
    st.write("Top Positively Correlated Compounds:")
    if top_positive_compounds.empty:
        st.write("None")
    else:
        for entry, value in zip(top_positive_compounds['ID'], top_positive_compounds_values):
            st.write(f"{entry}: {value:.4f}")
    
    st.write("Top Negatively Correlated Compounds:")
    if top_negative_compounds.empty:
        st.write("None")
    else:
        for entry, value in zip(top_negative_compounds['ID'], top_negative_compounds_values):
            st.write(f"{entry}: {value:.4f}")

