# Calculate_PBN_Results
Project to calculate bridge game statistics from a PBN file. Contains a standalone example notebook which creates dataframes from pbn files. Dataframes are augmented with par, double dummy (DD), single dummy (SD) probabilities, expected values (Exp), and best contracts (max expected value contract). Compatible with jupyter and vscode notebooks. Minimal documentation and support provided. Assumes programmer who is familiar with the game of bridge, github, jupyter/vscode notebook and python.

# Overview
1. Read a pbn file (local file).
2. Create a df from pbn file.
3. Augment df with par, double dummy, single dummy probabilities, expected values, best contract (max expected value contract).
4. Do some simple explorations of the augmented df.

# Installation:
1. git clone https://github.com/BSalita/Calculate_PBN_Results
2. pip install -U -r requirements.txt
3. streamlit run CalculatePBNResults_Streamlit.py

# Dependencies:
See requirements.txt
