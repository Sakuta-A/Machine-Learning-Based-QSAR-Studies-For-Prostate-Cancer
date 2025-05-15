import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Load Ki prediction data
ki_df = pd.read_csv("molecule_activity_table.csv")

# Load molecules from SDF
supplier = Chem.SDMolSupplier("MeFSAT_3D_Structures.sdf")
mols = [mol for mol in supplier if mol is not None]

# Create a mapping from index to molecule object
mol_dict = {i+1: mol for i, mol in enumerate(mols)}  # Assuming Molecule IDs start from 1

# Map Molecule ID to RDKit Mol object
ki_df["Mol"] = ki_df["Molecule"].map(mol_dict)

# Drop unmatched molecules
ki_df = ki_df.dropna(subset=["Mol"])

# Get top 10 molecules with lowest Ki values
top10 = ki_df.nsmallest(10, "Pred Ki")

# Generate molecule image grid
img = Draw.MolsToGridImage(
    top10["Mol"].tolist(),
    legends=[f"Ki: {ki:.2f} nM" for ki in top10["Pred Ki"]],
    molsPerRow=5,
    subImgSize=(200, 200)
)

# Show image (optional)
img.show()

# Save image to file
img.save("top10_lowest_ki.png")

# Add SMILES strings and export to CSV
top10["SMILES"] = top10["Mol"].apply(Chem.MolToSmiles)
top10[["Molecule", "Pred Ki", "Confidence", "Pred Scale", "SMILES"]].to_csv("top10_lowest_ki.csv", index=False)

# Confirm successful saving of files
print("Saved top10_lowest_ki.png and top10_lowest_ki.csv successfully.")