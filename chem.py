import requests
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to fetch SMILES string from PubChem
def fetch_smiles(chemical_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/property/IsomericSMILES/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    else:
        return None

# Function to compute molecular properties
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
            "Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol)
        }
    return None

# Sample training data for ML model (Replace with real data)
data = {
    "Molecular Weight": [180.16, 342.3, 58.44, 132.16],
    "LogP": [1.19, -0.7, -3.0, 0.5],
    "Solubility Class": [1, 0, 1, 0]  # 1 = Soluble, 0 = Insoluble
}

df = pd.DataFrame(data)
X = df[["Molecular Weight", "LogP"]]
y = df["Solubility Class"]

# Train a simple ML model
model = LinearRegression()
model.fit(X, y)

def predict_properties(molecular_data):
    features = np.array([[molecular_data["Molecular Weight"], molecular_data["LogP"]]])
    solubility = model.predict(features)
    return "Soluble" if solubility[0] > 0.5 else "Insoluble"

def main():
    chemical_name = input("Enter the chemical name: ")
    smiles = fetch_smiles(chemical_name)
    
    if smiles:
        print(f"SMILES: {smiles}")
        properties = compute_properties(smiles)
        if properties:
            print("Computed Properties:")
            for key, value in properties.items():
                print(f"{key}: {value}")
            
            # Predict additional properties using ML model
            solubility_prediction = predict_properties(properties)
            print(f"Predicted Solubility Class: {solubility_prediction}")
        else:
            print("Could not compute molecular properties.")
    else:
        print("Could not fetch SMILES string from PubChem.")

if __name__ == "__main__":
    main()