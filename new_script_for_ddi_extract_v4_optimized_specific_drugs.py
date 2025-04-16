import sqlite3
from lxml import etree
import csv

# Initialize SQLite database
def initialize_db(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS drugs (
            drug_id TEXT PRIMARY KEY,
            drug_name TEXT,
            smiles TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            drug_id TEXT,
            interacting_drug_id TEXT,
            interacting_drug_name TEXT,
            interacting_drug_smiles TEXT,
            interaction_description TEXT,
            FOREIGN KEY (drug_id) REFERENCES drugs (drug_id)
        )
    ''')
    conn.commit()
    return conn

# Parse DrugBank XML and store data in SQLite
def parse_drugbank_xml(xml_file, conn, drug_list):
    cursor = conn.cursor()
    
    # Use recover=True to fix XML errors
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(xml_file, parser)
    root = tree.getroot()

    # Detect XML namespace
    namespace = {'db': root.tag.split("}")[0].strip("{")}
    drug_tag = f"{{{namespace['db']}}}drug"

    # Step 1: Pre-load all drugs into a dictionary (Fast Lookup)
    drug_dict = {}

    for drug in root.findall(drug_tag, namespace):
        drug_id = drug.find("db:drugbank-id", namespace).text
        drug_name = drug.find("db:name", namespace).text

        if drug_name in drug_list:  # Only store relevant drugs
            smiles_elem = drug.find("db:calculated-properties/db:property[db:kind='SMILES']/db:value", namespace)
            smiles = smiles_elem.text if smiles_elem is not None else "N/A"
            drug_dict[drug_id] = (drug_name, smiles)

    print(f"Loaded {len(drug_dict)} drugs for fast lookup")

    # Step 2: Extract and insert data into database (Batch Insert)
    drug_data = []
    interaction_data = []

    for drug_id, (drug_name, smiles) in drug_dict.items():
        drug_data.append((drug_id, drug_name, smiles))

        # Extract interactions
        drug = root.find(f".//db:drug[db:drugbank-id='{drug_id}']", namespace)
        if drug is not None:
            for interaction in drug.findall("db:drug-interactions/db:drug-interaction", namespace):
                interacting_drug_id_elem = interaction.find("db:drugbank-id", namespace)
                interacting_drug_name_elem = interaction.find("db:name", namespace)
                interaction_description_elem = interaction.find("db:description", namespace)

                interacting_drug_id = interacting_drug_id_elem.text if interacting_drug_id_elem is not None else "N/A"
                interacting_drug_name = interacting_drug_name_elem.text if interacting_drug_name_elem is not None else "N/A"
                interaction_description = interaction_description_elem.text if interaction_description_elem is not None else "N/A"

                # FAST SMILES LOOKUP
                interacting_drug_smiles = drug_dict.get(interacting_drug_id, ("N/A", "N/A"))[1]

                interaction_data.append((drug_id, interacting_drug_id, interacting_drug_name, interacting_drug_smiles, interaction_description))

    # Step 3: Batch Insert Data into SQLite
    cursor.executemany('''
        INSERT OR IGNORE INTO drugs (drug_id, drug_name, smiles)
        VALUES (?, ?, ?)
    ''', drug_data)

    cursor.executemany('''
        INSERT INTO interactions (drug_id, interacting_drug_id, interacting_drug_name, interacting_drug_smiles, interaction_description)
        VALUES (?, ?, ?, ?, ?)
    ''', interaction_data)

    conn.commit()
    print("Batch insert completed successfully!")

# Export interactions to CSV
def export_interactions_to_csv(db_file, output_csv_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT d.drug_id, d.drug_name, d.smiles,
               i.interacting_drug_id, i.interacting_drug_name, i.interacting_drug_smiles, i.interaction_description
        FROM drugs d
        JOIN interactions i ON d.drug_id = i.drug_id
    ''')

    results = cursor.fetchall()
    
    filtered_results = [row for row in results if row[5] != "N/A"]

    with open(output_csv_file, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Drug ID', 'Drug Name', 'Drug SMILES',
                         'Interacting Drug ID', 'Interacting Drug Name', 'Interacting Drug SMILES', 'Interaction Description'])
        writer.writerows(filtered_results)

    conn.close()
    
    print(f"Successfully saved {len(filtered_results)} rows (filtered N/A values) to {output_csv_file}")


# Paths
xml_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/database.xml"
db_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/drugbank.db"
output_csv_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/drug_interactions_with_smiles.csv"
drug_list_file = "D:/New folder/MEM B8/Waddah/MSc in Data Science/WLV Documents/Research project/Omid/dataset/drug_list.txt"

# Load Drug List from File
with open(drug_list_file, "r", encoding="utf-8") as f:
    drug_list = {line.strip() for line in f}

# Run script
conn = initialize_db(db_file)
parse_drugbank_xml(xml_file, conn, drug_list)
export_interactions_to_csv(db_file, output_csv_file)
conn.close()

print("Script completed successfully! The dataset has been extracted and saved.")
