import os
import json
import shutil
import re

DATA_DIR = r"D:\Project\KhmerCurrencyOCR\data\numista_raw"
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

# Definitive modern circulating designs for Cambodia (KHR) based on circulation scope doc
TARGET_MODERN_COMMON = {
    "500": ["2015"],
    "1000": ["2013", "2017"],
    "2000": ["2013", "2022"],
    "5000": ["2017"],
    "10000": ["2015"],
    "20000": ["2018"],
    "50000": ["2014"]
}

TARGET_MODERN_RARE = {
    "50": ["2002"], # Circulating 50 Riels
    "100": ["2001", "2015"],
    "200": ["1995", "1998", "2022"],
    "15000": ["2019"],
    "30000": ["2021"],
    "100000": ["2013"],
    "200000": ["2024"]
}

def determine_circulation(country, denom, year_str):
    if country == "United States":
        # We only downloaded the circulating USD bills (1, 2, 5, 10, 20, 50, 100)
        return "in_circulation"
        
    # Cambodian Note logic - strictly limit circulating notes to year 2000+
    years = re.findall(r'\d{4}', str(year_str))
    if not years:
        return "out_of_circulation"
        
    for y in years:
        if int(y) >= 2000:
            return "in_circulation"
            
    return "out_of_circulation"


def organize():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: metadata.json does not exist at {METADATA_PATH}. Cannot organize.")
        return

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Loaded metadata.json containing {len(metadata)} items. Restructuring dataset...")
    
    new_metadata = {}
    
    moved_count = 0
    total_assets = 0

    for note_id, note_data in metadata.items():
        country = note_data.get("country", "Cambodia")
        country_dir = "cambodia" if country == "Cambodia" else "usa"
        
        denom = note_data.get("denomination", "unknown")
        years = note_data.get("years", "ND")
        
        circulation_status = determine_circulation(country, denom, years)
        
        # Clean title for directory name
        clean_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', note_data["title"])
        clean_title = re.sub(r'_+', '_', clean_title).strip('_')
        
        # Folder structure: data/numista_raw/<country>/<circulation_status>/<denom_year_id_title>/
        note_folder_name = f"{denom}_{years}_{note_id}_{clean_title}"
        note_folder_path = os.path.join(DATA_DIR, country_dir, circulation_status, note_folder_name)
        os.makedirs(note_folder_path, exist_ok=True)
        
        print(f"Organizing Note ID {note_id} ({note_data['title']}) -> {circulation_status}")
        
        old_files = note_data.get("files", {})
        new_files = {}
        
        # Source flat directory
        src_dir = os.path.join(DATA_DIR, country_dir)
        
        # Move standard files
        for asset_type, filename in old_files.items():
            if asset_type == "signatures" and isinstance(filename, list):
                new_sigs = []
                for sig_file in filename:
                    src_file = os.path.join(src_dir, sig_file)
                    if os.path.exists(src_file):
                        dest_file = os.path.join(note_folder_path, sig_file)
                        shutil.move(src_file, dest_file)
                        new_sigs.append(os.path.join(country_dir, circulation_status, note_folder_name, sig_file))
                        moved_count += 1
                        total_assets += 1
                new_files["signatures"] = new_sigs
            else:
                src_file = os.path.join(src_dir, filename)
                if os.path.exists(src_file):
                    dest_file = os.path.join(note_folder_path, filename)
                    shutil.move(src_file, dest_file)
                    new_files[asset_type] = os.path.join(country_dir, circulation_status, note_folder_name, filename)
                    moved_count += 1
                    total_assets += 1
                    
        # Update metadata paths
        new_metadata[note_id] = {
            "title": note_data["title"],
            "country": country,
            "circulation_status": circulation_status,
            "denomination": denom,
            "currency": note_data.get("currency", ""),
            "issuer": note_data.get("issuer", ""),
            "issuing_bank": note_data.get("issuing_bank", ""),
            "years": years,
            "size": note_data.get("size", ""),
            "features": note_data.get("features", {}),
            "directory": os.path.join(country_dir, circulation_status, note_folder_name),
            "files": new_files
        }

    # Save restructured metadata
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\nRestructuring Complete!")
    print(f"Created isolated folders for each of the {len(new_metadata)} notes.")
    print(f"Successfully moved {moved_count} assets into 'in_circulation' and 'out_of_circulation' directories.")
    
if __name__ == "__main__":
    organize()
