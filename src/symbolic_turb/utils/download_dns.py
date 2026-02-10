import os
import urllib.request
import shutil
import time

def download_duct_database():
    base_save_dir = "./dataset/reference"
    
    # Base FTP URL from Vinuesa's KTH repository
    ftp_base = "ftp://ftp.mech.kth.se/pub/rvinuesa/DuctData"

    # Format: (Folder Name on Server, File Suffix)
    configurations = [
        ("AR_1_180",  "1_180"),
        ("AR_1_360",  "1_360"),
        ("AR_3_180",  "3_180"),
        ("AR_3_360",  "3_360"),
        ("AR_5_180",  "5_180"),
        ("AR_7_180",  "7_180"),
        ("AR_10_180", "10_180"),
        ("AR_14_180", "14_180")
    ]

    variables = ["U", "V", "W", "uu", "vv", "ww", "uv", "uw", "vw"]
    
    coords_ar1 = ["zcoord"] # AR 1 is square duct, so only zcoord is needed
    coords_general = ["zcoord", "ycoord"]

    print(f"Starting download to {base_save_dir}...\n")

    for folder, suffix in configurations:
        local_dir = os.path.join(base_save_dir, folder)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"Created directory: {local_dir}")
        current_vars = variables.copy()
        if "AR_1_" in folder:
            current_vars += coords_ar1
        else:
            current_vars += coords_general

        for var in current_vars:
            filename = f"{var}_{suffix}.prof.txt"
            url = f"{ftp_base}/{folder}/{filename}"
            save_path = os.path.join(local_dir, filename)
            if os.path.exists(save_path):
                print(f"  [Skipping] {filename} (already exists)")
                continue

            print(f"  [Downloading] {filename}...")
            
            try:
                with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            except Exception as e:
                print(f"  !! FAILED to download {url}")
                print(f"     Error: {e}")

    print("\nAll downloads complete.")

if __name__ == "__main__":
    download_duct_database()