import os
import urllib.request
import zipfile
from tqdm import tqdm

WESAD_URL = "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
TARGET_DIR = "DataSet/WESAD"
ZIP_PATH = os.path.join(TARGET_DIR, "wesad.zip")

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def setup_wesad():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Check if already extracted
    if os.path.exists(os.path.join(TARGET_DIR, "WESAD")):
        print("WESAD already exists.")
        return
        
    print(f"Downloading WESAD from {WESAD_URL}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=WESAD_URL.split('/')[-1]) as t:
            urllib.request.urlretrieve(WESAD_URL, filename=ZIP_PATH, reporthook=t.update_to)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(TARGET_DIR)
            
        print("Extraction complete.")
        os.remove(ZIP_PATH)
    except Exception as e:
        print(f"Failed to automatically download WESAD: {e}")
        print("Please manually download WESAD:")
        print(f"1. Go to: {WESAD_URL}")
        print(f"2. Extract the contents into: {os.path.abspath(TARGET_DIR)}")
        print("You should end up with a directory structure like: DataSet/WESAD/WESAD/S2/S2.pkl")

if __name__ == "__main__":
    setup_wesad()
