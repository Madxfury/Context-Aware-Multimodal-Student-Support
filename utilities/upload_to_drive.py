from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

# ✅ Path to your folder correctly updated
LOCAL_FOLDER = os.path.join(os.getcwd(), "utilities", "realtime_results")

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.json")

if not gauth.credentials:
    print("⚠️ No credentials found. Opening browser for login...")
    gauth.LocalWebserverAuth()
    gauth.SaveCredentialsFile("mycreds.json")
else:
    print("✅ Google Drive authentication loaded.")

drive = GoogleDrive(gauth)

def upload_file(file_path):
    file_name = os.path.basename(file_path)
    print(f"⬆ Uploading: {file_name}")
    gfile = drive.CreateFile({'title': file_name})
    gfile.SetContentFile(file_path)
    gfile.Upload()
    print(f"✅ Uploaded: {file_name}")

if not os.path.exists(LOCAL_FOLDER):
    print("❌ Error: Folder not found ->", LOCAL_FOLDER)
else:
    has_files = False
    for root, _, files in os.walk(LOCAL_FOLDER):
        for file in files:
            has_files = True
            upload_file(os.path.join(root, file))

    if not has_files:
        print("⚠️ No files found to upload.")
    else:
        print("🎉 All files uploaded successfully!")
