"""
Model Downloader - Downloads large model files from Google Drive
=================================================================
Upload your models to Google Drive and use their direct download links
"""

import os
import requests
from pathlib import Path
import re


def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive using file ID
    Handles large files with virus scan warning by parsing UUID
    
    Args:
        file_id: The Google Drive file ID (from shareable link)
        destination: Local path where to save the file
    """
    print(f"   Connecting to Google Drive...")
    
    session = requests.Session()
    
    # First request to get the download page (may have virus warning)
    URL = "https://drive.google.com/uc?export=download"
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check if we got HTML (virus warning page)
    content_type = response.headers.get('Content-Type', '')
    if 'text/html' in content_type:
        print(f"   Large file detected, extracting download token...")
        
        # Get the HTML content
        html_content = response.text
        
        # Extract UUID from the HTML using regex
        uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html_content)
        confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html_content)
        
        if uuid_match and confirm_match:
            uuid = uuid_match.group(1)
            confirm = confirm_match.group(1)
            
            # Make the actual download request with UUID
            download_url = "https://drive.usercontent.google.com/download"
            params = {
                'id': file_id,
                'export': 'download',
                'confirm': confirm,
                'uuid': uuid
            }
            
            print(f"   Starting download with confirmation...")
            response = session.get(download_url, params=params, stream=True)
        else:
            print(f"   Warning: Could not extract download token, trying anyway...")
    
    # Download the file
    print(f"   Downloading...")
    CHUNK_SIZE = 32768
    total_size = 0
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
                # Show progress every 10MB
                if total_size % (10 * 1024 * 1024) < CHUNK_SIZE:
                    print(f"   Downloaded {total_size / (1024*1024):.1f} MB...")
    
    final_size_mb = total_size / (1024*1024)
    print(f"   Total: {final_size_mb:.1f} MB")
    
    # Verify we didn't just download an error page
    if total_size < 10000:  # Less than 10KB is suspicious
        with open(destination, 'r') as f:
            content = f.read(500)
            if 'html' in content.lower() or 'error' in content.lower():
                raise Exception(f"Download failed - received HTML error page instead of file")


def download_models():
    """
    Download model files if they don't exist locally
    Configure your Google Drive file IDs below
    """
    
    # ============ CONFIGURED GOOGLE DRIVE FILE IDs ============
    # Links provided:
    # 1. https://drive.google.com/file/d/1aeg0abLtqcNx7Uwv0U7Ya507dxS1a7Fc/view?usp=sharing
    # 2. https://drive.google.com/file/d/1aPtbKn9i-CVrA7eWdvh_JLkTrQ8RIBSe/view?usp=sharing
    # 3. https://drive.google.com/file/d/1VlB-u-16g54_on-6_iOAA0HJafWtBNJY/view?usp=sharing
    
    MODELS = {
        'yolo11x_leaf.pt': {
            'file_id': '1aeg0abLtqcNx7Uwv0U7Ya507dxS1a7Fc',
            'size_mb': 200  # Approximate size for progress
        },
        'plant_disease_model_optimized.h5': {
            'file_id': '1aPtbKn9i-CVrA7eWdvh_JLkTrQ8RIBSe',
            'size_mb': 30
        },
        'plant_disease_model_optimized.keras': {
            'file_id': '1VlB-u-16g54_on-6_iOAA0HJafWtBNJY',
            'size_mb': 30
        }
    }
    
    # ====================================================================
    
    print("🔍 Checking for model files...")
    
    for model_name, config in MODELS.items():
        model_path = Path(model_name)
        
        if model_path.exists():
            print(f"✅ {model_name} already exists")
            continue
        
        file_id = config['file_id']
        
        # Skip if file ID not configured
        if file_id.startswith('YOUR_'):
            print(f"⚠️  {model_name} - File ID not configured, skipping download")
            continue
        
        print(f"📥 Downloading {model_name} (~{config['size_mb']}MB)...")
        try:
            download_file_from_google_drive(file_id, str(model_path))
            print(f"✅ {model_name} downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            raise
    
    print("✅ All models are ready!")


if __name__ == "__main__":
    download_models()
