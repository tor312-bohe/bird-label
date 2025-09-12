#!/usr/bin/env python3
"""
Download Bird Watch model files for enhanced bird recognition.
This script helps download the TensorFlow model from the official Bird Watch repository.
"""

import os
import requests
import json
from pathlib import Path

def download_file(url, local_path):
    """Download a file from URL to local path"""
    print(f"📥 Downloading {os.path.basename(local_path)}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded {os.path.basename(local_path)}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {os.path.basename(local_path)}: {e}")
        return False

def get_latest_release():
    """Get the latest release information from GitHub API"""
    try:
        api_url = "https://api.github.com/repos/Thimira/bird_watch/releases/latest"
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to get release info: {e}")
        return None

def main():
    """Main download function"""
    print("🦅 Bird Watch Model Downloader")
    print("==============================")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Get latest release
    release_info = get_latest_release()
    if not release_info:
        print("❌ Could not fetch release information")
        return False
    
    print(f"📦 Latest release: {release_info['tag_name']}")
    
    # Find model files in assets
    model_file = None
    class_file = None
    
    for asset in release_info.get('assets', []):
        name = asset['name']
        if name.startswith('final_model_') and name.endswith('.h5'):
            model_file = asset
        elif name.startswith('class_indices_') and name.endswith('.npy'):
            class_file = asset
    
    if not model_file or not class_file:
        print("❌ Could not find required model files in release")
        print("Available files:")
        for asset in release_info.get('assets', []):
            print(f"  - {asset['name']}")
        return False
    
    # Download files
    success = True
    
    # Download model file
    model_path = models_dir / "final_model.h5"
    if not download_file(model_file['browser_download_url'], model_path):
        success = False
    
    # Download class indices file
    class_path = models_dir / "class_indices.npy"
    if not download_file(class_file['browser_download_url'], class_path):
        success = False
    
    if success:
        print("\n🎉 Bird Watch model files downloaded successfully!")
        print("   The model is now ready to use for enhanced bird recognition.")
        print(f"   Files saved to: {models_dir.resolve()}")
    else:
        print("\n❌ Some downloads failed. Please try again or download manually.")
        print("   Manual download: https://github.com/Thimira/bird_watch/releases/latest")
    
    return success

if __name__ == "__main__":
    main()
