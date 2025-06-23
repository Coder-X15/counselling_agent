import os
import sys
import random

## this file contains functions to help in loading assets
## the assets are stored in the `assets` folder in the workspace root
def load_assets():
    # this function lists the absolute paths to mp3 files in the assets folder
    assets_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
    if not os.path.exists(assets_folder):
        print("Assets folder not found. Please create the assets folder in the workspace root.")
        sys.exit(1)
    assets = []
    for root, dirs, files in os.walk(assets_folder):
        for file in files:
            if file.endswith('.mp3'):
                abs_path = os.path.join(root, file)
                assets.append(abs_path)
    return assets

# to obtain a random asset
assets = load_assets()
path = random.choice(assets) if assets else None