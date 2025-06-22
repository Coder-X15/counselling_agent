import os
import sys

## this file contains functions to help in loading assets
## the assets are stored in the `assets` folder in the workspace root
def load_assets():
    # this function has to list the filenames in the assets folder
    # the output is a list of the relative paths to the files in the assets folder
    # the output will look like ['assets/sound1.mp3', 'assets/sound2.mp3', ...]
    assets_folder = os.path.join(os.path.dirname(__file__), '..', 'assets')
    if not os.path.exists(assets_folder):
        print("Assets folder not found. Please create the assets folder in the workspace root.")
        sys.exit(1)
    assets = []
    for root, dirs, files in os.walk(assets_folder):
        for file in files:
            if file.endswith('.mp3'):
                relative_path = os.path.relpath(os.path.join(root, file), assets_folder)
                assets.append('assets/' + relative_path)
    return assets

# to obtain a random asset
import random
assets = load_assets()
print(assets)
path = random.choice(assets)