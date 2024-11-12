import os
import shutil

def delete_all_folders_in_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):  # Check if it's a directory
            shutil.rmtree(item_path)   # Delete the folder and its contents

# Usage
directory_path = './exp/runs'
delete_all_folders_in_directory(directory_path)
