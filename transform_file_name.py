import os
import re

# Folder path
folder_path = "./training_samples_1"

# Regex to find numbers followed by 'cm'
pattern = re.compile(r'(\d+\s*cm)')

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if os.path.isfile(old_path):
        # Replace matches with square brackets
        new_filename = pattern.sub(r'[\1]', filename)

        if new_filename != filename:
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')
