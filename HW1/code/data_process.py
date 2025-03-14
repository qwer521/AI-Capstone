import os
import glob

def rename_files_in_directory(root_dir):
    """
    Recursively renames .txt files in each subdirectory of root_dir.
    For each folder, files are renamed to a sequential number with .txt extension.
    Example: "1 - 複製 (21).txt" -> "0001.txt", "0002.txt", etc.
    """
    # Loop over each subdirectory in the root directory
    for label in os.listdir(root_dir):
        folder = os.path.join(root_dir, label)
        if os.path.isdir(folder):
            # Get a sorted list of txt files in the current folder
            txt_files = sorted(glob.glob(os.path.join(folder, '*.txt')))
            print(f"Renaming files in folder: {folder}")
            for idx, filepath in enumerate(txt_files, start=1):
                # Build new file name with zero-padded numbering (4 digits)
                new_name = f"{idx:04d}.txt"
                new_path = os.path.join(folder, new_name)
                # Rename the file
                os.rename(filepath, new_path)
                print(f"Renamed: {filepath} -> {new_path}")

# Apply renaming to both train and test directories
train_dir = './data/train'
test_dir = './data/test'

rename_files_in_directory(train_dir)
rename_files_in_directory(test_dir)
