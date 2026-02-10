import os

def rename_files_recursive(directory_path):
    """
    Recursively renames all files under a specified directory, keeping only characters before the first space.
    
    Args:
        directory_path (str): Path to the directory to process
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Get the full path of the current file
            old_path = os.path.join(root, file)
            
            # Split the filename and extension
            name, ext = os.path.splitext(file)
            
            # Find the position of the first space in the name
            space_pos = name.find(' ')
            
            # If space is found, keep only characters before it
            if space_pos != -1:
                new_name = name[:space_pos] + ext
            else:
                # If no space is found, keep the original name
                new_name = name + ext
            
            # Ensure we don't overwrite existing files
            new_path = os.path.join(root, new_name)
            
            # Handle potential name collisions
            counter = 1
            while os.path.exists(new_path) and old_path != new_path:
                if space_pos != -1:
                    new_name = name[:space_pos] + f"_{counter}" + ext
                else:
                    new_name = name + f"_{counter}" + ext
                new_path = os.path.join(root, new_name)
                counter += 1
            
            # Rename the file if the name is different
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            else:
                print(f"Skipped (no change needed): {old_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python rename_files.py <directory_path>")
        print("Example: python rename_files.py /path/to/directory")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        sys.exit(1)
    
    print(f"Renaming files in: {directory_path}")
    print("Keeping only characters before the first space...")
    
    rename_files_recursive(directory_path)
    
    print("Done!")