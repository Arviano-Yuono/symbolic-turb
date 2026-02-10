# this folder contains utility functions for folder operations

def create_folder_if_not_exists(folder_path: str) -> None:
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return None

def list_files_in_folder(folder_path: str) -> list:
    import os
    if not os.path.exists(folder_path):
        return []
    return os.listdir(folder_path)