import os
import shutil

def delete_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Delete the folder and its contents
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"The folder '{folder_path}' does not exist.")

    except Exception as e:
        print(f"An error occurred: {e}")

def create_folder(folder_path):
    try:
        # Check if the folder already exists
        if not os.path.exists(folder_path):
            # Create the folder and its parents if they don't exist
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"The folder '{folder_path}' already exists.")

    except Exception as e:
        print(f"An error occurred: {e}")


models_folder = "./models"


delete_folder(models_folder)
create_folder(models_folder)

print("Clear successful")