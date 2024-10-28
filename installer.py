import os
import shutil
import ctypes

def move_files_and_directories_to_user_folder():
    user_home = os.path.expanduser("~")

    target_folder_name = "File Organizer"
    target_folder_path = os.path.join(user_home, target_folder_name)

    os.makedirs(target_folder_path, exist_ok=True)

    specific_files = ["main.py", "app.py", "requirements.txt", "textclassifier.py", 
                      "sf.bat", "installer.py", "output.csv", "data.csv",'myutils.py']

    project_folder = os.getcwd()

    for file_name in specific_files:
        file_path = os.path.join(project_folder, file_name)

        if os.path.isfile(file_path):
            target_file_path = os.path.join(target_folder_path, file_name)

            shutil.move(file_path, target_file_path)
            print(f"Moved '{file_name}' to '{target_folder_path}'")

    for item in os.listdir(project_folder):
        item_path = os.path.join(project_folder, item)
        if os.path.isdir(item_path) and item != target_folder_name:
            target_dir_path = os.path.join(target_folder_path, item)
            shutil.move(item_path, target_dir_path)
            print(f"Moved directory '{item}' to '{target_folder_path}'")

    # print("Installation sucessful")
    ctypes.windll.user32.MessageBoxW(0, "Installation was successful!", "Installation Status", 0x40)

# Run the function
if __name__ == "__main__":
    move_files_and_directories_to_user_folder()
