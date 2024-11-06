import os
import time
import argparse
import shutil
from textclassifier import TextFileClassifier
import pandas as pd

        

class SortFiles:
    def __init__(self):
        self.args = self.parse_args()
        self.dirName = os.path.basename(os.getcwd())
        print(f"Current Directory: {self.dirName}")
        self.ls = os.listdir()
        print('listing all files:')
        for l in self.ls:
            print(l,end=',')
        # print(self.args)
        print('\nStarting File Sort: ')
        self.log_file_path = "sorting_log.txt"
        with open(self.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write("Sorting Log:\n")

        if self.args.s is not None:
            self.sortSize(self.args.s)
        elif self.args.t:
            self.sortType()
        elif self.args.e:
            self.sortExtension()
        elif self.args.af:
            self.sortAccessFrequency()
        elif self.args.d is not None:
            self.sortDate(self.args.d)
        elif self.args.c is not None:
            self.sortContent(self.args.c)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Choose sorting modes")
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--s", type=int, help="Sort according to size mentioned. It requires an integer (size in bytes), which it uses as a base to classify. example: `sf --s 1024`.")
        group.add_argument("--t", action="store_true", help="Sort by file type. example: `sf --t`.")
        group.add_argument("--e", action="store_true", help="Sort by file extension. example: `sf --e`.")
        group.add_argument("--af", action="store_true", help="Sort files by access frequency. example: `sf --af`.")
        group.add_argument("--d", type=str, help="Sort according to date of creation. It accepts a string argument, it takes values 'd' for day, 'm' for month, 'y' for year, 'ty' for 10 years. example: `sf --d d`.")
        group.add_argument("--c", type=str, help="Sort according to file content. It accepts a string argument, its values can be 'n' for normal mode, 'ai' for advanced mode. example `sf --c ai`.")
        
        return parser.parse_args()

    def log_change(self, original_path, new_directory):

        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"'{original_path}' - '{new_directory}'\n")
            print(f"'{original_path}' - '{new_directory}'")

    def sortSize(self, size):

        for file in self.ls:
            if os.path.isfile(file):
                file_size = os.path.getsize(file)
                if file_size > size:
                    directory_name = 'larger_than_' + str(size) + 'B'
                else:
                    directory_name = 'smaller_than_' + str(size) + 'B'
                
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
                
                original_path = os.path.join(self.dirName, file)
                shutil.move(file, os.path.join(directory_name, file))
                self.log_change(original_path, os.path.join(directory_name, file))

    def sortExtension(self):

        for file in self.ls:
            if os.path.isfile(file):
                file_extension = os.path.splitext(file)[1][1:]  
                directory_name = file_extension if file_extension else 'no_extension'

                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
                
                original_path = os.path.join(self.dirName, file)
                shutil.move(file, os.path.join(directory_name, file))
                self.log_change(original_path, os.path.join(directory_name, file))

    def sortType(self):

        file_types = {
            'code': ['.py', '.java', '.cpp', '.js', '.html', '.css'],
            'document': ['.pdf', '.doc', '.docx', '.txt', '.ppt', '.pptx'],
            'picture': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'audio': ['.mp3', '.wav', '.aac', '.ogg'],
            'video': ['.mp4', '.avi', '.mov', '.mkv'],
            'archive': ['.zip', '.tar', '.gz', '.rar'],
            'other': [] 
        }

        for file in self.ls:
            if os.path.isfile(file):
                file_extension = os.path.splitext(file)[1].lower()
                directory_name = 'other' 
                
                for file_type, extensions in file_types.items():
                    if file_extension in extensions:
                        directory_name = file_type
                        break
                
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
                
                original_path = os.path.join(self.dirName, file)
                shutil.move(file, os.path.join(directory_name, file))
                self.log_change(original_path, os.path.join(directory_name, file))

    def sortAccessFrequency(self):

        current_time = time.time()
        for file in self.ls:
            if os.path.isfile(file):
                last_accessed = os.path.getatime(file)
                if current_time - last_accessed < 86400: 
                    directory_name = 'accessed_today'
                elif current_time - last_accessed < 604800: 
                    directory_name = 'accessed_this_week'
                else:
                    directory_name = 'accessed_older_than_a_week'

                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
                
                original_path = os.path.join(self.dirName, file)
                shutil.move(file, os.path.join(directory_name, file))
                self.log_change(original_path, os.path.join(directory_name, file))

    def sortDate(self, date_type):

        for file in self.ls:
            if os.path.isfile(file):
                creation_time = os.path.getctime(file)
                creation_time_struct = time.localtime(creation_time)

                if date_type == 'd':
                    directory_name = time.strftime('%Y-%m-%d', creation_time_struct)
                elif date_type == 'm':
                    directory_name = time.strftime('%Y-%m', creation_time_struct)
                elif date_type == 'y':
                    directory_name = time.strftime('%Y', creation_time_struct)
                elif date_type == 'ty':
                    decade = (creation_time_struct.tm_year // 10) * 10
                    directory_name = f"{decade}s"
                else:
                    print(f"\nInvalid argument for date sorting: {date_type}, do `sf -h` for help")
                    continue

                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)

                original_path = os.path.join(self.dirName, file)
                shutil.move(file, os.path.join(directory_name, file))
                self.log_change(original_path, os.path.join(directory_name, file))

    def sortContent(self,mode):
        if mode == "n":
            classifier = TextFileClassifier()
            if classifier.vectorizer is None or classifier.clf is None:
                data_path = os.path.join(classifier.DATA_DIR, "data.csv")
                data = pd.read_csv(data_path)
                texts = data["text"].tolist()
                labels = data["label"].tolist()
                classifier.train_classifier(texts, labels)
            print(self.dirName)
            classifier.classify_files_in_folder(self.dirName)
        elif mode == 'ai':
            pass
            
        else :
            print('\nInvalid arguments passed for Content based sorting: '+ mode+ ' do `sf -h` for help')

if __name__ == "__main__":
    SortFiles()
