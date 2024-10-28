
# File Organizer
___ ___
###### made using [python](https://www.python.org/)

![Build Status](https://badgen.net/github/last-commit/fluex4/file-organizer)


This project presents an intelligent file organizer designed to automate digital file classification and management. Using machine learning and natural language processing (NLP), this system categorizes files by type, date, and content. Its core features include:
- Automated categorization: by file type, extension, size, creation date, and access frequency.
- Content analysis: to group files by topics, keywords, language, and content length.
- Document purpose identification: to organize files intuitively.


## Features 
-- -- 
### 1. Various criteria to classify and organize files
- File Size
- File Access Frequency
- File Type
- File Extension
- Date of Creation 
- File Content

### 2. Tracking File Path Changes
- a log file, which tracks all the change in file paths.



## Tech
-- -- 
File Organizer uses a number of packages to work properly:

- [PyPDF2](https://pypdf2.readthedocs.io/en/latest/): A library for reading, manipulating, and writing PDF files in Python.
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/): A fast and lightweight library for working with PDF and other document formats.
- [python-docx](https://python-docx.readthedocs.io/en/latest/): A library for creating, modifying, and extracting information from Microsoft Word (.docx) files.
- [python-pptx](https://python-pptx.readthedocs.io/en/latest/): A library for creating and manipulating PowerPoint (.pptx) presentations.
- [joblib](https://joblib.readthedocs.io/en/latest/): A library for lightweight pipelining in Python, particularly useful for saving and loading Python objects.
- [pandas](https://pandas.pydata.org/): A powerful data manipulation and analysis library for Python, providing data structures like DataFrames.
- [nltk](https://www.nltk.org/): The Natural Language Toolkit, a library for working with human language data (text) in Python.
- [scikit-learn](https://scikit-learn.org/stable/): A machine learning library in Python that provides simple and efficient tools for data mining and analysis.
- [wordcloud](https://github.com/amueller/word_cloud): A library for creating word clouds from text data in Python.
- [matplotlib](https://matplotlib.org/): A comprehensive library for creating static, animated, and interactive visualizations in Python.



## Installation
-- -- 
It requires [Python](https://www.python.org/) v3+ to run.

##### 1. Clone this repo.
``` git clone "https://github.com/fluex4/file-organizer.git" ```

##### 2.Install the dependencies.
```pip install -r requirements.txt```

##### 3. Run the `install.bat`
it creates a folder in the user directory, consisting of all the necessary files.

##### 4. Add this folder path to user environment variable / path.
so that you can run it any where

> [!NOTE]
> this is mandatory as sorting files need to be done at any place.

## Guide
-- -- 
it is a CLI application, to sort files into their directory, just open cmd from that directory and run `sf -h` or `sf --help` for help.
`sf` stands for 'sort files'
(You can change this `sf` command by renaming the `/sf.bat` file in the user directory.)
there are different modes available to sort files which can be given as arguments,they are as follows: 
- `--af`: sort files by access frequency
        takes no other arguments.
- `--c`: sorts files by file content.
        takes string:mode as argument, which can be `n` for normal mode, `ai` for advanced mode.
- `--d`: sorts files by date of creation.
        takes string:span as argument, which can be `d` for day, `m` for month, `y` for year and `ty` for a decade.
- `--e`: sorts file by the extension.
- `--s`: sorts file by its type,
         takes int:size as argument.
- `--t`: sorts file by its type.
-- -- 
##### Example usage: 
```
    sf --c ai
```

> [!TIP]
> Report bugs @_itsneverreallyover on discord or instagram.

> [!WARNING]
> Currently the advanced mode of sort by content is under build, thus selecting it will do nothing
