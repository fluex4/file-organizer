import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import shutil

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Expanded labeled data for training
# data = [
#     # Reports
#     ("Hyderabad September 12: The weather today is sunny with a high of 75 degrees and a low of 55 degrees.", "report"),
#     ("Bangalore March 14: The conference focused on technology and innovation in artificial intelligence.", "report"),
#     ("Gangtok Jan 13: The company released its quarterly financial report, showing a 10 percent increase in revenue.", "report"),
#     ("Washington DC Jun 12: The research study found significant improvements in the treatment of the disease.", "report"),
#     ("ANC October 13:The annual report highlights the achievements and challenges faced by the organization.", "report"),
#     ("Kolkata Dec 12: The economic outlook for the next quarter predicts steady growth.", "report"),
#     ("Delhi August 22: A recent survey indicates that consumer confidence is at an all-time high.", "report"),
#     ("ISS September 14: The environmental impact report outlines the effects of pollution on local wildlife.", "report"),

#     # Stories
#     ("Once upon a time in a land far away, there lived a dragon.", "story"),
#     ("She lived in a small cottage in the woods, surrounded by animals.", "story"),
#     ("The brave knight set out on a quest to slay the dragon and save the kingdom.", "story"),
#     ("In a galaxy far, far away, a young hero embarks on an adventure to find their destiny.", "story"),
#     ("A young girl discovered a hidden portal to another world in her backyard.", "story"),
#     ("The old man told tales of his youth, filled with adventure and mystery.", "story"),
#     ("Every night, the stars whispered secrets to those who dared to listen.", "story"),
#     ("In the heart of the forest, a magical tree granted wishes to those pure of heart.", "story"),
    
#     # Transcripts
#     ("The meeting was held on Monday, discussing the quarterly results and future strategies.", "transcript"),
#     ("The following is a transcript of the interview with the CEO regarding company vision.", "transcript"),
#     ("The court proceedings were recorded and transcribed for the official record.", "transcript"),
#     ("The panel discussion covered various aspects of the new legislation and its implications.", "transcript"),
#     ("The lecture on climate change emphasized the urgency of taking action now.", "transcript"),
#     ("This is the transcript of the podcast episode featuring industry experts.", "transcript"),
#     ("The seminar included a Q&A session, which was also transcribed for reference.", "transcript"),
#     ("The conference call transcript details the discussions held among board members.", "transcript"),
    
#     # Reviews
#     ("This product is amazing! It exceeded my expectations and I would highly recommend it.", "review"),
#     ("This movie was a thrilling experience with great performances and stunning visuals.", "review"),
#     ("The restaurant offers a delightful dining experience with excellent service and delicious food.", "review"),
#     ("The book kept me engaged from the first page to the last, with a captivating storyline.", "review"),
#     ("The new smartphone has an impressive camera and long battery life, making it worth the price.", "review"),
#     ("I was disappointed with the service at the hotel; it did not meet my expectations.", "review"),
#     ("The concert was unforgettable, with the band playing all their hits.", "review"),
#     ("The online course was well-structured and provided valuable insights into the subject.", "review"),
    
#     # Code
#     ("def my_function(): return 'Hello, World!'", "code"),
#     ("print('Hello, World!')", "code"),
#     ("class MyClass:\n    def __init__(self):\n        self.value = 42", "code"),
#     ("import numpy as np\ndata = np.random.randn(100, 10)", "code"),
#     ("for i in range(10):\n    print(i)", "code"),
#     ("def add(a, b):\n    return a + b", "code"),
#     ("if __name__ == '__main__':\n    main()", "code"),
#     ("try:\n    risky_code()\nexcept Exception as e:\n    print(e)", "code"),
# ]

df = pd.read_csv('final.csv')
data = df.values.tolist()

def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def classify_files_in_folder(folder_path, vectorizer, clf):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # Loop through all text files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Read file content
            file_content = read_file_content(file_path)
            if file_content:
                # Predict the category of the file content
                predicted_category = predict_category(vectorizer, clf, file_content)

                # Create a folder for the predicted category if it doesn't exist
                category_folder = os.path.join(folder_path, predicted_category)
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)

                # Move the file to the appropriate category folder
                destination_path = os.path.join(category_folder, filename)
                shutil.move(file_path, destination_path)
                print(f"Moved {filename} to {category_folder}")
        else:
            print(f"{filename} is not a text file. Skipping.")

# Prepare the training data
texts, labels = zip(*data)

# Function to scrape content from a URL
def scrape_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

# Function to process text
def process_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

# Function to analyze frequencies
def analyze_frequencies(words):
    freq_dist = nltk.FreqDist(words)
    bigrams = ngrams(words, 2)
    bigram_freq = nltk.FreqDist(bigrams)
    trigrams = ngrams(words, 3)
    trigram_freq = nltk.FreqDist(trigrams)
    
    return freq_dist, bigram_freq, trigram_freq

# Function to visualize frequencies
def visualize_frequencies(freq_dist, bigram_freq, trigram_freq):
    # Word Cloud
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(freq_dist)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Frequency Tables
    print("Most Common Words:")
    print(freq_dist.most_common(10))
    
    print("\nMost Common Bigrams:")
    print(bigram_freq.most_common(10))
    
    print("\nMost Common Trigrams:")
    print(trigram_freq.most_common(10))

# Function to train the classifier
def train_classifier(texts, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Unigrams, bigrams, and trigrams
    X = vectorizer.fit_transform(texts)
    y = labels
    return vectorizer, X, y

# Function to predict the category of a new text
def predict_category(vectorizer, clf, text):
    text_vector = vectorizer.transform([text])
    return clf.predict(text_vector)[0]

if __name__ == "__main__":
    # Train the classifier
    vectorizer, X, y = train_classifier(texts, labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    # 0.2 40 - 90

    # Train the Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy:\n", accuracy_score(y_test, y_pred))

    # Get folder path from user
    folder_path = input("Enter the folder path containing text files: ")

    # Classify and organize files in the folder
    classify_files_in_folder(folder_path, vectorizer, clf)