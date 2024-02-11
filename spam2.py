import tkinter as tk
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

#nltk.download("stopwords")

# Load data from your CSV file
df = pd.read_csv('spam.csv')  # Replace 'your_dataset.csv' with the actual file path

labels = df.iloc[:, 0].tolist()  # Assuming the labels are in the first column
corpus = df.iloc[:, 1].tolist()  # Assuming the email text is in the second column

# Tokenize and remove stopwords
stop_words = set(stopwords.words("english"))
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)

# Train a simple spam classifier
clf = MultinomialNB()
clf.fit(X, labels)

def classify_email():
    email_text = email_text_widget.get("1.0", "end-1c")  # Get the text from the text widget
    test_email_features = vectorizer.transform([email_text])
    prediction = clf.predict(test_email_features)

    if prediction[0] == "spam":
        result_label.config(text="This is a spam email.")
    else:
        result_label.config(text="This is not a spam email (ham).")

# Create a simple Tkinter window
window = tk.Tk()
window.title("Email Spam Classifier")

# Label
email_label = tk.Label(window, text="Paste your email text:")
email_label.pack()

# Text Entry (use a Text widget instead)
email_text_widget = tk.Text(window, height=10, width=40)
email_text_widget.pack()

# Classify Button
classify_button = tk.Button(window, text="Classify", command=classify_email)
classify_button.pack()

# Result Label
result_label = tk.Label(window, text="")
result_label.pack()

# Run the UI
window.mainloop()
