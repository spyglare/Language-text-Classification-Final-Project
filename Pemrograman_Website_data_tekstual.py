import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))  # Change 'english' to the appropriate language
    filtered_words = [word.lower() for word in tokens if word.lower() not in stop_words]

    # Join the words back into a single string
    preprocessed_text = ' '.join(filtered_words)

    return preprocessed_text

def classify_language(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the preprocessed text to TF-IDF features
    tfidf_features = vectorizer.fit_transform([preprocessed_text])

    # Define the SVM classifier
    svm = SVC(C=1.0, kernel='linear')

    # Training data
    train_texts = [
        "Bonjour, comment ça va?", "How are you doing?", "Wie geht es dir?",
        "J'aime les croissants", "I love croissants", "Ich liebe Croissants",
        "Merci beaucoup", "Thank you very much", "Vielen Dank"
    ]
    train_labels = ["french", "english", "german", "french", "english", "german", "french", "english", "german"]

    # Transform the training data to TF-IDF features
    train_features = vectorizer.transform(train_texts)

    # Train the SVM classifier
    svm.fit(train_features, train_labels)

    # Classify the input text
    predicted_label = svm.predict(tfidf_features)[0]

    return predicted_label

# Streamlit app
st.title("Language Text Classification")

text = st.text_input("Enter the text:")
if text:
    language = classify_language(text)
    st.write(f"The language of the text is: {language}")
