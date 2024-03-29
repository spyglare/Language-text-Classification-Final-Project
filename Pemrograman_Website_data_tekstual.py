{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPpCg4gMqAhLkbLmeUaiMT/",
      "include_colab_link": True
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/spyglare/Language-text-Classification-Final-Project/blob/main/Pemrograman_Website_data_tekstual.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bZGnJtZCnKgF",
        "outputId": "ef4d7cac-835a-4e90-ccc6-e379e7fc6504"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enter the text: Ich liebe die\n",
            "The language of the text is: german\n"
          ]
        }
      ],
      "source": [
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.svm import SVC\n",
        "\n",
        "nltk.download('punkt')\n",
        "nltk.download('stopwords')\n",
        "\n",
        "def preprocess_text(text):\n",
        "    # Tokenize the text into words\n",
        "    tokens = word_tokenize(text)\n",
        "\n",
        "    # Filter out stopwords\n",
        "    stop_words = set(stopwords.words('english'))  # Change 'english' to the appropriate language\n",
        "    filtered_words = [word.lower() for word in tokens if word.lower() not in stop_words]\n",
        "\n",
        "    # Join the words back into a single string\n",
        "    preprocessed_text = ' '.join(filtered_words)\n",
        "\n",
        "    return preprocessed_text\n",
        "\n",
        "def classify_language(text):\n",
        "    # Preprocess the input text\n",
        "    preprocessed_text = preprocess_text(text)\n",
        "\n",
        "    # Create TF-IDF vectorizer\n",
        "    vectorizer = TfidfVectorizer()\n",
        "\n",
        "    # Transform the preprocessed text to TF-IDF features\n",
        "    tfidf_features = vectorizer.fit_transform([preprocessed_text])\n",
        "\n",
        "    # Define the SVM classifier\n",
        "    svm = SVC(C=1.0, kernel='linear')\n",
        "\n",
        "    # Training data\n",
        "    train_texts = [\n",
        "        \"Bonjour, comment ça va?\", \"How are you doing?\", \"Wie geht es dir?\",\n",
        "        \"J'aime les croissants\", \"I love croissants\", \"Ich liebe Croissants\",\n",
        "        \"Merci beaucoup\", \"Thank you very much\", \"Vielen Dank\"\n",
        "    ]\n",
        "    train_labels = [\"french\", \"english\", \"german\", \"french\", \"english\", \"german\", \"french\", \"english\", \"german\"]\n",
        "\n",
        "    # Transform the training data to TF-IDF features\n",
        "    train_features = vectorizer.transform(train_texts)\n",
        "\n",
        "    # Train the SVM classifier\n",
        "    svm.fit(train_features, train_labels)\n",
        "\n",
        "    # Classify the input text\n",
        "    predicted_label = svm.predict(tfidf_features)[0]\n",
        "\n",
        "    return predicted_label\n",
        "\n",
        "text = input(\"Enter the text: \")\n",
        "language = classify_language(text)\n",
        "print(f\"The language of the text is: {language}\")"
      ]
    }
  ]
}
