{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/spyglare/Language-text-Classification-Final-Project/blob/main/Copy_of_Welcome_To_Colaboratory.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
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
      ],
      "metadata": {
        "id": "vwxY2-h4vK9J",
        "outputId": "68836451-e21a-41db-afdb-464b88915e21",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
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
            "Enter the text:  je t'aime\n",
            "The language of the text is: french\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "! pip install streamlit -q"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vDu02DuWPAJb",
        "outputId": "feb363b7-71c1-4338-f74a-985839023a76"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.9/8.9 MB\u001b[0m \u001b[31m47.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m164.8/164.8 kB\u001b[0m \u001b[31m18.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m184.3/184.3 kB\u001b[0m \u001b[31m18.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.8/4.8 MB\u001b[0m \u001b[31m85.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.1/82.1 kB\u001b[0m \u001b[31m9.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m6.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m341.8/341.8 kB\u001b[0m \u001b[31m24.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Building wheel for validators (setup.py) ... \u001b[?25l\u001b[?25hdone\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "st.write('Hello, *World!* :sunglasses:')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YcrIG10xPmmU",
        "outputId": "6c3a5dfc-759e-4774-ded5-8870f2aa3c37"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!npm install localtunnel"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f4pQsVNhP_iL",
        "outputId": "818fd9cb-52b2-466a-b1a8-9f565f587ff4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[K\u001b[?25h\u001b[37;40mnpm\u001b[0m \u001b[0m\u001b[30;43mWARN\u001b[0m \u001b[0m\u001b[35msaveError\u001b[0m ENOENT: no such file or directory, open '/content/package.json'\n",
            "\u001b[0m\u001b[37;40mnpm\u001b[0m \u001b[0m\u001b[34;40mnotice\u001b[0m\u001b[35m\u001b[0m created a lockfile as package-lock.json. You should commit this file.\n",
            "\u001b[0m\u001b[37;40mnpm\u001b[0m \u001b[0m\u001b[30;43mWARN\u001b[0m \u001b[0m\u001b[35menoent\u001b[0m ENOENT: no such file or directory, open '/content/package.json'\n",
            "\u001b[0m\u001b[37;40mnpm\u001b[0m \u001b[0m\u001b[30;43mWARN\u001b[0m\u001b[35m\u001b[0m content No description\n",
            "\u001b[0m\u001b[37;40mnpm\u001b[0m \u001b[0m\u001b[30;43mWARN\u001b[0m\u001b[35m\u001b[0m content No repository field.\n",
            "\u001b[0m\u001b[37;40mnpm\u001b[0m \u001b[0m\u001b[30;43mWARN\u001b[0m\u001b[35m\u001b[0m content No README data\n",
            "\u001b[0m\u001b[37;40mnpm\u001b[0m \u001b[0m\u001b[30;43mWARN\u001b[0m\u001b[35m\u001b[0m content No license field.\n",
            "\u001b[0m\n",
            "+ localtunnel@2.0.2\n",
            "added 22 packages from 22 contributors and audited 22 packages in 3.133s\n",
            "\n",
            "3 packages are looking for funding\n",
            "  run `npm fund` for details\n",
            "\n",
            "found \u001b[92m0\u001b[0m vulnerabilities\n",
            "\n",
            "\u001b[K\u001b[?25h"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run app.py &>/content/logs.txt &"
      ],
      "metadata": {
        "id": "Om-9shXMQDUj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!npx localtunnel --port 8501  & curl ipv4.icanhazip.com"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9HNW59AlQJBk",
        "outputId": "65597a0c-fd55-40bc-a6d8-1a468362ddac"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "34.147.74.93\n",
            "\u001b[K\u001b[?25hnpx: installed 22 in 2.372s\n",
            "your url is: https://icy-ravens-bet.loca.lt\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import urllib\n",
        "print(\"Password/Enpoint IP for localtunnel is:\",urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip(\"\\n\"))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SL0o1i70QUEj",
        "outputId": "826c2152-2853-4f09-8ea5-57d374b66003"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Password/Enpoint IP for localtunnel is: 34.147.74.93\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ! pip install pyngrok"
      ],
      "metadata": {
        "id": "K12TZFhmZ2Wr"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "toc_visible": true,
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}