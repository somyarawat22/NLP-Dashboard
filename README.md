# 🧠 NLP Intelligence Dashboard

An interactive Natural Language Processing (NLP) dashboard built using Python, Gradio, spaCy, and NLTK. This project provides a user-friendly interface to analyze text using seven different NLP techniques.

## 🚀 Features

1. **Named Entity Recognition (NER):** Identifies entities such as people, organizations, locations, and dates.
2. **Part-of-Speech (POS) Tagging:** Identifies the grammatical category of each word, such as nouns, verbs, adjectives, and adverbs.
3. **POS Distribution:** Displays a bar chart showing the frequency of different parts of speech.
4. **Lemmatization:** Converts words into their meaningful dictionary base forms.
5. **Stemming:** Reduces words to their root-like forms using the Porter Stemming algorithm.
6. **Morphological Analysis:** Displays grammatical features of words, such as tense, number, and verb form.
7. **Dependency Parsing:** Visualizes grammatical relationships between words.

## 🛠️ Technologies Used

- Python
- Gradio
- spaCy
- NLTK
- Pandas
- Matplotlib

## 📁 Project Structure

```text
NLP-Dashboard/
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/NLP-Dashboard.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Navigate to the Project Folder

```bash
cd NLP-Dashboard
```

### 3. Install the Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy English Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the Application

```bash
python app.py
```

Open the local URL displayed after the application starts to use the dashboard.

## 💻 How to Use

1. Open the NLP Intelligence Dashboard.
2. Enter a sentence or paragraph in the text input box.
3. Click the **Analyze Text** button.
4. Select any of the seven tabs to explore the results.
5. View the NLP analysis in tables, charts, and dependency visualizations.

## 🌐 Deployment on Render

This application can be deployed on Render by connecting the GitHub repository.

**Build Command:**

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
```

**Start Command:**

```bash
python app.py
```

After successful deployment, Render provides a public URL to access the dashboard.

## 🎯 Project Objective

The objective of this project is to demonstrate fundamental Natural Language Processing techniques through an interactive dashboard. It combines text processing, grammatical analysis, and data visualization in a single interface.

## 📚 Applications

- Text analysis
- Grammatical analysis
- Linguistic feature extraction
- Educational NLP demonstrations
- Basic language processing

## 👩‍💻 Project Information

**Project Name:** NLP Intelligence Dashboard  
**Domain:** Natural Language Processing (NLP)  
**Programming Language:** Python  
**Frontend:** Gradio  
**NLP Libraries:** spaCy and NLTK  
**Visualization:** Matplotlib

---
