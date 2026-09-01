import gradio as gr
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.stem import PorterStemmer
from spacy import displacy


# =========================================================
# LOAD NLP MODEL
# =========================================================

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()


# =========================================================
# 1. NAMED ENTITY RECOGNITION
# =========================================================

def named_entity_recognition(text):

    if not text.strip():
        return pd.DataFrame(
            columns=["Entity", "Label", "Description"]
        )

    doc = nlp(text)

    data = []

    for ent in doc.ents:
        data.append({
            "Entity": ent.text,
            "Label": ent.label_,
            "Description": spacy.explain(ent.label_)
        })

    return pd.DataFrame(data)


# =========================================================
# 2. POS TAGGING
# =========================================================

def pos_tagging(text):

    if not text.strip():
        return pd.DataFrame(
            columns=[
                "Token",
                "POS",
                "Detailed Tag",
                "Description"
            ]
        )

    doc = nlp(text)

    data = []

    for token in doc:
        if not token.is_space:
            data.append({
                "Token": token.text,
                "POS": token.pos_,
                "Detailed Tag": token.tag_,
                "Description": spacy.explain(token.tag_)
            })

    return pd.DataFrame(data)


# =========================================================
# 3. POS DISTRIBUTION
# =========================================================

def pos_distribution(text):

    if not text.strip():
        return None

    doc = nlp(text)

    counts = Counter(
        token.pos_
        for token in doc
        if not token.is_space
    )

    if not counts:
        return None

    df = pd.DataFrame(
        list(counts.items()),
        columns=["POS", "Count"]
    ).sort_values(
        "Count",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        df["POS"],
        df["Count"]
    )

    ax.set_title("Part-of-Speech Distribution")
    ax.set_xlabel("POS Tag")
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig


# =========================================================
# 4. LEMMATIZATION
# =========================================================

def lemmatization(text):

    if not text.strip():
        return pd.DataFrame(
            columns=["Token", "Lemma", "POS"]
        )

    doc = nlp(text)

    data = []

    for token in doc:
        if not token.is_space:
            data.append({
                "Token": token.text,
                "Lemma": token.lemma_,
                "POS": token.pos_
            })

    return pd.DataFrame(data)


# =========================================================
# 5. STEMMING
# =========================================================

def stemming(text):

    if not text.strip():
        return pd.DataFrame(
            columns=["Token", "Stem"]
        )

    words = text.split()

    data = []

    for word in words:

        clean_word = word.strip(
            ".,!?;:\"'()[]{}"
        )

        if clean_word:
            data.append({
                "Token": clean_word,
                "Stem": stemmer.stem(clean_word)
            })

    return pd.DataFrame(data)


# =========================================================
# 6. MORPHOLOGY
# =========================================================

def morphology(text):

    if not text.strip():
        return pd.DataFrame(
            columns=[
                "Token",
                "POS",
                "Morphology"
            ]
        )

    doc = nlp(text)

    data = []

    for token in doc:

        if not token.is_space:

            morph = str(token.morph)

            if not morph:
                morph = "None"

            data.append({
                "Token": token.text,
                "POS": token.pos_,
                "Morphology": morph
            })

    return pd.DataFrame(data)


# =========================================================
# 7. DEPENDENCY PARSING
# =========================================================

def dependency_parsing(text):

    if not text.strip():
        return """
        <div class="empty-message">
            Please enter text to generate
            dependency parsing.
        </div>
        """

    doc = nlp(text)

    html = displacy.render(
        doc,
        style="dep",
        page=True
    )

    return html


# =========================================================
# CUSTOM CSS
# =========================================================

custom_css = """

body {
    background: #0b1020;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

.ai-header {
    text-align: center;
    padding: 25px;
    margin-bottom: 20px;
}

.ai-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 1px;
}

.ai-subtitle {
    font-size: 17px;
    opacity: 0.75;
    margin-top: 8px;
}

.ai-badge {
    display: inline-block;
    padding: 7px 18px;
    border-radius: 30px;
    font-size: 13px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.2);
}

.input-card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
}

.analyze-btn {
    width: 100%;
    margin-top: 12px;
    font-size: 17px !important;
    font-weight: bold !important;
    border-radius: 12px !important;
}

.tabs {
    border-radius: 16px;
}

button.selected {
    font-weight: bold !important;
}

.section-heading {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 8px;
}

.description {
    opacity: 0.7;
    margin-bottom: 15px;
}

.empty-message {
    padding: 30px;
    text-align: center;
    opacity: 0.65;
}

.footer {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    opacity: 0.55;
    font-size: 13px;
}

"""


# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="AI NLP Dashboard",
    css=custom_css
) as demo:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    gr.HTML(
        """
        <div class="ai-header">

            <div class="ai-badge">
                ✦ AI POWERED NLP
            </div>

            <div class="ai-title">
                NLP Intelligence Dashboard
            </div>

            <div class="ai-subtitle">
                Analyze text using Natural Language Processing
                techniques powered by Python, spaCy and NLTK.
            </div>

        </div>
        """
    )


    # -----------------------------------------------------
    # INPUT SECTION
    # -----------------------------------------------------

    with gr.Group():

        gr.Markdown(
            "### 🧠 Input Text",
            elem_classes="section-heading"
        )

        gr.Markdown(
            "Enter a sentence or paragraph to begin NLP analysis.",
            elem_classes="description"
        )

        text_input = gr.Textbox(
            placeholder=(
                "Example: Apple is planning to open "
                "a new office in Delhi next year."
            ),
            lines=6,
            label="Text Input"
        )

        analyze_button = gr.Button(
            "✨ Analyze Text",
            variant="primary",
            elem_classes="analyze-btn"
        )


    # =====================================================
    # NLP MODULES
    # =====================================================

    with gr.Tabs():

        # -------------------------------------------------
        # NER
        # -------------------------------------------------

        with gr.Tab("🏷️ Named Entities"):

            gr.Markdown(
                "### Named Entity Recognition"
            )

            gr.Markdown(
                "Identifies people, organizations, locations, "
                "dates and other entities.",
                elem_classes="description"
            )

            ner_output = gr.Dataframe(
                headers=[
                    "Entity",
                    "Label",
                    "Description"
                ],
                interactive=False
            )


        # -------------------------------------------------
        # POS TAGGING
        # -------------------------------------------------

        with gr.Tab("🔤 POS Tagging"):

            gr.Markdown(
                "### Part-of-Speech Tagging"
            )

            gr.Markdown(
                "Analyzes the grammatical role of each token.",
                elem_classes="description"
            )

            pos_output = gr.Dataframe(
                headers=[
                    "Token",
                    "POS",
                    "Detailed Tag",
                    "Description"
                ],
                interactive=False
            )


        # -------------------------------------------------
        # POS DISTRIBUTION
        # -------------------------------------------------

        with gr.Tab("📊 POS Distribution"):

            gr.Markdown(
                "### POS Distribution"
            )

            gr.Markdown(
                "Visual representation of the frequency "
                "of different POS tags.",
                elem_classes="description"
            )

            pos_graph = gr.Plot()


        # -------------------------------------------------
        # LEMMATIZATION
        # -------------------------------------------------

        with gr.Tab("🔎 Lemmatization"):

            gr.Markdown(
                "### Lemmatization"
            )

            gr.Markdown(
                "Converts words into their meaningful "
                "dictionary base form.",
                elem_classes="description"
            )

            lemma_output = gr.Dataframe(
                headers=[
                    "Token",
                    "Lemma",
                    "POS"
                ],
                interactive=False
            )


        # -------------------------------------------------
        # STEMMING
        # -------------------------------------------------

        with gr.Tab("🌱 Stemming"):

            gr.Markdown(
                "### Stemming"
            )

            gr.Markdown(
                "Reduces words to their root-like form "
                "using the Porter Stemmer.",
                elem_classes="description"
            )

            stem_output = gr.Dataframe(
                headers=[
                    "Token",
                    "Stem"
                ],
                interactive=False
            )


        # -------------------------------------------------
        # MORPHOLOGY
        # -------------------------------------------------

        with gr.Tab("🧬 Morphology"):

            gr.Markdown(
                "### Morphological Analysis"
            )

            gr.Markdown(
                "Displays grammatical features such as "
                "tense, number, gender and aspect.",
                elem_classes="description"
            )

            morphology_output = gr.Dataframe(
                headers=[
                    "Token",
                    "POS",
                    "Morphology"
                ],
                interactive=False
            )


        # -------------------------------------------------
        # DEPENDENCY PARSING
        # -------------------------------------------------

        with gr.Tab("🔗 Dependencies"):

            gr.Markdown(
                "### Dependency Parsing"
            )

            gr.Markdown(
                "Visualizes grammatical relationships "
                "between words using dependency parsing.",
                elem_classes="description"
            )

            dependency_output = gr.HTML()


    # =====================================================
    # CONNECT BACKEND TO FRONTEND
    # =====================================================

    analyze_button.click(
        fn=named_entity_recognition,
        inputs=text_input,
        outputs=ner_output
    )

    analyze_button.click(
        fn=pos_tagging,
        inputs=text_input,
        outputs=pos_output
    )

    analyze_button.click(
        fn=pos_distribution,
        inputs=text_input,
        outputs=pos_graph
    )

    analyze_button.click(
        fn=lemmatization,
        inputs=text_input,
        outputs=lemma_output
    )

    analyze_button.click(
        fn=stemming,
        inputs=text_input,
        outputs=stem_output
    )

    analyze_button.click(
        fn=morphology,
        inputs=text_input,
        outputs=morphology_output
    )

    analyze_button.click(
        fn=dependency_parsing,
        inputs=text_input,
        outputs=dependency_output
    )


    # =====================================================
    # FOOTER
    # =====================================================

    gr.HTML(
        """
        <div class="footer">
            NLP Intelligence Dashboard • Python • spaCy • NLTK • Gradio
        </div>
        """
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    import os

    port = os.environ.get("PORT")

    if port:
        demo.launch(
            server_name="0.0.0.0",
            server_port=int(port)
        )
    else:
        demo.launch()