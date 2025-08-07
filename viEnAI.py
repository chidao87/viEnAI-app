import streamlit as st
from streamlit_modal import Modal
import streamlit_lottie
import time
import json
import os
from transformers import MarianMTModel, MarianTokenizer
import spacy
from nltk.corpus import wordnet
from PIL import Image


# Set up page configuration
st.set_page_config(page_title="ViEng AI", page_icon="logo.png", initial_sidebar_state="expanded")

# Display tab options
chosen_tab = st.radio(
    "Choose a section:",
    ['🏠 Home', '🌐 Translate', '🔍 Analyze'],
    index=0,
    horizontal=True
)


# Footer style and content
footer_style = """
    <style>
    footer {
        text-align: center;
        padding: 1rem;
        background-color: #f1f1f1;
    }
    </style>
"""
footer = '<footer><p> AI-powered English learning website for Vietnamese </p></footer>'

# Session state for loading screen
if "loading_done" not in st.session_state:
    st.session_state.loading_done = False

if not st.session_state.loading_done:
    st.image("logo.gif", caption="Loading...", use_column_width=True)  # Add your GIF path
    time.sleep(3)
    st.session_state.loading_done = True
    st.rerun()

# Layout adjustments
max_width_str = f"max-width: {75}%;"
st.markdown(f"""
    <style>
    .appview-container .main .block-container{{{max_width_str}}}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        margin-top: 40px; 
    }
    </style>
""", unsafe_allow_html=True)

# Define page names and icons for tabs
HOME = 'Home'
TRANSLATE = 'Translate'
ANALYZE = 'Analyze'

tabs = [
    HOME,
    TRANSLATE,
    ANALYZE
]

option_data = [
    {'icon': "🏠", 'label': HOME},
    {'icon': "🌐", 'label': TRANSLATE},
    {'icon': "🔍", 'label': ANALYZE}
]

over_theme = {'txc_inactive': 'white', 'menu_background': '#F5B7B1', 'txc_active': 'white', 'option_active': '#CD5C5C'}

# Set up the main app layout with the NavBar options
chosen_tab = st.radio(
    "Choose a section:",
    [HOME, TRANSLATE, ANALYZE],
    index=0,
    horizontal=True
)

# Define functions for each page

# Home Page
def home_page():
# Load image from file
    img = Image.open("vietnam.png")
    st.image(img)
    st.title("📖 ViEng AI 📖: Your Smart English Learning Assistant")
    
    # Add introductory content
    st.markdown("""
    ViEng AI is an innovative, AI-powered platform designed to assist Vietnamese speakers in learning English with ease. Our cutting-edge technology helps you improve your English proficiency through seamless translation, grammar analysis, and tense detection. 
    """)

    # Features section
    st.subheader("Key Features of ViEng AI")
    st.markdown("""
    - **Instant Translation**: Effortlessly translate Vietnamese sentences into English with high accuracy.
    - **Grammar Analysis**: Gain insights into the grammatical structure of your translated sentences.
    - **Tense Detection**: Automatically detect the tenses used in your English translations for a deeper understanding.
    - **User-Friendly Interface**: Experience a clean, intuitive interface that makes learning easy and engaging.
    """)

    # How it works section
    st.subheader("How ViEng AI Works")
    st.markdown("""
    1. **Enter Your Sentence**: Simply type a sentence in Vietnamese.
    2. **AI Translates**: The system translates your sentence into English instantly.
    3. **Grammar Breakdown**: Understand the parts of speech and sentence structure with our detailed analysis.
    4. **Tense Detection**: Get insights on the tense used in the sentence, improving your grammar comprehension.
    """)

    # Call-to-action section
    st.subheader("Ready to Enhance Your English Skills?")
    st.markdown("""
    Get started today and see how ViEng AI can help you become fluent in English by providing you with accurate translations, detailed grammar insights, and a better understanding of tenses.
    """)


# Translate Page
def translate_page():
    st.title("🌐 Vietnamese to English Translation")
    user_input = st.text_input("Enter Vietnamese text:")
    
    if user_input:
        tokenizer, model = load_model()
        with st.spinner("Translating..."):
            translation = translate_text(user_input, tokenizer, model)
        
        st.success("Translation:")
        st.write(translation)

        # Store the translated sentence in session state for later use in Analyze tab
        st.session_state.translated_sentence = translation
		
# Analyze Page
def analyze_page():
    st.title("🔍 Analyze the Translated Sentence")    

    # Retrieve the translated sentence from session state
    if "translated_sentence" in st.session_state:
        translated_sentence = st.session_state.translated_sentence
        st.write(f"**Translated Sentence:** {translated_sentence}")

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(translated_sentence)
        
        # Get POS tags for each word
        pos_tags = analyze_sentence(doc)
        
        # Visualize POS tags
        html_analysis = visualize_analysis(pos_tags)
        st.markdown(f'<div>{html_analysis}</div>', unsafe_allow_html=True)

        # Display the entire text with entities and dependencies highlighted
        st.subheader("Full Text Visualization:")
        html_full_text = spacy.displacy.render(doc, style="dep", page=True)
        st.components.v1.html(html_full_text,width=1000, height=500)
        
        # Tense Analysis
        st.subheader("Tense Analysis:")
        sentence_tenses = analyze_tense(doc)

        st.write("**Detected Tenses in the Sentence:**")
        if sentence_tenses:
            for tense in sentence_tenses:
                st.write(tense)
        else:
            st.write("No recognizable tense found.")

    else:
        st.write("No translation available. Please translate a sentence first.")

# Helper functions
def load_model():
    model_name = "Helsinki-NLP/opus-mt-vi-en"  # Vietnamese to English model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(input_text, tokenizer, model):
    translated = model.generate(**tokenizer(input_text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def analyze_sentence(doc):
    # Extract named entities and part-of-speech tags
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def visualize_analysis(pos_tags):
    # Create a list to display words and their POS tags
    html_output = ""
    for word, pos_tag in pos_tags:
        html_output += f"{word}: {pos_tag} <br>"
    return html_output

def analyze_tense(doc):
    tenses = {
        "Past Simple": False,
        "Past Continuous": False,
        "Past Perfect": False,
        "Past Perfect Continuous": False,
        "Present Simple": False,
        "Present Continuous": False,
        "Present Perfect": False,
        "Present Perfect Continuous": False,
        "Future Simple": False,
        "Future Continuous": False,
        "Future Perfect": False,
        "Future Perfect Continuous": False
    }

    # Check for auxiliary verbs and main verb forms for tense detection
    for token in doc:
        # Check for auxiliary verbs: is, am, are (Present Simple), was, were (Past Simple)
        if token.pos_ == "AUX":
            if token.lemma_ in ["be"]:  # 'is', 'am', 'are', 'was', 'were'
                if "Tense=Pres" in token.morph:  # Present Tense: "is", "am", "are"
                    tenses["Present Simple"] = True
                elif "Tense=Past" in token.morph:  # Past Tense: "was", "were"
                    tenses["Past Simple"] = True
            elif token.lemma_ == "will":  # Detect Future Simple: "will" + verb
                tenses["Future Simple"] = True
        
        elif token.pos_ == "VERB":
            if "Tense=Pres" in token.morph:
                tenses["Present Simple"] = True
            elif "Tense=Past" in token.morph:
                tenses["Past Simple"] = True
            # Add Present Continuous detection logic: "is + verb-ing"
            elif token.dep_ == "ROOT" and "Aspect=Prog" in token.morph:
                tenses["Present Continuous"] = True

    sentence_tenses = [tense for tense, present in tenses.items() if present]
    return sentence_tenses

# Sidebar for POS tag explanations
st.sidebar.title("Welcome to ✨	ViEng AI✨ !")
st.sidebar.markdown("---")
st.sidebar.header("Part-of-Speech Tags Explanation")
st.sidebar.write("""
    * **ADJ**: Adjective
    * **ADP**: Adposition (preposition or postposition)
    * **ADV**: Adverb
    * **AUX**: Auxiliary verb
    * **CCONJ**: Coordinating conjunction
    * **DET**: Determiner
    * **INTJ**: Interjection
    * **NOUN**: Noun
    * **NUM**: Numeral
    * **PART**: Particle
    * **PRON**: Pronoun
    * **PROPN**: Proper noun
    * **PUNCT**: Punctuation
    * **SCONJ**: Subordinating conjunction
    * **SYM**: Symbol
    * **VERB**: Verb
    * **X**: Other (catch-all)
""")

# Logic to render the chosen page
if chosen_tab == HOME:
    home_page()

elif chosen_tab == TRANSLATE:
    translate_page()

elif chosen_tab == ANALYZE:
    analyze_page()
