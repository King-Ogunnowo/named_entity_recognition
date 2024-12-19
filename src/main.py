import os
import re
import spacy
import pickle
import warnings
import subprocess
import streamlit as st
import numpy as np
import contractions

from spacy.tokens import Span
from pathlib import Path
from spacy import displacy
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

from IPython.core.display import display, HTML
from IPython.display import Markdown

model_path = Path("../named_entity_recognition/models/en_core_web_sm")
nlp = spacy.util.load_model_from_path(model_path)

def load_artifacts():
    one_hot_encoder = pickle.load(open("../models/one_hot_encoder.pkl", 'rb'))
    NER_model = load_model('../models/NER_tensorflow_3_input_model/')
    label_encoder = pickle.load(open("../models/label_encoder.pkl", 'rb'))
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    return one_hot_encoder, NER_model, embedding_model, label_encoder

one_hot_encoder, NER_model, embedding_model, label_encoder = load_artifacts()

def clean_text(text):
    text = ' '.join([contractions.fix(word) for word in text.split()])
    text = re.sub("[^a-zA-Z0-9 ]", "", text)
    return text      

def fetch_pos_tag(text):
    tags = np.array([token.tag_ for token in nlp(text)]).reshape(-1, 1)
    return tags

def generate_vectors(text):
    tokens = text.split()
    token_embeddings = np.array([embedding_model.encode(token) for token in tokens])
    sentence_embeddings = np.array([embedding_model.encode(text) for token in tokens])
    return token_embeddings, sentence_embeddings

def markdown_highlight_with_labels(text, tags):
    """
    Highlights text using BIO tags and displays the entity label alongside the token.
    """
    # Define entity colors
    colors = {
        "per": "#a781f9",    # Person entities
        "tim": "#e59edb",    # Time entities
        "gpe": "#faa419",    # Geopolitical entities
        "geo": "#80e5d9",    # Geographical entities
        "org": "#4ea8de",    # Organizations
        "art": "#d3c8a8",    # Art entities
        "nat": "#81c784",    # Natural entities
        "eve": "#ffb74d",
        "O": "#e0e0e0"# Event entities
    }
    
    tokens = text.split()
    markdown_text = ""

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-") or tag.startswith("I-"):
            entity_type = tag.split("-")[1]  # Extract the entity type (e.g., "per", "gpe")
            color = colors.get(entity_type.lower(), "gray")
            markdown_text += f"""
            <span style='background-color:{color}; padding:2px; border-radius:3px;'>{
                token
            } <sub style='color:white; font-size:0.7em;'>[{entity_type.upper()}]</sub></span>"""
        else:
            markdown_text += f"{token} "

    return Markdown(markdown_text.strip())

    
def main(text):
    cleaned_text = clean_text(text)
    token_embedding, sentence_embedding = generate_vectors(cleaned_text)
    pos_tags = one_hot_encoder.transform(fetch_pos_tag(cleaned_text)).toarray()
    prediction = label_encoder.inverse_transform(np.argmax(NER_model.predict([token_embedding, pos_tags, sentence_embedding]), axis = 1))
    return cleaned_text, prediction
