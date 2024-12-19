import streamlit as st
from main import main

from IPython.core.display import display, HTML
from IPython.display import Markdown

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
        "eve": "#ffb74d",    # Event entities
        "O": "#e0e0e0"       # Non-entity tokens
    }
    
    tokens = text.split()
    markdown_text = ""

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-") or tag.startswith("I-"):
            entity_type = tag.split("-")[1]  # Extract the entity type (e.g., "per", "gpe")
            color = colors.get(entity_type.lower(), "gray")
            markdown_text += f"""
            <span style='background-color:{color}; padding:2px; border-radius:3px; margin-right:3px;'>{
                token
            } <sub style='color:white; font-size:0.7em;'>[{entity_type.upper()}]</sub></span> """
        else:
            markdown_text += f"<span style='margin-right:3px;'>{token}</span> "

    return markdown_text.strip()

st.title("Named Entity Recognition (NER) Demo")
st.write("Input text below to see the entities named:")

user_input = st.text_area("Enter text here", "Elon Musk founded SpaceX in 2002.")

if st.button("Analyze"):
    with st.spinner("Processing..."):
        cleaned_text, prediction = main(user_input)
        st.markdown(markdown_highlight_with_labels(cleaned_text, prediction), unsafe_allow_html=True)

