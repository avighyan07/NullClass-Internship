import streamlit as st
import nltk
import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


st.title(" Extractive Text Summarization Tool")

text = st.text_area("Enter your document text below:", height=300)

def summarize(text, top_n=3):
 
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        sentences = sent_tokenize(text)

    stop_words = set(stopwords.words('english'))

    clean_sentences = []
    for sent in sentences:
      
        sent_clean = re.sub(r'[^a-zA-Z]', ' ', sent).lower()
        sent_clean = ' '.join([word for word in sent_clean.split() if word not in stop_words])
        clean_sentences.append(sent_clean)

    
    if not any(clean_sentences):
        return "Could not generate summary: All sentences were empty after preprocessing."

    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(clean_sentences)

    
    sim_mat = cosine_similarity(vectors)

    
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

   
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

   
    summary = ' '.join([ranked[i][1] for i in range(min(top_n, len(ranked)))])
    return summary

if st.button("Generate Summary"):
    if text.strip():
        summary = summarize(text)
        st.write("### 🧾 Summary:")
        st.success(summary)
    else:
        st.warning(" Please enter some text to summarize.")
