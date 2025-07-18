import streamlit as st
import arxiv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# LLM explanation model (DistilGPT2)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Function to fetch arXiv papers
def fetch_arxiv_papers(query, max_results=3):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return papers

# Function to summarize paper abstracts
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to generate explanations
def generate_explanation(prompt, max_len=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_len, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("arXiv Domain Expert Chatbot")
st.write("Ask me about advanced topics and recent research papers.")

query = st.text_input("🔍 Enter your research topic or concept query:")

if query:
    st.write(f"Fetching arXiv papers for **{query}**...")
    papers = fetch_arxiv_papers(query, max_results=2)
    
    for p in papers:
        st.subheader(p['title'])
        st.write("[View Paper](%s)" % p['url'])
        st.write("**Abstract:**")
        st.write(p['summary'])
        
        # Summarize abstract
        with st.spinner("Summarizing abstract..."):
            summary = summarize_text(p['summary'])
            st.write("**Summarized Abstract:**")
            st.write(summary)
        
        # Generate explanation
        with st.spinner("Generating concept explanation..."):
            explanation_prompt = f"Explain in simple terms: {p['title']}."
            explanation = generate_explanation(explanation_prompt)
            st.write("💡 **Concept Explanation:**")
            st.write(explanation)
        
        st.markdown("---")
