
import streamlit as st
from transformers import AutoTokenizer, TFT5ForConditionalGeneration, pipeline

# 1. Set the title of the Streamlit application
st.title("Text Summarization App (T5-small)")

# 2. Load the t5-small model and tokenizer (cached for efficiency)
@st.cache_resource
def load_summarizer_model():
    t5_id = "t5-small"
    t5_tok = AutoTokenizer.from_pretrained(t5_id)
    t5_tf = TFT5ForConditionalGeneration.from_pretrained(t5_id, from_pt=True)
    summarizer_pipeline = pipeline(
        task="summarization",
        model=t5_tf,
        tokenizer=t5_tok,
        framework="tf"
    )
    return summarizer_pipeline

summarizer = load_summarizer_model()

# 3. Create a text area widget for user input
article_input = st.text_area(
    "Enter Text to Summarize",
    "Transformers use self-attention to understand relationships between words. This lets them capture long-range context and generate coherent outputs. They power modern NLP tasks like chatbots, translation, and summarization.",
    height=200
)

# 4. Add number input widgets for max_length and min_length
max_len = st.number_input("Max Summary Length", min_value=10, max_value=200, value=60, step=5)
min_len = st.number_input("Min Summary Length", min_value=5, max_value=100, value=15, step=5)

# 5. Create a button to trigger summarization
if st.button("Summarize"):
    if article_input:
        st.info("Summarizing...")
        try:
            summary_output = summarizer(article_input, max_length=max_len, min_length=min_len, do_sample=False)
            st.success("Original Text:")
            st.write(article_input)
            st.success("Summary:")
            st.write(summary_output[0]["summary_text"])
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
    else:
        st.warning("Please enter some text to summarize.")

st.markdown("--- Source: `t5-small` model from Hugging Face Transformers. --- ")
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("Streamlit app code saved to app.py")
