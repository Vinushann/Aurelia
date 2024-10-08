import streamlit as st

def main():
    # Set the page configuration
    st.set_page_config(
        page_title="Aurelia Search Engine Documentation",
        page_icon="🔍",
        layout="centered",
    )

    # Title and subtitle
    st.title("✨ Welcome to Aurelia Search Engine Documentation")
    st.markdown("""
    An interactive guide to understanding the **Aurelia Search Engine**.
    """)

    # Section 1: What is Aurelia?
    st.header("❓ What is Aurelia?")
    st.markdown("""
    **Aurelia** is an advanced search engine designed to efficiently search and retrieve relevant PDF documents based on user queries. It leverages advanced natural language processing (NLP) techniques and machine learning algorithms to provide accurate and speedy search results.

    **Key Features:**
    - 🔎 **Efficient Search and Retrieval**: Quickly search through a collection of PDF documents.
    - 🤖 **Natural Language Processing**: Utilizes lemmatization and stop word removal for better search accuracy.
    - 🧠 **Query Expansion**: Enhances user queries by adding synonyms using WordNet.
    - 📊 **Document Ranking**: Ranks search results based on TF-IDF weighting and cosine similarity.
    - 🖥️ **Interactive User Interface**: Built with Streamlit for a responsive and user-friendly experience.
    - ⌨️ **Keyboard Shortcuts**: Supports shortcuts for enhanced usability.
    """)

    # Section 2: How Does Aurelia Work?
    st.header("⚙️ How Does Aurelia Work?")
    st.markdown("""
    Aurelia operates by combining advanced NLP preprocessing with efficient search algorithms to deliver relevant search results.

    **Workflow:**
    1. **Document Preprocessing**:
       - 📄 Extracts text from PDF documents using PyPDF2.
       - 🧹 Preprocesses text by converting to lowercase, removing punctuation, tokenizing, removing stop words, and lemmatizing.
    2. **Indexing**:
       - 🗂️ Creates an inverted index for efficient term-to-document mapping.
       - 🧮 Computes TF-IDF vectors for documents.
    3. **Query Processing**:
       - 💬 Accepts user queries and preprocesses them similarly to documents.
       - 🔄 Expands queries by adding synonyms from WordNet.
    4. **Similarity Computation**:
       - 📐 Calculates cosine similarity between the query vector and document vectors.
    5. **Ranking and Retrieval**:
       - 🥇 Ranks documents based on similarity scores.
       - 📋 Presents top-ranked documents to the user.
    """)

    # Section 3: Keyboard Shortcuts
    st.header("⌨️ Keyboard Shortcuts")
    st.markdown("""
    Enhance your productivity with these keyboard shortcuts in Aurelia:

    - **`/`**: Focus on the search input field.
    - **`Enter`**: Submit the search query.
    - **`1` to `5`**: Open the corresponding document from the search results.
    - **`c`**: Clear the search input field.
    """)

    # Section 4: Additional Details
    st.header("📖 Additional Details")
    st.markdown("""
    **Why Choose Aurelia?**
    - ⚡ **Speed**: Returns search results within seconds.
    - 🎯 **Accuracy**: High precision and recall due to advanced NLP techniques.
    - 🌐 **User-Friendly Interface**: Easy to navigate and interact with.
    - 🔒 **Security**: Basic input validation and data security measures implemented.
    """)

    # Footer
    st.markdown("---")
    st.markdown("""

    🌟 **Thank you for using Aurelia!**
    """)

if __name__ == "__main__":
    main()
