import os
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from query_processing import preprocess_query, query_expansion

# Load the vectorizer and TF-IDF matrix
@st.cache_resource
def load_vectorizer_and_matrix():
    """
    Load the TF-IDF vectorizer and the TF-IDF matrix from pickle files.
    Returns the vectorizer, TF-IDF matrix, and document IDs.
    """
    with open('../pickle/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('../pickle/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix, doc_ids = pickle.load(f)
    return vectorizer, tfidf_matrix, doc_ids

vectorizer, tfidf_matrix, doc_ids = load_vectorizer_and_matrix()

# Load document snippets
@st.cache_resource
def load_document_snippets():
    """
    Load precomputed document snippets from a pickle file.
    Returns a dictionary mapping document IDs to snippets.
    """
    with open('../pickle/doc_snippets.pkl', 'rb') as f:
        doc_snippets = pickle.load(f)
    return doc_snippets

doc_snippets = load_document_snippets()

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("✨ Aurelia Search Engine Evaluation")

    # Search form
    with st.form(key='search_form'):
        query = st.text_input("Enter your search query:")
        submit_button = st.form_submit_button(label='Evaluate')

    if submit_button:
        if query:
            start_time = time.time()

            # Process the query
            preprocessed_query = preprocess_query(query)
            tokens = preprocessed_query.split()

            # Expand the query using synonyms
            expanded_tokens = query_expansion(tokens)
            expanded_query = ' '.join(expanded_tokens)

            # Transform the query using the vectorizer
            query_vector = vectorizer.transform([expanded_query])

            # Compute similarity scores between the query and all documents
            similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

            # Rank documents based on similarity scores
            doc_scores = list(zip(doc_ids, similarity_scores))
            ranked_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)

            end_time = time.time()
            processing_time = end_time - start_time

            # Display evaluation metrics
            display_metrics(query, tokens, expanded_tokens, ranked_docs, similarity_scores, processing_time)
        else:
            st.warning("Please enter a query.")

def display_metrics(query, tokens, expanded_tokens, ranked_docs, similarity_scores, processing_time):
    """
    Display evaluation metrics and visualizations based on the query and retrieved documents.
    """
    st.header("Evaluation Metrics")

    # Number of documents in the corpus
    total_documents = len(doc_ids)

    # Number of documents retrieved (with non-zero similarity score)
    non_zero_scores = [score for score in similarity_scores if score > 0]
    num_documents_retrieved = len(non_zero_scores)

    # Average similarity score
    average_similarity = np.mean(non_zero_scores) if non_zero_scores else 0

    # Display basic metrics
    st.subheader("Basic Metrics")
    st.markdown(f"- **Total Documents in Corpus:** {total_documents}")
    st.markdown(f"- **Documents Retrieved:** {num_documents_retrieved}")
    st.markdown(f"- **Average Similarity Score:** {average_similarity:.4f}")
    st.markdown(f"- **Processing Time:** {processing_time:.4f} seconds")

    # Display query details
    st.subheader("Query Details")
    st.markdown(f"- **Original Query:** {query}")
    st.markdown(f"- **Preprocessed Query Tokens:** {tokens}")
    st.markdown(f"- **Expanded Query Tokens:** {expanded_tokens}")

    # Ask the user to input relevant documents for the query
    st.subheader("Input Relevant Documents")
    st.markdown("Please enter the IDs of the relevant documents separated by commas (e.g., doc1.pdf, doc5.pdf):")
    relevant_docs_input = st.text_input("Relevant Documents:", value="Burj Khalifa.pdf, Eiffel Tower.pdf")
    # relevant_docs_input = 
    if relevant_docs_input:
        relevant_docs = [doc.strip() for doc in relevant_docs_input.split(',')]
        compute_evaluation_metrics(ranked_docs, relevant_docs)
    else:
        st.info("Enter the relevant documents to compute Precision, Recall, F1-Score, MRR, and MAP.")

    # Similarity score distribution visualization
    st.subheader("Similarity Score Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(non_zero_scores, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Histogram of Similarity Scores')
    ax1.set_xlabel('Similarity Score')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)

    # Term frequency visualization
    st.subheader("Term Frequencies")
    query_term_freq = vectorizer.transform([' '.join(expanded_tokens)]).toarray()[0]
    corpus_term_freq = tfidf_matrix.sum(axis=0).A1

    # Get indices of query terms in the vectorizer vocabulary
    vocab = vectorizer.get_feature_names_out()
    query_term_indices = [vectorizer.vocabulary_.get(term) for term in expanded_tokens if term in vectorizer.vocabulary_]

    if query_term_indices:
        # Get frequencies
        query_terms = [vocab[i] for i in query_term_indices]
        query_term_freqs = query_term_freq[query_term_indices]
        corpus_term_freqs = corpus_term_freq[query_term_indices]

        # Create a DataFrame for visualization
        df = pd.DataFrame({
            'Term': query_terms,
            'Query Frequency': query_term_freqs,
            'Corpus Frequency': corpus_term_freqs
        })

        # Bar chart of term frequencies
        fig2, ax2 = plt.subplots()
        bar_width = 0.35
        index = np.arange(len(query_terms))

        ax2.bar(index, df['Query Frequency'], bar_width, label='Query')
        ax2.bar(index + bar_width, df['Corpus Frequency'], bar_width, label='Corpus')

        ax2.set_xlabel('Terms')
        ax2.set_ylabel('TF-IDF Weight')
        ax2.set_title('Term Frequencies in Query vs. Corpus')
        ax2.set_xticks(index + bar_width / 2)
        ax2.set_xticklabels(query_terms, rotation=45)
        ax2.legend()

        st.pyplot(fig2)
    else:
        st.write("No matching terms found in the corpus vocabulary.")

    # Display top N documents by similarity score
    st.subheader("Top Documents by Similarity Score")
    num_top_docs = min(5, len(ranked_docs))
    top_docs = ranked_docs[:num_top_docs]
    doc_table = {
        'Document': [doc_id for doc_id, score in top_docs],
        'Similarity Score': [f"{score:.4f}" for doc_id, score in top_docs]
    }
    st.table(doc_table)

def compute_evaluation_metrics(ranked_docs, relevant_docs):
    """
    Compute and display evaluation metrics based on the ranked documents and user-provided relevant documents.
    """
    # Prepare the list of retrieved documents
    retrieved_docs = [doc_id for doc_id, score in ranked_docs if score > 0]

    # Calculate True Positives, False Positives, False Negatives
    true_positives = len(set(retrieved_docs) & set(relevant_docs))
    false_positives = len(set(retrieved_docs) - set(relevant_docs))
    false_negatives = len(set(relevant_docs) - set(retrieved_docs))

    # Compute Precision, Recall, F1-Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Compute Mean Reciprocal Rank (MRR)
    rank = 0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            rank = i + 1  # Ranks are 1-based
            break
    mrr = 1 / rank if rank > 0 else 0

    # Compute Mean Average Precision (MAP)
    num_relevant_docs = len(relevant_docs)
    num_hits = 0
    sum_precisions = 0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            sum_precisions += precision_at_i
    map_score = sum_precisions / num_relevant_docs if num_relevant_docs > 0 else 0

    # Display the metrics
    st.subheader("Evaluation Metrics Based on Relevance Judgments")
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'MRR', 'MAP'],
        'Score': [f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}", f"{mrr:.4f}", f"{map_score:.4f}"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

if __name__ == '__main__':
    main()