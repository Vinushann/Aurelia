import os
import pickle
import streamlit as st
from spellchecker import SpellChecker
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity
from query_processing import preprocess_query, query_expansion
from FurtherEnhancement import cluster_documents, get_candidates

# Set the path to your PDF documents
pdf_folder_path = './static/pdfs'

# Set the base URL for PDFs served by the HTTP server
pdf_base_url = 'http://localhost:8502/static/pdfs'


# Spell Checker
spell = SpellChecker()

def correct_query(query):
    corrected_query = " ".join([spell.correction(word) for word in query.split()])
    return corrected_query

# Load the vectorizer and TF-IDF matrix
@st.cache_resource
def load_vectorizer_and_matrix():
    with open('./pickle/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('./pickle/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix, doc_ids = pickle.load(f)
    return vectorizer, tfidf_matrix, doc_ids

vectorizer, tfidf_matrix, doc_ids = load_vectorizer_and_matrix()

# Load document snippets
@st.cache_resource
def load_document_snippets():
    with open('./pickle/doc_snippets.pkl', 'rb') as f:
        doc_snippets = pickle.load(f)
    return doc_snippets

doc_snippets = load_document_snippets()

# Load preprocessed documents
@st.cache_resource
def load_preprocessed_documents():
    with open('./pickle/preprocessed_documents.pkl', 'rb') as f:
        preprocessed_documents = pickle.load(f)
    return preprocessed_documents

documents = load_preprocessed_documents()

"""
    kmeans: model
    document_clusters: {'doc1': 2, 'doc2': 5, 'doc3': 1}
    purpose: it will create cluster for each documents
"""
# Cluster the documents initially
def load_clusters():
    kmeans, document_clusters = cluster_documents(tfidf_matrix, doc_ids)
    return kmeans, document_clusters

kmeans, document_clusters = load_clusters()


def main():
    """
    Purpose: Check the result variable inside the dictionary (session_state is dict)
    """
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    
    st.title("Aurelia ")
    st.text("Travelling Search Engine")

    # Inject JavaScript to handle keyboard shortcuts
    components.html("""
        <script>
            const docSearchInput = window.parent.document.querySelectorAll('input[type=text]')[0];
            window.parent.document.addEventListener('keydown', function(e) {
                // Check if the input field is not focused
                if (document.activeElement !== docSearchInput) {
                    // Keyboard shortcuts
                    // Focus on input field when '/' is pressed
                    if (e.key === '/') {
                        e.preventDefault();
                        docSearchInput.focus();
                    }
                    // Open PDFs with numbers 1 to 5
                    if (['1', '2', '3', '4', '5'].includes(e.key)) {
                        e.preventDefault();
                        const index = parseInt(e.key) - 1;
                        const links = window.parent.document.querySelectorAll('a.pdf-link');
                        if (links.length > index) {
                            window.open(links[index].href, '_blank');
                        }
                    }
                }
            });
        </script>
        """,
        height=0,
    )
    
    # Purpose: Creating a form using Streamlit
    # Key: Helps us uniquely identify the form
    with st.form(key='search_form'):
        query = st.text_input("Enter your search query:", key='search_input')
        # Updated the options to include 'All' at the end
        num_results_options = ['All'] + [str(i) for i in range(1, 11)]
        num_results_str = st.selectbox("Number of results to display:", options=num_results_options, index=0)
        submit_button = st.form_submit_button(label='Search')

    # Purpose: Checking whether submit button is pressed or not
    if submit_button:
        if query:
            # Spell checking
            corrected_query = correct_query(query)
            
            # Process the query
            preprocessed_query = preprocess_query(corrected_query)
            tokens = preprocessed_query.split()

            # Expanded query is in the list
            expanded_tokens = query_expansion(tokens)
            expanded_query = ' '.join(expanded_tokens)

            # Get candidate documents using clustering, BM25, and threshold filtering
            relevant_docs = get_candidates(expanded_query, vectorizer, tfidf_matrix, documents, doc_ids, kmeans, document_clusters)

            # Compute similarity scores for the final list of documents
            query_vector = vectorizer.transform([expanded_query])
            similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
            doc_scores = [(doc_id, similarity_scores[doc_ids.index(doc_id)]) for doc_id in relevant_docs]
            
            ranked_docs = sorted(doc_scores, key= lambda x: x[1], reverse=True)

            # Convert num_results_str to an integer or set to show all results
            if num_results_str == 'All':
                num_results = len(ranked_docs)
            else:
                num_results = int(num_results_str)

            # Store the results in st.session_state
            st.session_state['results'] = {
                'ranked_docs': ranked_docs,
                'num_results': num_results
            }

            display_results = True
        else:
            st.warning("Please enter a query.")
            display_results = False
    else:
        # If not submit_button, check if we have results in session_state
        if st.session_state['results'] is not None:
            ranked_docs = st.session_state['results']['ranked_docs']
            num_results = st.session_state['results']['num_results']
            display_results = True
        else:
            display_results = False

    # Display results if available
    if display_results:
        st.header("Search Results")
        results_found = False

        for idx, (doc_id, score) in enumerate(ranked_docs[:num_results]):
            if score > 0:
                results_found = True
                doc_path = os.path.join(pdf_folder_path, doc_id)
                pdf_url = f"{pdf_base_url}/{doc_id}"

                st.write(f"**Document {idx + 1}:** {doc_id}")
                st.write(f"**Relevance Score:** {score:.4f}")

                # Display snippet
                snippet = doc_snippets.get(doc_id, "")
                st.write(snippet)

                # Create two columns for buttons
                button_col1, button_col2 = st.columns([1, 1])

                with button_col1:
                    # Open PDF button styled with HTML
                    st.markdown(
                        f'<a href="{pdf_url}" target="_blank" class="pdf-link"><button style="width:100%">Open PDF</button></a>',
                        unsafe_allow_html=True
                    )
                with button_col2:
                    # Download PDF button
                    with open(doc_path, 'rb') as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=doc_id,
                        mime='application/pdf'
                    )

                # Add divider
                st.markdown('---')
            else:
                break
        if not results_found:
            st.write("No relevant documents found.")

if __name__ == '__main__':
    main()