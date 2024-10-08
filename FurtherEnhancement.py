import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Use Clustering to Narrow Down Document Search Space
def cluster_documents(tfidf_matrix, doc_ids, n_clusters=10):
    """
        Cluster documents to narrow down the search space.
    """
    kmeans = KMeans(n_clusters= n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    
    # Store cluster assignments for each document
    document_clusters = {doc_id: cluster for doc_id, cluster in zip(doc_ids, kmeans.labels_)}
    return kmeans, document_clusters

# BM25 for Candidate Selection
def bm25_candidate_selection(query, documents, doc_ids, top_n=50):
    """
    Use BM25 to select top candidate documents for a given query.
    """
    tokenized_documents = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top_n documents based on BM25 score, ensuring index is within bounds
    top_n_indices = np.argsort(bm25_scores)[::-1][:min(top_n, len(doc_ids))]
    top_n_docs = [doc_ids[i] for i in top_n_indices if i < len(doc_ids)]
    
    return top_n_docs



# Threshold-Based Document Filtering
def threshold_based_filtering(query_vector, tfidf_matrix, doc_ids, threshold=0.1):
    """
    Filter documents based on a similarity score threshold.
    """
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    filtered_docs = [(doc_ids[i], score) for i, score in enumerate(similarity_scores) if score >= threshold]
    return filtered_docs


# Use Document Frequency to Limit Vocabulary
def limit_vocabulary(documents,doc_ids, min_df=0.01, max_df=0.85):
    """
    Limit vocabulary based on document frequency to improve performance.
    """
    vectorizer_limited = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=ENGLISH_STOP_WORDS)
    limited_tfidf_matrix = vectorizer_limited.fit_transform(documents)
    # Save the new vectorizer and TF-IDF matrix
    with open('./pickle/limited_tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer_limited, f)
    with open('./pickle/limited_tfidf_matrix.pkl', 'wb') as f:
        pickle.dump((limited_tfidf_matrix, doc_ids), f)


def get_candidates(query, vectorizer, tfidf_matrix, documents, doc_ids, kmeans, document_clusters):
    # Step 1: Cluster selection (find the most relevant cluster)
    query_vector = vectorizer.transform([query])
    cluster_label = kmeans.predict(query_vector)[0]
    relevant_docs = [doc_id for doc_id, label in document_clusters.items() if label == cluster_label]

    # Step 2: Use BM25 to further filter top N candidates
    top_bm25_docs = bm25_candidate_selection(query, documents, doc_ids)
    relevant_docs = list(set(relevant_docs).intersection(top_bm25_docs))

    # Step 3: Threshold-based filtering on TF-IDF similarity
    threshold_docs = threshold_based_filtering(query_vector, tfidf_matrix, doc_ids)
    relevant_docs = list(set(relevant_docs).intersection([doc_id for doc_id, _ in threshold_docs]))

    return relevant_docs