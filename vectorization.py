import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Purpose: 
        Load the preprocessed documents pickle file
"""
def load_preprocessed_documents():
    with open('./pickle/preprocessed_documents.pkl', 'rb') as f:
        preprocessed_documents = pickle.load(f)
    # it is gonna return it as dictonary
    return preprocessed_documents

"""
Purpose: 
    Vectorize the preprocessed documents using TF-IDF
Para: 
    dictonary
        {
            "doc_id" = "content"
        }
"""
def vectorize_documents(preprocessed_documents):
    # Create a list of document texts 
    """
        documents:  contain all the contents in one list
        doct_ids:   contain all the documents names in one list
        return:     list of words [word1, word2, word3]
    """
    documents = list(preprocessed_documents.values())
    doc_ids = list(preprocessed_documents.keys())

    # Initialize TfidfVectorizer with appropriate parameters
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Save the vectorizer
    with open('./pickle/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save the tfidf_matrix for later use
    with open('./pickle/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump((tfidf_matrix, doc_ids), f)


if __name__ == '__main__':
    preprocessed_documents = load_preprocessed_documents()
    vectorize_documents(preprocessed_documents)