import os
import nltk
import glob #Used to find all file paths matching a pattern, like all PDF files in a folder.
import string
import pickle # A library to save and load Python objects (e.g., dictionaries, lists)
import PyPDF2 # Libraries to extract text from PDF files.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pdfminer.high_level import extract_text # Libraries to extract text from PDF files.


# NLTK library
"""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet') #lemmatizing
"""

#COMMENED CODE
"""def extract_text_from_pdfs(pdf_folder_path):
    pdf_files = glob.glob(os.path.join(pdf_folder_path, '*.pdf'))
    documents = {}

    for pdf_file in pdf_files:
        text = extract_text(pdf_file)
        doc_id = os.path.basename(pdf_file)
        
        if not text.strip():
            print(f"Document {doc_id} is empty after text extraction.")
        else:
            print(f"Document {doc_id} has content after text extraction.")
        documents[doc_id] = text
    return documents
"""

# ORGINAL CODE: 1
"""
Purpose: 
        Extract the texts from the pdf
        store them in dictonary
                {
                    "sri lanka.pdf": "the content"
                }
"""
def extract_text_from_pdfs(pdf_folder_path):
    # Find all PDF files in the specified folder
    pdf_files = glob.glob(os.path.join(pdf_folder_path, '*.pdf'))
    
    # Create an empty dictionary to store the text content of each PDF
    documents = {}

    # Loop through each PDF file found in the folder
    for pdf_file in pdf_files:
        # Extract the text content from the current PDF file using the extract_text function
        text = extract_text(pdf_file)
        
        # Get the name of the PDF file name with extension
        doc_id = os.path.basename(pdf_file)

# ?strip()
        # Check if the extracted text is empty (after removing any leading/trailing whitespace)
        if not text.strip():
            # If the text is empty, print a message indicating that the PDF has no content
            print(f"Document {doc_id} is empty after text extraction.")
        else:
            # If the text has content, print a message confirming the content was extracted
            print(f"Document {doc_id} has content after text extraction.")
        
        # Add the extracted text to the documents dictionary with the file name as the key
        documents[doc_id] = text
 
    # Return the dictionary containing the extracted text for all PDFs
    return documents


"""
    Purpose: Preprocess the document
"""
def preprocess_text(text):
    
    text = text.lower()
    # Step 1: Lowercase Text: the runners were running quickly towards the finish-line!
    # ?
    punctuations = string.punctuation.replace('-', '') 
    #Step 2: Remove Punctuation: the runners were running quickly towards the finish-line

    # ?
    text = text.translate(str.maketrans('', '', punctuations))
    
    tokens = nltk.word_tokenize(text)
    # Step 3: Tokenization: ['the', 'runners', 'were', 'running', 'quickly', 'towards', 'the', 'finish-line']

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Step 4: Remove Stopwords: ['runners', 'running', 'quickly', 'towards', 'finish-line']
    
    #Step 5: Lemmatization: ['runner', 'running', 'quickly', 'towards', 'finish-line']
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    #Step 6: Final Preprocessed Text: runner running quickly towards finish-line
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# In preprocess_documents.py
def preprocess_documents(documents):

    preprocessed_docs = {}
    doc_snippets = {}

    for doc_id, text in documents.items():
        preprocessed_text = preprocess_text(text)
        preprocessed_docs[doc_id] = preprocessed_text
        # Save the first 100 words of the original text
        snippet_words = text.split()[:100]
        snippet = ' '.join(snippet_words)
        doc_snippets[doc_id] = snippet

    # Save snippets
    with open('./pickle/doc_snippets.pkl', 'wb') as f:
        pickle.dump(doc_snippets, f)

    return preprocessed_docs

if __name__ == '__main__':
    pdf_folder_path = './static/pdfs'

    documents = extract_text_from_pdfs(pdf_folder_path)
    print()
    preprocessed_documents = preprocess_documents(documents)
    
    # Save preprocessed documents for later use
    with open('./pickle/preprocessed_documents.pkl', 'wb') as f:
        pickle.dump(preprocessed_documents, f)