import os
import math
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
nltk.download("punkt")


def read_text_file(file_path):
    """Read the contents of a text file."""
    with open(file_path, 'r') as file:
        return file.read()


def calculate_similarity(tfidf_matrix, query_vector):
    print('entered value',tfidf_matrix)
    print('entered value',query_vector)
    # numpy_array = np.array( tfidf_matrix.toarray())
    query_vector = np.array(query_vector)
    
    # if tfidf_matrix.shape[1] == query_vector.shape[0]:
    #     similarity = calculate_similarity(numpy_array, query_vector)
    #     print('\nSimilarity:\n', similarity)
    # else:
    #     print('Incompatible shapes for cosine similarity calculation.')
    """
    Calculate cosine similarity between the query vector and each document in the TF-IDF matrix.

    Parameters:
    - tfidf_matrix: The TF-IDF matrix (scipy.sparse.csr_matrix).
    - query_vector: The TF-IDF vector representation of the query (numpy array).

    Returns:
    - similarities: A list of cosine similarities between the query and each document.
    """
    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix, query_vector.reshape(1, -1))

    # Flatten the similarities array
    similarities = similarities.flatten()

    return similarities

def tokenize_and_remove_stopwords(text, stop_words):
    """Tokenize text and remove stopwords. and stemming  """
    tokens = word_tokenize(text)
    tokens= [word for word in tokens if word not in stop_words]
    #Stemming
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return tokens



def build_positional_index(documents):
    """Build a positional index from a list of documents."""
    positional_index = {}
    document_number = 1

    for document in documents:
        for positional, term in enumerate(document):
            if term in positional_index:
                positional_index[term][0] += 1

                if document_number in positional_index[term][1]:
                    positional_index[term][1][document_number].append(positional)
                else:
                    positional_index[term][1][document_number] = [positional]
            else:
                positional_index[term] = [1, {document_number: [positional]}]

        document_number += 1

    return positional_index


def get_term_frequency(document, all_words):
    """Calculate term frequency for a document."""
    words_found = dict.fromkeys(all_words, 0)
    for word in document:
        words_found[word] += 1
    return dict(sorted(words_found.items()))


def get_weighted_term_frequency(x):
    """Calculate weighted term frequency."""
    if x > 0:
        return int(math.log(x) + 1)
    return 0


def calculate_inverse_document_frequency(term_freq):
    """Calculate inverse document frequency."""
    tfd = pd.DataFrame(columns=('df', 'idf'))
    for i in range(len(term_freq)):
        frequency = term_freq.iloc[i].values.sum()
        tfd.loc[i, 'df'] = frequency
        tfd.loc[i, 'idf'] = math.log10(10 / float(frequency))
    tfd.index = term_freq.index
    return tfd


def calculate_normalized_term_freq_idf(term_freq_inve_doc_freq, document_length):
    """Calculate normalized term frequency-inverse document frequency."""
    normalized_term_freq_idf = pd.DataFrame()
    for column in term_freq_inve_doc_freq.columns:
        normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(
            lambda x: x / document_length[column ].values[0] if document_length[column ].values[
                                                                               0] != 0 else 0
        )
    return normalized_term_freq_idf


def handle_query(positional_index, normalized_term_freq_idf, tfd, query):
    """Handle a search query."""
    documents_found = query_search(positional_index, query)
    if not documents_found:
        print('No matching documents found.')
        return

    query_df = create_query_dataframe(normalized_term_freq_idf, query)


    product = calculate_product(normalized_term_freq_idf, query_df)
    query_df['idf'] = tfd['idf'] * query_df['w_tf']
    query_df['tf_idf'] = query_df['w_tf'] * query_df['idf']
    query_df['norm'] = query_df['idf'] / math.sqrt(sum(query_df['idf'].values ** 2))

    ## add sum of query product to dataframe
    # query_product['sum'] = query_product.sum(axis=1)
    print('\nQuery\n', query_df.loc[query.split()])
    # print('\nQuery product\n', query_product.loc[query.split()])

    product2 = product.multiply(query_df['norm'], axis=0)
    query_length = math.sqrt(sum([x ** 2 for x in query_df['idf'].loc[query.split()]]))
    print('\nQuery length:\n', query_length)
    print('\nProduct:\n', product2)
    scores = calculate_scores(product2, documents_found)
    final_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for doc, score in final_scores:
        print(f" similarity (q , {doc}) " ,score)
    print('\nReturned docs:')
    for doc, score in final_scores:
        print(doc, end=" ", flush=True)

def query_search_boolean(positional_index, query):
    """Search for documents based on boolean queries."""
    terms = query.split()
    stack = []
    operator = None

    for term in terms:
        if term.upper() in ["AND", "OR", "NOT"]:
            operator = term.upper()
        else:
            if term in positional_index.keys():
                if operator == "NOT":
                    # Exclude documents containing the term for NOT operation
                    stack.append(set(range(1, 11)) - set(positional_index[term][1].keys()))
                else:
                    # Add documents containing the term for AND and OR operations
                    stack.append(set(positional_index[term][1].keys()))

    if not stack:
        return []

    # Perform boolean operations
    if operator == "AND":
        result = set.intersection(*stack)
    elif operator == "OR":
        result = set.union(*stack)
    else:
        result = stack[0]

    # Convert document numbers to strings
    positions = [f'doc{doc}' for doc in sorted(result)]
    return positions


def query_search(positional_index, query):
    """Search for documents containing all terms in the query."""
    matching_docs = [[] for _ in range(10)]
    print(query.split())
    for term in query.split():

        if term in positional_index.keys():
            for doc_id in positional_index[term][1].keys():
                if matching_docs[doc_id - 1]:
                    if matching_docs[doc_id - 1][-1] == positional_index[term][1][doc_id][0] - 1:
                        matching_docs[doc_id - 1].append(positional_index[term][1][doc_id][0])
                else:
                    matching_docs[doc_id - 1].append(positional_index[term][1][doc_id][0])

    positions = []
    for pos, positions_list in enumerate(matching_docs, start=1):
        if len(positions_list) == len(query.split()):
            positions.append(f'doc{pos}')
    return positions


def create_query_dataframe(normalized_term_freq_idf, query):
    """Create a DataFrame for the search query."""
    query_df = pd.DataFrame(index=normalized_term_freq_idf.index)
    query_df['tf'] = [1 if x in query.split() else 0 for x in list(normalized_term_freq_idf.index)]
    query_df['w_tf'] = query_df['tf'].apply(lambda x: math.log10(max(1, x)) + 1)
    ## add last row sum of query
    query_df.loc['sum'] = query_df.sum(axis=0)

    return query_df


def calculate_product(normalized_term_freq_idf, query_df):
    """Calculate the product of normalized term frequency-inverse document frequency and query."""
    product = normalized_term_freq_idf.multiply(query_df['w_tf'], axis=0)
    return product


def calculate_scores(product, documents_found):
    """Calculate cosine similarity scores for the matching documents."""
    scores = {}
    for col in documents_found:
        scores[col] = product[col].sum()
    return scores


def main():
    stop_words = set(stopwords.words('english')) - {'in', 'to', 'where'}
    directory = './data'
    documents = []

    docs= os.listdir(directory)
    print(docs)
    for doc in docs:
        document_text = read_text_file(doc)
        document_tokens = tokenize_and_remove_stopwords(document_text, stop_words)
        documents.append(document_tokens)

    positional_index = build_positional_index(documents)

    all_words = [word for doc in documents for word in doc]

    term_freq = pd.DataFrame(get_term_frequency(documents[0], all_words).values(),
                             index=get_term_frequency(documents[0], all_words).keys())
    for i in range(1, len(documents)):
        term_freq[i] = get_term_frequency(documents[i], all_words).values()

    term_freq.columns = [f'doc{i}' for i in range(1, 11)]

    print('\nTerm frequency:\n', term_freq)

    term_freq_weighted = term_freq.applymap(get_weighted_term_frequency)
    print('\nWeighted term frequency (log(tf) + 1):\n', term_freq_weighted)

    tfd = calculate_inverse_document_frequency(term_freq)
    print('\nInverse document frequency:\n', tfd)

    term_freq_inve_doc_freq = term_freq_weighted.multiply(tfd['idf'], axis=0)
    print('\nTerm frequency-inverse document frequency (tf-idf):\n', term_freq_inve_doc_freq)

    document_length = pd.DataFrame()

    for column in term_freq_inve_doc_freq.columns:
        # column.replace('doc', '')
        document_length.loc['length', f'{column}'] = math.sqrt(
            term_freq_inve_doc_freq[column].apply(lambda x: x ** 2).sum())

    print('\nDocument length:\n', document_length.transpose())

    normalized_term_freq_idf = calculate_normalized_term_freq_idf(term_freq_inve_doc_freq, document_length)
    print('\nNormalized tf-idf:\n', normalized_term_freq_idf)

    query = input('\nEnter query:\n')
    query = ' '.join(tokenize_and_remove_stopwords(query, stop_words))
    print(query)
    # query=
    handle_query(positional_index, normalized_term_freq_idf, tfd, query)


if __name__ == "__main__":
    main()
