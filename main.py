import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import nltk
import numpy as np
from nltk.corpus import stopwords
nltk.download("punkt")
import numpy as np

file1 = open("/Users/moaazsalama/Downloads/yarab/1.txt", 'r')
file3 = open("/Users/moaazsalama/Downloads/yarab/3.txt", 'r')
file2 = open("/Users/moaazsalama/Downloads/yarab/2.txt", 'r')
file4 = open("/Users/moaazsalama/Downloads/yarab/4.txt", 'r')
file5 = open("/Users/moaazsalama/Downloads/yarab/5.txt", 'r')
file6 = open("/Users/moaazsalama/Downloads/yarab/6.txt", 'r')
file7 = open("/Users/moaazsalama/Downloads/yarab/7.txt", 'r')
file8 = open("/Users/moaazsalama/Downloads/yarab/8.txt", 'r')
file9 = open("/Users/moaazsalama/Downloads/yarab/9.txt", 'r')
file10 = open("/Users/moaazsalama/Downloads/yarab/10.txt", 'r')


#import nltk
from nltk.tokenize import word_tokenize

text1 = file1.read()
text1_tokens=word_tokenize(text1)

text2 = file2.read()
text2_tokens=word_tokenize(text2)

text3 = file3.read()
text3_tokens=word_tokenize(text3)

text4 = file4.read()
text4_tokens=word_tokenize(text4)

text5 = file5.read()
text5_tokens=word_tokenize(text5)

text6 = file6.read()
text6_tokens=word_tokenize(text6)

text7 = file7.read()
text7_tokens=word_tokenize(text7)

text8 = file8.read()
text8_tokens=word_tokenize(text8)

text9 = file9.read()
text9_tokens=word_tokenize(text9)

text10 = file10.read()
text10_tokens=word_tokenize(text10)



#nltk.download('stopwords')
#from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) - set(['in', 'to','where'])
print(stop_words)




t1=[word for word in text1_tokens if word not in stop_words]
t2=[word for word in text2_tokens if word not in stop_words]
t3=[word for word in text3_tokens if word not in stop_words]
t4=[word for word in text4_tokens if word not in stop_words]
t5=[word for word in text5_tokens if word not in stop_words]
t6=[word for word in text6_tokens if word not in stop_words]
t7=[word for word in text7_tokens if word not in stop_words]
t8=[word for word in text8_tokens if word not in stop_words]
t9=[word for word in text9_tokens if word not in stop_words]
t10=[word for word in text10_tokens if word not in stop_words]


documents = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10]
print('TOKENIZED DOCUMENTS: ')
for file in documents:
    print(file)

print('**********************************************************************************')


document_number = 1
positional_index = {}

for document in documents:

    # For position and term in the tokens.
    for positional, term in enumerate(document):
        # print(pos, '-->' ,term)

        # If term already exists in the positional index dictionary.
        if term in positional_index:

            # Increment total freq by 1.
            positional_index[term][0] = positional_index[term][0] + 1

            # Check if the term has existed in that DocID before. ( exisit in the same doc or not)
            if document_number in positional_index[term][1]:
                # if the same doc add only poisiton
                positional_index[term][1][document_number].append(positional)


            else:  # else it is new doc the define new positional index
                 positional_index[term][1][document_number] = [positional]

        # If term does not exist in the positional index dictionary
        # (first encounter).
        else:

            # Initialize the list.  ==> define empty dictionary then define it by add word {}
            #ex { 'antony' : []}
            positional_index[term] = []

            # The total frequency is 1. ==> add one in list of word antony mean it is exisit
            # ex { 'antony' : [1]}
            positional_index[term].append(1)


            # The postings list is initially empty.  ==> define new dictionary to add its document and position
            # ex { 'antony' : [1] , {dic of doc and pos}}
            positional_index[term].append({})

            # Add doc ID to postings list.
            positional_index[term][1][document_number] = [positional]

    # Increment the file no. counter for document ID mapping
    document_number += 1
print('POSITIONAL INDEX : ')
print(positional_index)

# import warnings
# warnings.filterwarnings("ignore")


# query = 'fools fear'
#positional_index = [
#   'fools' : [4, {7:[1] ,8:[1], 9[1], 10[0] }],
#    'fear' : [3, {7:[2] , 8:[2], 10[1] }]
# ]
def Query(q):
    lis = [[] for i in range(10)]
    for term in q.split():
        if term in positional_index.keys():
        #term=>freq in positional_index   [1 ==> positing list that contain doc] .key(doc that contain term)
            for key in positional_index[term][1].keys():
                if lis[key - 1] != []:
                    if lis[key - 1][-1] == positional_index[term][1][key][0] - 1:
                        # query = 'fools fear'
                        # positional_index = [
                        #   'fools' : [4, {7:[1] ,8:[1], 9[1], 10[0] }],
                        #    'fear' : [3, {7:[2] , 8:[2], 10[1] }]
                        # ] so if i sub it to 1 they will equal so i will append it
                        lis[key - 1].append(positional_index[term][1][key][0])
                else:                                                  #[0] first position in first matched doc
                    lis[key - 1].append(positional_index[term][1][key][0])
    positions = []
    for pos, list in enumerate(lis, start=1):
        if len(list) == len(q.split()):
            positions.append('doc' + str(pos))
    return positions


# ---------------------------------------------------------------------------------------
all_words = []
for doc in documents:
    for word in doc:
        all_words.append(word)


def get_term_freq(doc):
    words_found = dict.fromkeys(all_words, 0)
    for word in doc:
        words_found[word] += 1
    return dict(sorted(words_found.items()))
    #return words_found


term_freq = pd.DataFrame(get_term_freq(documents[0]).values(), index=get_term_freq(documents[0]).keys())
for i in range(1, len(documents)):
    term_freq[i] = get_term_freq(documents[i]).values()
term_freq.columns = ['doc' + str(i) for i in range(1, 11)]
print('**********************************************************************************')

print('                             Term frequency')
print(term_freq, "\n")


def get_weighted_term_freq(x):
    if x > 0:  # as log 0 = o problem
        return int(math.log(x) + 1)
    return 0

print('**********************************************************************************')

print('         Weighted term frequency =log(tf)+1')
for i in range(1, len(documents)+1):
    term_freq['doc' + str(i)] = term_freq['doc' + str(i)].apply(get_weighted_term_freq)
print(term_freq, "\n")
print('**********************************************************************************')

print('          inverse document frequency')

tfd = pd.DataFrame(columns=('df', 'idf'))
for i in range(len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()
    # ==> iloc[i] return position of word in frame ex iloc[o] =>row 0    tfd.loc[i, 'df'] = frequency
    # ==> then sum all rows
    # tfd.loc[i, 'idf'] = frequency
    tfd.loc[i, 'df'] = frequency
    tfd.loc[i, 'idf'] = math.log10(10 / float(frequency))

tfd.index = term_freq.index
print(tfd)
print('**********************************************************************************')

print('            tf-idf')
term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)   #axis=0 => multply rows together
print(term_freq_inve_doc_freq)
print('**********************************************************************************')

print('          Document length')
document_length = pd.DataFrame()

## take sqrt for every row for every doc
def get_docs_length(col):
    return np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x ** 2).sum())


for column in term_freq_inve_doc_freq.columns:
    document_length.loc['length', column + '_length'] = get_docs_length(column)
                                # =>column + _length is name of doc by column &length

print('**********************************************************************************')

print('\n', document_length)
print('         Normalized tf-idf')

## problem  pass to every value in tf-idf and divie to its length
## sometimes length is zero so it is error to solve it ..
##by
normalized_term_freq_idf = pd.DataFrame()


                    ## we will pass every col and doc
def get_normalized(col, x):
    try:
        return x / document_length[col + '_length'].values[0]    # ==> values is return as array so you write is in this way
                             # =>column + _length is name of doc by column &length
    except:   ## if it zero
        return 0


for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column] = term_freq_inve_doc_freq[column].apply(lambda x: get_normalized(column, x))

print(normalized_term_freq_idf)
# ---------------------------------------------------------------------------------
print('**********************************************************************************')

print('          Final result ')


def returneed(q):
    documents_found = Query(q)
    if documents_found == []:
     return print ('miss search')
    query = pd.DataFrame(index=normalized_term_freq_idf.index)
    query['tf'] = [1 if x in q.split() else 0 for x in list(normalized_term_freq_idf.index)]

    def get_w_tf(x):
        try:
            return math.log10(x) + 1
        except:
            return 0

    query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
    product = normalized_term_freq_idf.multiply(query['w_tf'], axis=0)
    query['idf'] = tfd['idf'] * query['w_tf']
    query['tf_idf'] = query['w_tf'] * query['idf']
    query['norm'] = 0
    for i in range(len(query)):
        query['norm'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values ** 2))
    print('\nQuery\n', query.loc[q.split()])
    product2 = product.multiply(query['norm'], axis=0)
    query_length = math.sqrt(sum([x ** 2 for x in query['idf'].loc[q.split()]]))
    print('\nquery length :\n', query_length)
    # #--------------------------------------------------------------------------
    scores = {}
    for col in documents_found:
         # product2.columns
         #if 0 in product2[col].loc[q.split()].values:
          #   pass
         #else:
        scores[col] = product2[col].sum()
    # #------------------------------------------------------------------
    product_result = product2[list(scores.keys())].loc[q.split()]
    print('\nproduct (query*match doc)')
    print('\n', product_result)
    print('\nproduct sum')
    print(product_result.sum())
    print('\ncosine similarity:')

    print(product_result.sum())
    final_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print('\nreturned docs')
    for typle in final_scores:
        print(typle[0], end=" ")


que = input('enter query:\n')
returneed(que)