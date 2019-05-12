import pandas as pd
import numpy as np
import gensim

from gensim.models import Doc2Vec

from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument

from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from warnings import simplefilter
import nltk
import multiprocessing
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simplefilter(action='ignore', category=FutureWarning)

cores = multiprocessing.cpu_count()
data_frame = pd.read_csv('tagged_plots_movielens.csv')
stop_words = stopwords.words('english')

# Train and test data separated between each other
train = data_frame.iloc[:2000,:]
test = data_frame.iloc[2000:,:]

unnecessary_tokens = ["'s", "``", "''", "n't", "--", "an", "..."]
[unnecessary_tokens.append(x) for x in stop_words]

def textTokenizer(text):
    return list([word.lower() for sentence in nltk.sent_tokenize(text.strip())
     for word in nltk.word_tokenize(sentence) if (len(word) >= 2 and word not in unnecessary_tokens)])


tagged_train_data = train.apply(lambda df: TaggedDocument(words=textTokenizer(str(df['plot'])), tags=[df['tag']]), axis=1)
tagged_test_data = test.apply(lambda df: TaggedDocument(words=textTokenizer(str(df['plot'])), tags=[df['tag']]), axis=1)

model_dbow = Doc2Vec(dm=0, window=30 , vector_size=50, workers=cores, epochs=0)
model_dbow.build_vocab(tagged_train_data)

# train_documents  = utils.shuffle(tagged_train_data)
model_dbow.train(tagged_train_data, total_examples=model_dbow.corpus_count, epochs=20)

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

y_train, X_train = vec_for_learning(model_dbow, tagged_train_data)
y_test, X_test = vec_for_learning(model_dbow, tagged_test_data)

logreg = LogisticRegression(n_jobs=5, solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

sentence = "dragons big snakes castles magic knights elf goblin"

count = 0
for i in tagged_test_data:
    inferred_vector = model_dbow.infer_vector(i.words)
    sims = model_dbow.docvecs.most_similar([inferred_vector], topn=len(model_dbow.docvecs))
    if i.tags[0] == sims[0][0]:
        count += 1

print(count)


inferred_vector = model_dbow.infer_vector(sentence.split())
sims = model_dbow.docvecs.most_similar([inferred_vector], topn=len(model_dbow.docvecs))
print(sims)
print('Accuracy of classification with doc2vec task is {:.2f} percent'.format((accuracy_score(y_test, y_pred))*100))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))