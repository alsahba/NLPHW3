import multiprocessing
from warnings import simplefilter

import gensim
import nltk
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
simplefilter(action='ignore', category=FutureWarning)


class DocumentClassification(object):
    # Number of cores got and csv file read into data frame with help of pandas library.
    cores = multiprocessing.cpu_count()
    data_frame = pd.read_csv('tagged_plots_movielens.csv')
    stop_words = stopwords.words('english')

    # Train and test data separated between each other.
    train = data_frame.iloc[:2000, :]
    test = data_frame.iloc[2000:, :]

    # This method divides documents to sentences, then divides sentences to words.
    def textTokenizer(self, text):
        return list([word.lower() for sentence in nltk.sent_tokenize(text.strip())
                     for word in nltk.word_tokenize(sentence) if
                     (len(word) >= 2 and word not in (self.stop_words))])

    # Sentences and their tags divided from tagged documents and returned.
    def vec_for_learning(self, model, documents):
        tags, sentences = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in documents])
        return tags, sentences

    # Train data and test data created with respect to train/test slices of data frame.
    # Model of doc2vec is created. And vocabulary created.
    # Doc2vec model trained with train data that we created before.
    # Train and test tags and sentences got and created variables.
    # LogisticRegression model is created.
    # Train variables' result points fitted in a line with an optimal equation.
    # Prediction estimation doing with test variables.
    # Accuracy of predictions printed in the console window.
    def classify(self):
        print("\nDocument classification started...")
        tagged_train_data = self.train.apply(
            lambda df: TaggedDocument(words=self.textTokenizer(str(df['plot'])), tags=[df['tag']]),
            axis=1)

        tagged_test_data = self.test.apply(
            lambda df: TaggedDocument(words=self.textTokenizer(str(df['plot'])), tags=[df['tag']]),
            axis=1)

        model_dbow = Doc2Vec(dm=0, window=30, vector_size=50, workers=self.cores)
        model_dbow.build_vocab(tagged_train_data)
        model_dbow.train(tagged_train_data, total_examples=model_dbow.corpus_count, epochs=20)

        y_train, X_train = self.vec_for_learning(model_dbow, tagged_train_data)
        y_test, X_test = self.vec_for_learning(model_dbow, tagged_test_data)
        # logreg = LogisticRegression(n_jobs=self.cores, solver='lbfgs')
        logreg = LogisticRegression(n_jobs=5, solver='lbfgs')

        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        print(
            'Accuracy of classification with doc2vec task is {:.2f} percent.'.format(
                (accuracy_score(y_test, y_pred)) * 100))
