import sys

import numpy as np
from gensim.models import KeyedVectors


# Class of first task.
class Analogy(object):
    # Word2vec model loaded from google's pretrained word vectors.
    word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=200000)
    # All word vectors' magnitudes calculated with help of numpy library.
    vector_magnitudes = np.apply_along_axis(np.linalg.norm, 1, word2vec_model.vectors)

    # This method calculates cosine similarity between test vector and all word vectors.
    # Basic cosine similarity formula applied. (DotProduct / (Magnitude(test) * Magnitude (word)
    # Numpy library helps us while applying dot product for all word vectors.
    # At the end a similarity vector array returned.
    def calcCosineSimilarity(self, test_vector):
        dot_products = np.dot(self.word2vec_model.vectors, test_vector)
        test_vector_magnitude = np.linalg.norm(test_vector)
        similarities = dot_products / (self.vector_magnitudes * test_vector_magnitude)
        return similarities

    # This method used for cleaning unwanted lines in the test file.
    # Also divide sentences into words then return list of lists.
    def organizeAnalogyTestFile(self):
        test_lines = []
        [test_lines.append(t_line.split()) for t_line in open("analogy_small.txt", "r").readlines() if
         len(t_line.strip().split()) == 4]
        return test_lines

    # This method used for comparing result of prediction to given parameters.
    # If analogy prediction same as any of the parameters that we calculated test vector, we look second, third..
    # similar word with respect to test vector until prediction is not same any of the parameters that we calculated
    # test vector at the first time.
    def analogyCorrecter(self, test_words, similarity_calculations, sorted_array, index):
        m = np.where(similarity_calculations == sorted_array[index])
        predicted_word = self.word2vec_model.index2word[m[0][0]]
        if (predicted_word == test_words[0]) or (predicted_word == test_words[1]) or (predicted_word == test_words[2]):
            predicted_word = self.analogyCorrecter(test_words, similarity_calculations, sorted_array, index + 1)
        return predicted_word

    # Main function of this task.
    # Firstly, gets sentences that divided into words list, then calculate the test vector that we used later
    # for analogy prediction.
    # Then cosine similarities calculated with respect to test vector.
    # Most similar word determined with highest similarity.
    # Then prediction is checked whether the word is equal to the given parameters.
    # If prediction is correct, correct counter increased one by one.
    # After all that accuracy calculated with respect to number of test sentences and printed in the console window.
    def findAnalogy(self):
        analogy_test_lines = self.organizeAnalogyTestFile()
        correct_predict, percentage_count = 0, 0
        # todo try excepti kaldir
        for test_words in analogy_test_lines:
            try:
                test_vector = self.word2vec_model[test_words[2]] + self.word2vec_model[test_words[1]] \
                              - self.word2vec_model[test_words[0]]
                similarity_calculations = self.calcCosineSimilarity(test_vector)
                predicted_word = self.word2vec_model.index2word[np.argmax(similarity_calculations)]

                #Checking of prediction with respect to test_words.
                if ((predicted_word == test_words[0]) or (predicted_word == test_words[1]) or (
                        predicted_word == test_words[2])):
                    sorted_array = np.sort(similarity_calculations, 0)[::-1]
                    predicted_word = self.analogyCorrecter(test_words, similarity_calculations, sorted_array, 1)

                if predicted_word == test_words[3]:
                    correct_predict += 1
                percentage_count += 1
            except:
                percentage_count += 1
                continue

            sys.stdout.write("\r%s%d%%" % (
                "Analogy test lines are processing...\t", ((percentage_count / len(analogy_test_lines)) * 100)))
            sys.stdout.flush()
        print("\nAccuracy of analogy task is {:.2f} percent.".format((correct_predict / len(analogy_test_lines)) * 100))
