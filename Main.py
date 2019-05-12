import numpy as np
from gensim.models import KeyedVectors


word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True, limit=300000)
vector_magnitudes = np.apply_along_axis(np.linalg.norm, 1, word2vec_model.vectors)

def calculateCosine(test_vector):
    dot_products = np.dot(word2vec_model.vectors, test_vector)
    test_vector_magnitude = np.linalg.norm(test_vector)
    similarities = dot_products / (vector_magnitudes * test_vector_magnitude)
    return similarities

def organizeAnalogyTestFile():
    test_lines = []
    [test_lines.append(t_line.split()) for t_line in open("analogy_small.txt", "r").readlines() if len(t_line.strip().split()) == 4]
    return test_lines


def analogyCorrecter(test_words, similarity_calculations, sorted_array, index):
    m = np.where(similarity_calculations == sorted_array[index])
    predicted_word = word2vec_model.index2word[m[0][0]]
    if (predicted_word == test_words[0]) or (predicted_word == test_words[1]) or (predicted_word == test_words[2]):
        predicted_word = analogyCorrecter(test_words, similarity_calculations,sorted_array, index + 1)
    return predicted_word


def analogyTask():
    analogy_test_lines = organizeAnalogyTestFile()
    correct_predict = 0
    # todo try excepti kaldir
    for test_words in analogy_test_lines:
        try:
            test_vector = word2vec_model[test_words[2]] + word2vec_model[test_words[1]] - word2vec_model[test_words[0]]
            similarity_calculations = calculateCosine(test_vector)
            predicted_word = word2vec_model.index2word[np.argmax(similarity_calculations)]
            if ((predicted_word == test_words[0]) or (predicted_word == test_words[1]) or (predicted_word == test_words[2])):
                sorted_array = np.sort(similarity_calculations, 0)[::-1]
                predicted_word = analogyCorrecter(test_words, similarity_calculations, sorted_array, 1)

            if predicted_word == test_words[3]:
                correct_predict = correct_predict + 1
        except:
            print("henak")

    print("Accuracy of analogy task is {}".format((correct_predict/len(analogy_test_lines)) * 100))


# analogyTask()
