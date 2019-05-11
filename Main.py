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

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()




