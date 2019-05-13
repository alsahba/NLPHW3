from Analogy import Analogy
from DocumentClassification import DocumentClassification


# WARNING!!!
# Without google word vectors this program will be crash.
# So, before you run this program please put 'GoogleNews-vectors-negative300.bin' file into same directory.
# Since github refused nearly 4GB file while pushing, i could not add it to submit.

analogy = Analogy()
doc_classification = DocumentClassification()

analogy.findAnalogy()
doc_classification.classify()
