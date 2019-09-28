import glob
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle


java_files = glob.glob('languages/java/*.java')
java_codes = []

print("Getting Java data...")
for filename in java_files:
    text = open(filename, encoding="utf-8").read()
    java_codes.append(text)


python_files = glob.glob('languages/python/*.py')
python_codes = []

print("Getting Python data...")
for filename in python_files:
    text = open(filename, encoding="utf-8").read()
    python_codes.append(text)


sql_files = glob.glob('languages/sql/*.sql')
sql_codes = []

print("Getting SQL data...")
for filename in sql_files:
    text = open(filename, encoding="utf-8").read()
    sql_codes.append(text)


print("Getting HTML data...")
html_files = glob.glob('languages/html/*.html')
html_codes = []

for filename in html_files:
    text = open(filename, encoding='utf-8').read()
    html_codes.append(text)
# realizar a classificacao em listas separadas, cada lista em x corresponde a uma classificacao em y (ex:
# caracteristica1(lista1) = positivo/negativo(lista2) )
# modelo bag of words

# X = caracteristicas do dados de treinamento

X = java_codes + python_codes + sql_codes + html_codes
Y = [0]*34 + [1]*34  + [2]*34 + [3]*34

count_vectorizer = CountVectorizer()

# quantas palavras unicas tem e vai gerar um vetor grande , para cada um dos textos, vai fazer a contagem e gerar um vetor
# o tamnaho de xcount tem que ser 2mil tambem
x_count = count_vectorizer.fit_transform(X)

# diferenciar positivo negativo

# TDDR = diminuiu as contagens em frequencias que aparecem demais, para balancear
tfdf = TfidfTransformer()
X_tfdr = tfdf.fit_transform(x_count)

print("Training Genie...")
#chama o naive bayes
classifier = MultinomialNB()
classifier.fit(X_tfdr, Y)


print("Saving Genie...")
with open('./model/classifier', 'wb') as file:
	pickle.dump(classifier, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('./model/tfdf', 'wb') as file:
	pickle.dump(tfdf, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('./model/count_vectorizer', 'wb') as file:
	pickle.dump(count_vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)
