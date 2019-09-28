import glob
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

java_files = glob.glob('languages/java/*.java')

java_codes = []


for filename in java_files:
    text = open(filename).read()
    java_codes.append(text)


python_files = glob.glob('languages/python/*.py')

python_codes = []


for filename in python_files:
    text = open(filename).read()
    python_codes.append(text)


sql_files = glob.glob('languages/sql/*.sql')

sql_codes = []


for filename in sql_files:
    text = open(filename).read()
    sql_codes.append(text)


    
html_files = glob.glob('languages/html/*.html')

html_codes = []


for filename in html_files:
    text = open(filename).read()
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

#chama o naive bayes
classifier = MultinomialNB()
classifier.fit(X_tfdr, Y)


# o que voce vai classificar tem que passar pelo mesmo tratamento do treino

example = [
''''
System.out.println("ola'mundo')
'''
]
# metodo diferente, porque o fit_transform faz o fit e o transform
# o fit ve quantas palavras unicas tem, e saber a posicao de cada palavra
# e o tranform ele dominiu as contagens 
example_count = count_vectorizer.transform(example)
example_tfdf = tfdf.transform(example_count)


result = classifier.predict(example_tfdf)

print(result)

