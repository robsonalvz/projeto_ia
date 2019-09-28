from flask import Flask, request, render_template
import pickle

with open('./model/classifier', 'rb') as file:
	classifier = pickle.load(file)

with open('./model/tfdf', 'rb') as file:
	tfdf = pickle.load(file)

with open('./model/count_vectorizer', 'rb') as file:
	count_vectorizer = pickle.load(file)

known_languages = {
	0: "Java",
	1: "Python",
	2: "SQL",
    3: "HTML"
}

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'GET':
		return render_template('home.html')

	if request.method == 'POST':
		content = request.form.get('predict')
		_prediction = [content]
		_pred_count  = count_vectorizer.transform(_prediction)
		_pred_tfdf = tfdf.transform(_pred_count)

		result = classifier.predict(_pred_tfdf)
		result = known_languages.get(result[0], "Prediction error")

		return render_template('home.html', result=result, content=content)


if __name__ == "__main__":
    app.run(debug=True)
