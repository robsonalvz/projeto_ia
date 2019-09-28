import pickle


with open('./model/classifier', 'rb') as file:
	classifier = pickle.load(file)

with open('./model/tfdf', 'rb') as file:
	tfdf = pickle.load(file)

with open('./model/count_vectorizer', 'rb') as file:
	count_vectorizer = pickle.load(file)


def predict_from_file(filename):
	with open(predictions_folder + filename, 'r', encoding='utf-8') as file:
		content = file.read()

		_prediction = [content]
		_pred_count  = count_vectorizer.transform(_prediction)
		_pred_tfdf = tfdf.transform(_pred_count)

		result = classifier.predict(_pred_tfdf)
		return known_languages.get(result[0], "Prediction errror")

def how_it_works():
	print(f"""
We have trained our Genie to recognize the following languages:
Java, Python, SQL and HTML.
Put any file you want our Genie to predict which language it belongs to in the predictions folder that comes with your version of our Genie.
The file doesn't have to end with the extension of the language you want to predict, you could put a .txt there if you wish for it!
You can try our Genie by choosing the "Predict from file" option!
""")

predictions_folder = './predictions/'

known_languages = {
	0: "Java",
	1: "Python",
	2: "SQL",
    3: "HTML"
}

# o que voce vai classificar tem que passar pelo mesmo tratamento do treino

options = ['1', '2', '3', '0']
guide = """
[1] How it works
[2] Predict from file
[3] Print options
[0] Exit
"""
choice = 'None'
print_guide = True
while choice != '0':
	print("Choose an option: ")
	if print_guide:
		print(guide)
		print_guide = False

	choice = input('> ').lower()

	if(choice not in options):
		print("Invalid option!")
		continue

	if choice == '1':
		how_it_works()
		continue

	if choice == '2':
		print("Inform the exact file name to be predicted: (Including file extension!)")
		filename = input("> ")

		try:
			result = predict_from_file(filename)
		except FileNotFoundError:
			print("Our Genie didn't find any file! :(\nAre you sure there is a file with that name?")
			continue

		print(f"\nOur Genie says your file contains code from the language {result}!\n")
		print_guide = True
		continue

	elif choice == '3':
		print_guide = True
		continue

	elif (choice == '0'):
		print("Our Genie says goodbye! :)")
		break
