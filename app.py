from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/Predict',methods=['POST'])
def Predict():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	# Features and Labels
	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
	X = df['message']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Learn Vocab and create document-term matrix
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=53) # Train Test Split

	clf = MultinomialNB() # Initializing NaiveBayes Model
	clf.fit(X_train,y_train) # Fit the model
	# clf.score(X_test,y_test)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray() # Create document-term matrix
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)