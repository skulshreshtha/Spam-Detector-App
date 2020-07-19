from flask import Flask,render_template,url_for,request
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


app = Flask(__name__)

# Adding in code from the Jupyter for training the model
sms = pd.read_csv("spam.csv", encoding="latin-1")
sms.columns = ['label', 'message']
# Features and Labels
sms['label_num'] = sms['label'].map({'ham': 0, 'spam': 1})
	
def text_process(message):
	"""
	Takes in a string of text and performs the following steps:
	1. Remove punctuation
	2. Tokenize
	3. Remove stopwords
	4. Stems words to their root forms
	Return type: string
	Returns: String of cleaned & stemmed message
	"""
	STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
	# Check characters to see if they are in punctuation
	nopunc = [char for char in message if char not in string.punctuation]

	# Join the characters again to form the string
	nopunc = ''.join(nopunc)
	
	# Instantiating a PorterStemmer object
	porter = PorterStemmer()
	token_words = word_tokenize(nopunc)
	stem_message=[]
	for word in token_words:
		stem_message.append(porter.stem(word))
		stem_message.append(" ")
	return ''.join(stem_message)

sms['clean_message'] = sms['message'].apply(text_process)

X = sms.clean_message
y = sms.label_num

pipe = Pipeline([('bow', CountVectorizer()), 
				 ('tfid', TfidfTransformer()),  
				 ('model', LogisticRegression(solver='liblinear'))])

pipe.fit(X, y)
	

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/Predict',methods=['POST'])
def Predict():
	
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction = pipe.predict(data)
	return render_template('result.html',prediction = my_prediction)

app.run(debug=True)