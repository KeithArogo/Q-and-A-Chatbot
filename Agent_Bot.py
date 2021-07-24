#Import relevant libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import io
import nltk 
import warnings
warnings.filterwarnings('ignore')

#Reading the corpus:
f=open('Dataset.txt','r',errors = 'ignore')
doc=f.read()
doc=doc.lower() #lowercase

#Tokenization:
sentence_tokens = nltk.sent_tokenize(doc)#sentences 
word_tokens = nltk.word_tokenize(doc)#words


#  FUNCTIONS:
#PRE-PROCESSING:
#Handles Lemmatisation of tokens
lemmer = nltk.stem.WordNetLemmatizer()	
def Lemmatize_token(tokens):
	return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

#Handles Lemmatisation of text
def Normalizationn(text):
	return Lemmatize_token(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Function that handles name changes......
def ChangeName():
	print("Agent_Bot:What's your preferred name?") 
	global user_name
	user_name = input() 			
	print("Agent_Bot:Your name has been changed to ",user_name)

#Handles the responses for any small_talk (greetings) (keyword matching)
small_talk_inputs = ("hi", "hello", "sup", "what's up","hey",)
small_talk_responses = ["hey, what do you wanna know?....", "What up? How can I help....", "hi there,what do you need?....", "hello, what can I do you for?....", "I am glad you are talking to me, how can I help?...."]

def small_talk(sentence_tokens):	
	for word in sentence_tokens.split():
		if word.lower() in small_talk_inputs:
			return random.choice(small_talk_responses)

#Handles the responses for any small talk *beyond greetings* (keyword matching)
def smaltalk(sentence_tokens):
	SMALLT_INPUTS = ("hello, how are you?", "what is your name?", "what's my name?",)
	SMALT_RESPONSES = ["I'm fine, thanks for asking....", "My name is Agent Botty....", "your name is "+user_name,]    
	if sentence_tokens.lower() in SMALLT_INPUTS:
		index = SMALLT_INPUTS.index(sentence_tokens)
		return SMALT_RESPONSES[index]			

#Handles the general responses by similarity			
def response(utterance):
	Agent_response=''
	sentence_tokens.append(utterance)
	Tfiidf_vect = TfidfVectorizer(tokenizer=Normalizationn, stop_words='english')
	Tfiidf = Tfiidf_vect.fit_transform(sentence_tokens)
	Values = cosine_similarity(Tfiidf[-1], Tfiidf)
	Indx=Values.argsort()[0][-2]
	flat = Values.flatten()
	flat.sort()
	req_Tfiidf = flat[-2]
	if(req_Tfiidf==0):
		Agent_response=Agent_response+"Agent_Bot :Sorry! I can't understand you, could rephrase your question?....."
		return Agent_response
	else:
		Agent_response = Agent_response+sentence_tokens[Indx] #Returns most likely answer from dataset
		return Agent_response

#________________________________________________________________________________________________________________________

#                                   MAIN BODY OF THE CHAT BOT
#________________________________________________________________________________________________________________________

flag=True

print("Agent_Bot: My name is Agent_Bot......Obviously...... What is your name?. (If you want to end interaction, type Bye)")
user_name = input()

print("Agent_Bot : Hi! "+user_name)

#Execute only while flag=true
while(flag==True):
    utterance = input()
    utterance=utterance.lower()
    if(utterance!='bye'):
        if(utterance=='ok' or utterance=='thank you' ):
            flag=False
            print("Agent_Bot: You are welcome..")
        elif(utterance=='change name' or utterance=='modify name' or utterance== 'edit name'):
             ChangeName()
        elif(small_talk(utterance)!=None):
             print("Agent_Bot: "+small_talk(utterance))
        else:
            if(smaltalk(utterance)!=None):
             print("Agent_Bot: "+smaltalk(utterance)) 
            else:
                print("Agent_Bot: ",end="")
                print(response(utterance))
                sentence_tokens.remove(utterance)
    else:
        flag=False
        print("Agent_Bot: Farewell! take care of yourself..")        