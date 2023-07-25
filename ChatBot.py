import nltk
import random
import string
import warnings
warnings.filterwarnings('ignore')

file=open("F:/Code/ML/Internship/ChatBotTask1/queries.txt",'r')
row=file.read()
row=row.lower()

sent_token=nltk.sent_tokenize(row)
word_token=nltk.word_tokenize(row)
#print(sent_token[:2])
#print(word_token[:4])

lemmer=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punc=dict((ord(punc),None) for punc in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc)))

Greeting_Input=("hello","hi","greeting","sup","what's up","hey")
Greeting_Response=["hi","hey","nods","hi there","hello","I am glad! your talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in Greeting_Input:
            return random.choice(Greeting_Response)+"\n"


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    chatbot_response=""
    sent_token.append(user_response)
    tfidfvec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf=tfidfvec.fit_transform(sent_token)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        chatbot_response=chatbot_response+"I am sorry! I don't understand you\n"
        return chatbot_response
    else:
        chatbot_response=chatbot_response+sent_token[idx]+"\n"
        return chatbot_response


if __name__=="__main__":
    flag=True
    print("Hello, There my name is Nirob.I will answer your queries. If you wnat to exit, type bye!")
    while(flag==True):
        user_response=input()
        user_response=user_response.lower()
        if(user_response!="bye"):
            if user_response=="thanks" or user_response=="thank you":
                flag=False
                print("Nirob: you're welcome!")
            else:
                if(greeting(user_response)!=None):
                    print("Nirob:"+greeting(user_response))
                else:
                    print("Nirob:",end="")
                    print(response(user_response))
                    sent_token.remove(user_response)
        else:
            flag=False
            print("Nirob: Bye!, Have a great time!" )
                    
            
    
    
    






