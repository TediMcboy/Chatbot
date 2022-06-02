
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import sys
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
from speech_recognition import Recognizer
import os
import subprocess
from pynput.keyboard import Key, Controller
import speech_recognition as sr
keyboard = Controller()

G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue

r: Recognizer = sr.Recognizer()
mic = sr.Microphone()

d = '/System/Applications'
fd = os.open("Steam.app", os.O_RDWR | os.O_CREAT)



intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))



def say(text):
    subprocess.call(['say', text])

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            return result


def chatbot_response(transcript):


    ints = predict_class(transcript, model)
    res = getResponse(ints, intents)
    rioresponse = str(res)
    say(rioresponse)
    print('\n' + B + rioresponse)



def listen():
    while True:

        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        try:

            transcript = r.recognize_google(audio)
            name = 'my name is'
            savedname = transcript.split()
            print('\n' + G + transcript)

            #commands
            if name in transcript:
                if savedname[-1] == "Teddy":
                    say("I know who you are, you created me")

                else :
                    say("Nice to meet you " + savedname[-1])
                transcript = ''


            if transcript == "open Safari":
                os.system('open ' + '/Applications/Safari.app')
                say("Opening Safari")
            if transcript == "kill yourself":
                say("Not cool man")
                sys.exit()

            elif transcript == "search":
                keyboard.press(Key.tab)

            elif transcript == "new tab":
                keyboard.press(Key.cmd)
                keyboard.press('t')
                keyboard.release(Key.cmd)
                keyboard.release('t')

            elif transcript == "show more":
                keyboard.press(Key.space)
                keyboard.release(Key.space)

            elif transcript == "send":
                say("Off it goes")
                keyboard.press(Key.enter)
                keyboard.release(Key.enter)

            elif transcript == "enter":
                say("boop")
                keyboard.press(Key.enter)

            elif transcript == "type":
                say("What do you want to type")
                with mic as source:
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                #transcript: Union[List[Any], Any] = r.recognize_google(audio)
                keyboard.type(transcript)
            elif transcript == "open Steam":
                os.system('open ' + '/Applications/Steam.app')
                say("Opening Steam")
            elif transcript == "delete":
                keyboard.press(Key.delete)
                keyboard.release(Key.delete)
            elif transcript == "mute":
                keyboard.press(Key.media_volume_mute)

            elif transcript == "volume up":
                keyboard.press(Key.media_volume_up)

            elif transcript == "volume down":
                keyboard.press(Key.media_volume_down)

            elif transcript == "delete all":
                say("deleting all selected text")
                keyboard.press(Key.cmd)
                keyboard.press('a')
                keyboard.release(Key.cmd)
                keyboard.release('a')
                keyboard.press(Key.delete)
                keyboard.release(Key.delete)
            chatbot_response(transcript)
        except sr.UnknownValueError:
            print('\n' + O + "No speech detected")
            continue


listen()






































