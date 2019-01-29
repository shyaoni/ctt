from argparse import ArgumentParser

from flask import Flask, request, jsonify
app = Flask(__name__)
from datetime import datetime

from copy import copy
from random import randint

import api

testStacks = {}

usrStacks = {}
date_format = "%Y-%m-%d'T'%H%-%M-%S.SSS"

usr_stacks = {}
sess_stack = {}
bot_stack = {}

bot_name = {}
bot = {}
bot_un = {}
end = {}
topic = {}
start_uttr = {}
sess = {}

start_uttrs = ['hello! how are you today?']
bots = [api.chatbot('config.att')]

topic_pool = ['work']

def random_from(lst):
    return lst[randint(0, len(lst)-1)]

@app.route('/<uuid>', methods=['GET', 'POST'])
def add_message(uuid):
    if uuid == "init": #Create a new session
        content = request.json
        #A message from DialPort: sessionID, timeStamp

        sessionID = content["sessionID"]
        timeStamp = content["timeStamp"]
        userID = content['userID']

        # Assign an instance to Dictionary
        #usrStacks[sessionID] = API()

        #A message to DialPort: sessionID, version, terminal, sys, and imageurl
        Output = {}
        Output["sessionID"] = sessionID # DialPort will send sessionID.
        Output["version"] = "0.1" # Please provide your system version.
        Output["terminal"] = False
        Output["timeStamp"] = datetime.now().isoformat()

        if userID not in usr_stacks:
            usr_stacks[userID] = True
            bot_name[userID] = {}
            bot[userID] = {}
            bot_un[userID] = copy(bots)
            end[userID] = {}
            topic[userID] = random_from(topic_pool)
            start_uttr[userID] = random_from(start_uttrs)

        end[userID][sessionID] = False

        if sessionID not in bot[userID]:
            bot[userID][sessionID] = random_from(bot_un[userID])
            bot_name[userID][sessionID] = bot[userID][sessionID].name

        sess[sessionID] = api.sess(bot[userID][sessionID])

        Output['sys'] = start_uttr[userID]
        sess[sessionID].step('chatbot: ' + Output['sys'])

        return jsonify(Output)

    if uuid == "next": #Get your system's next response
        content = request.json
        #A message from DialPort: sessionID, text (an user input), asrConf, timeStamp
        sessionID = content["sessionID"]
        userID = content['userID']
        text =  content["text"]
        asrConf = content["asrConf"]
        timeStamp = content["timeStamp"]

        #bot[userID][sessionID].

        # A reponse from your bot
        #response = usrStacks[sessionID].GetResponse(text)

        # A message to DialPort: sessionID, version, sys (system utterance), terminal (true if the end of the dialog), imageurl
        Output = {}
        Output["sessionID"] = sessionID
        Output["timeStamp"] = datetime.now().isoformat()
        Output["version"] = "0.1"
        Output['terminal'] = False

        if text == 'END':
            end[userID][sessionID] = True
            if sum(list(end[userID].values())) >= 2:
                Output['sys'] = ('You have finished the chats with two bots!'
                    'The TOPIC: {}.'.format(topic[userID]))
            else:
                Output['sys'] = ('You have done 1 chat. Please finish the second chat to see the topic and then do the evaluation.')
            Output['terminal'] = True
            return jsonify(Output)

        if len(sess[sessionID].raw_history) > 20:
            Output['sys'] = 'IGNORED. (max turns achieved. please input END to finish the chat)'
            return jsonify(Output)

        sess[sessionID].step('human: ' + text)
        sess[sessionID].infer_step()
        Output['sys'] = sess[sessionID].raw_history[-1]

        if len(sess[sessionID].raw_history) > 20:
            Output['sys'] += '\n (max turns achieved. please input END to finish the chat)'

        return jsonify(Output)

        """
        if response["slu"]["act"] == "exit":
            Output["sys"] = "Goodbye. See you later"
            Output["terminal"] = True # At the end of the dialog, please send us True
            del usrStacks[sessionID]
        else:
            Output["imageurl"] = response["imageurl"]
            Output["terminal"] = False
            Output["sys"] = response["sys"]
        return jsonify(Output)
        """

    if uuid == "end":  #Terminate a session with your system
        content = request.json
        #A message from DialPort: sessionID, timeStamp
        sessionID = content["sessionID"]
        timeStamp = content["timeStamp"]
        userID = content['userID']

        # A message to DialPort: sessionID, version, sys (system utterance), terminal (true if the end of the dialog)
        Output = {}
        Output["sessionID"] = sessionID
        Output["timeStamp"] = datetime.now().isoformat()
        Output["version"] = "0.1"
        Output["terminal"] = True # At the end of the dialog, please send us True
        del usrStacks[sessionID]


        return jsonify(Output)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int)
    parser.add_argument('--config')
    args = parser.parse_args()

    app.run(host='0.0.0.0',port=args.port)
