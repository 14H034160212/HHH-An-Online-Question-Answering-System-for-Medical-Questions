from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from question_classifier import *
from question_parser import *
from answer_search import *
from chatbot_graph import *
from chatbot import get_response


class ChatServer(WebSocket):
        
    def handleMessage(self):
        # echo message back to client
        message = self.data
#        handler = ChatBotGraph()
#        response = handler.chat_main(message)
        #response = self.chat_main(message)
        response = get_response(message)
        self.sendMessage(response)

    def handleConnected(self):
        print(self.address, 'connected')

    def handleClose(self):
        print(self.address, 'closed')



server = SimpleWebSocketServer('', 8000, ChatServer)
server.serveforever()
