from chatterbot import ChatBot
from question_classifier import *
from question_parser import *
from answer_search import *
from chatbot_graph import ChatBotGraph

#'''问答类'''
#class ChatBotGraph:
#    def __init__(self):
#        self.classifier = QuestionClassifier()
#        self.parser = QuestionPaser()
#        self.searcher = AnswerSearcher()
#
#    def chat_main(self, sent):
#        answer = "Hello, I am XiaoMar Medical Assistant, I hope I can help you. If I don't answer it, I suggest you consult a professional doctor. I wish you a great body!"
#        res_classify = self.classifier.classify(sent)
#        if not res_classify:
#            return answer
#        res_sql = self.parser.parser_main(res_classify)
#        final_answers = self.searcher.search_main(res_sql)
#        if not final_answers:
#            return answer
#        else:
#            return '\n'.join(final_answers)


def get_response(usrText):
    handler = ChatBotGraph()
    while True:
        if usrText.strip()!= 'Bye':
            result = handler.chat_main(usrText)                        
            reply = str(result)
            return(reply)
        if usrText.strip() == 'Bye':
            return('Bye')
            break
        

        
