#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: main.py 
@desc: chatbot 入口函数
@time: 2018/07/08 
"""

'''

   Main script. See README.md for more information


'''

from Chatbot import chatbot

if __name__ == "__main__":
    chatbot = chatbot.Chatbot()
    chatbot.main()