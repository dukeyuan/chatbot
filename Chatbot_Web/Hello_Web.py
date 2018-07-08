#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Hello_Web.py
@desc:
@time: 2018/07/08 
"""

import web


urls = ('/Welcome', 'Welcome')

class Welcome:
    def GET(self):
        return "First Py Web"

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()