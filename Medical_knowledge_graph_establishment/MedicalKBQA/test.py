# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:47:05 2019

@author: 包启明
"""

def check_words(wds, sent):
    for wd in wds:
        if wd in sent:
            return True
    return False

s1 = "How much"
s2 = "How much time"

result = check_words(s1,s2)
print (result)