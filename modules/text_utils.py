import opencc
import re

def convert_to_traditional(text):
    """將簡體中文轉換為繁體中文"""
    converter = opencc.OpenCC('s2t') 
    return converter.convert(text)
