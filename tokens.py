import os
#import sys
#import logging
#import re
import openai
#import zulip
from dotenv import load_dotenv
import tiktoken
#import sqlite3
#import datetime

load_dotenv()
openai.api_key=os.environ['OPENAI_API_KEY']

LOW_THRES = 100
MED_THRES = 500
CONVO_THRES = MED_THRES #Threshold to limit conversation history

delimiter = "###"

def get_completion_with_tokens(messages, prompt, model='gpt-3.5-turbo'): #tuple: str, dict: {"prompt_tokens": #, "completion_tokens": #, "total_tokens": #}
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    messages.append(response.choices[0].message)
    return response.choices[0].message["content"], response.usage

def get_completion(messages, prompt, model='gpt-3.5-turbo'): #str
    r, _ = get_completion_with_tokens(messages, prompt, model)
    return r

def count_token_history(messages, model='gpt-3.5-turbo'): #int
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = 0
    for message in messages: #role: ###, content: ###
        tokens += len(encoding.encode(message["content"]))
    
    return tokens

def usage_level(tokens):
    if tokens < LOW_THRES:
        return "LOW"
    elif tokens < MED_THRES:
        return "MEDIUM"
    else:
        return "HIGH"

def sum_history(history, model='gpt-3.5-turbo'):
    text = ""
    for m in history[:-2]:
        text += m["content"] + " "
    
    text = text.rstrip()

    prompt = f"""
    Summarize the following text delimited by {delimiter}. \
    Keep your summarization short, but do not loose important information. \
    {delimiter}{text}{delimiter}
    """
    
    nhistory = [get_completion([], prompt, model)]
    nhistory.extend(history[:-2])
    return nhistory

def handle_new(messageHistory, nPrompt, model='gpt-3.5-turbo'):
    response, usage = get_completion_with_tokens(messages=messageHistory, prompt=nPrompt, model=model)
    
    if usage["total_tokens"] > CONVO_THRES:
        messageHistory = sum_history(messageHistory, model)
    
    return response, messageHistory

history = [{"role": "system", "content": "You are a friendly chatbot that can hold conversations for a long period and summarizes message history efficiently."}]

def print_history(history):
    for h in history:
        print("{}: {}".format(h["role"], h["content"]))

user_in = input("--> ")
while user_in != "quit":
    if user_in == "history":
        print_history(history)
    else:
        response, history = handle_new(history, user_in)
        print("AI: {}".format(response))
    user_in = input("--> ")
