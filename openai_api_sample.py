import os
import json
import openai
#import remote_execution as remote
import tempfile
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage
import re
import sys
from loguru import logger
from datetime import datetime
import json
import string

def is_first_char_symbol(s):
    if s[0] in string.punctuation:
        return True
    else:
        return False

def is_last_char_symbol(s):
    if s[-1] in string.punctuation:
        return True
    else:
        return False

os.environ["OPENAI_API_KEY"] = "xxx"

openai.api_key = os.getenv('OPENAI_API_KEY')


#llm = ChatOpenAI(model="gpt-3.5-turbo-0613")
# to make the result reproducible; (a lower temperature leads to a more deterministic result)
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

rootdir = './laion_10k_data_2/'

with open(os.path.join(rootdir, "all_captions_blip.json"),'r') as f:
    json_str = f.read()
    orig = json.loads(json_str)

result_dict = {}

for key in orig:
    value = orig[key]
    print(key)
    tmp_list = []
    for v in value:
        while (is_first_char_symbol(v) and is_last_char_symbol(v)):
            if is_first_char_symbol(v):
                v = v[1:]
            if is_last_char_symbol(v):
                v = v[:-1]
        #v = "convert this caption of an image to a more general caption: \"" + v + "\""
        #v = "Make this caption of an image extremely general (result in one caption)：\"" + v + "\""
        v = "Make this caption of an image extremely general (result in less than 5 words)：\"" + v + "\""
        #print(v)
        tasks = llm.predict_messages([HumanMessage(content=v)])
        logger.debug(tasks.content)
        tmp_list.append(tasks.content)
    result_dict[key] = tmp_list

json_object = json.dumps(result_dict, indent=4)

base = rootdir.split('/')[1]
json_path = os.path.join(rootdir, f"{base}_blip_5_words_captions.json")

with open(json_path, "w") as outfile:
    outfile.write(json_object)

