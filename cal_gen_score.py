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
import numpy as np

os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = os.getenv('OPENAI_API_KEY')


#llm = ChatOpenAI(model="gpt-3.5-turbo-0613")
# to make the result reproducible; (a lower temperature leads to a more deterministic result)
llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

rootdir = './laion_10k_data_2/'
with open(os.path.join(rootdir, "laion_10k_data_2_GPT_clean_captions.json"),'r') as f:
    json_str = f.read()
    orig = json.loads(json_str)

result_dict = {}

for key in orig:
    value = orig[key]
    print(key)
    tmp_list = []
    for v in value:
        tmp_list.append(v)
        v = "give me a score of generality (between 0 to 10) of the sentence: \"" + v + "\" based on: Specificity of Information, Broadness of Terms, Tense and Modality, Degree of Abstraction, Context Dependence, Universality of Statements. The result should in a dictionary with 6 keys. Only give me the dictionary!"
        #print(v)
        tasks = llm.predict_messages([HumanMessage(content=v)])
        logger.debug(tasks.content)
        score_dict = json.loads(tasks.content)
        score_dict["average"] = np.mean(list(score_dict.values()))
            
        tmp_list.append(score_dict)
    result_dict[key] = tmp_list

json_object = json.dumps(result_dict, indent=4)

base = rootdir.split('/')[1]
json_path = os.path.join(rootdir, f"{base}_GPT_captions_withgen_score.json")

with open(json_path, "w") as outfile:
    outfile.write(json_object)