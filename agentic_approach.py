import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import random
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
from agent_question_builder import QuestionBuilder

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="o3", api_key=OPENAI_API_KEY)

text = """
MR AND MRS Brown first met Paddington on a railway platform. 
In fact, that was how he came to have such an unusual name for a bear, for Paddington was the name of the station. 
The Browns were there to meet their daughter Judy, who was coming home from school for the holidays. 
It was a warm summer day and the station was crowded with people on their way to the seaside. Trains were humming, 
loudspeakers blaring, porters rushing about shouting at one another, and altogether there was so much noise that Mr Brown, who saw him first, had to tell his wife several times before she understood.
"""

text0 = """
Apples also contain various polyphenols, including anthocyanins, which help give some apple peel its red colour and are associated with improved heart health. Another polyphenol you'll find in apples is phloridzin. It has been found to help control blood glucose.

There's also lots of fibre in apples, largely pectin, which reduces the amount of low-density lipoproteins (LDLs) – the unhealthy form of cholesterol – in our blood. Pectin also lowers the amount of sugar and fat we absorb from food, helping to stabilise our blood sugar levels.

These nutrients in apples do seem to offer health benefits. A 2017 review of five studies reported that eating apples is associated with an 18% reduction in the risk of developing type 2 diabetes. Another review from 2022, which analysed 18 studies, found that eating more apples, or apple-derived foods such as apple juice, can reduce cholesterol, if you sustain the habit for more than one week.
"""

# recommendation: use only knowledge, not a common sense
initial_prompt = ChatPromptTemplate.from_template("""You are a Logic & Reasoning teacher.  
Generate difficult follow-up question with proper answer, based on the text below.  
Don't provide context as it is already given in the text. 
Question must be one sentence only and not have introductory or "given" section. 
Answer must be short (no more than 3 words). 
To answer a question, reader must perform reasoning steps, using the knowledge from the text and that is typically expected from a high school graduate.
Do not rely on specialized university-level knowledge, advanced academic concepts, or expert-level details.

Return your response in the following JSON format:
{{
  "Question": "your question",
  "Answer": "your answer"
}}

Text:
--------------------------------
{text}
--------------------------------
""")

def run_initial_prompt(text):
    chain = initial_prompt | model
    try:
        res = chain.invoke({"text": text})  
    except Exception as e:
        print(e)
        return None
   
    return json.loads(res.content)
  
  
# add constraints
# add multiple objectives

judgement_prompt = ChatPromptTemplate.from_template("""You are LLM judge. 
Score the difficulty of the question (from 1 [very easy to answer] to 100 [extremely hard to answer]) based on the text and provide recommendation on how to improve the difficulty using knowledge from the text only if score is less than 50. 
Otherwise, just return "None" for recommendation.

Return your response in the following JSON format:
{{
  "Score": "integer number from 1 to 100",
  "Recommendation": "recommendation text or None"
}}

Text:
--------------------------------
{text}
--------------------------------

Question:
{question}
""")

def run_judgement_prompt(text, question):
    chain = judgement_prompt | model
    try:
        res = chain.invoke({"text": text, "question": question})  
    except Exception as e:
        print(e)
        return None
    return json.loads(res.content)



def run_improvement_prompt(text, question, recommendation):
    chain = improvement_prompt | model
    try:
        res = chain.invoke({"text": text, "question": question, "recommendation": recommendation})  
    except Exception as e:
        print(e)
        return None

    return json.loads(res.content)

if __name__ == "__main__":
  question_builder = QuestionBuilder(text, "English", "English", "English")
  results = question_builder.build_qna()
  print(results)
