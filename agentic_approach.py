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
  
# add constraints
# add multiple objectives

if __name__ == "__main__":
  question_builder = QuestionBuilder(text, "English", "English", "English")
  results = question_builder.build_qna()
  print(results)
