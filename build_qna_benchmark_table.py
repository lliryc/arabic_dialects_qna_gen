from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import tqdm
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import random
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
from langchain_core.output_parsers import JsonOutputParser
from agent_question_builder import QuestionBuilder
import corrected_passage_restoration_google_sheets as restoration
from camel_tools.utils.dediac import dediac_ar

from corrected_passage_restoration_google_sheets import get_file_ids_from_folder, get_document_tab_names, get_table, get_table_with_background, get_hyperlink, isNone, is_skip_mark, get_corrected_speaker_mark, process_row, write_to_google_sheet, get_corrected_transcription
from gemini_passages_assessment import censorship_check
import copy
import pandas as pd
import krippendorff

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]


TEMPLATE_ID = "18GDzBq-YU7mGdcysjE6DYwXgAMQZpToa0lQbNT_62bk"


tabs = ["ar-eg-1", "ar-eg-2", "ar-eg-3", "ar-eg-4", "ar-eg-5"]

def get_qnq_log():
    records = []
    with open("question_logs_egyptian.txt", "r") as log_in:
        log_content = log_in.read()
        log_strs = log_content.split("\n\n")
        for log_str in log_strs:
            try:
                log_array = json.loads(log_str)
                complexities = ["Challenging", "Moderate", "Easy"]
                for log, complexity in zip(log_array, complexities):
                    record = {}
                    record["Question"] = log["Question"]
                    record["Answer"] = log["Answer"]
                    record["Quotes"] = log["Quotes"]
                    record["LLMQuestionDifficulty"] = complexity
                    records.append(record)
                    print(record)
            except Exception as e:
                print(e)
                continue
    return records

records_log = get_qnq_log()

def get_survey_results(survey_id, tab):
    global records_log
    creds = service_account.Credentials.from_service_account_file(
        'google_api_credentials2.json', scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    texts = sheet.values().get(spreadsheetId=survey_id, range=f"{tab}!B2:B2").execute()
    texts_array = texts.get('values', [])
    passage = texts_array[0][0]
    result = sheet.values().get(spreadsheetId=survey_id, range=f"{tab}!B5:B22").execute()
    table_array = result.get('values', [])
    table_len = len(table_array)
    records = []
    for i in range(table_len//2):
        row1 = table_array[2*i]
        row2 = table_array[2*i+1]
        record = {}
        record["Passage"] = passage
        record["Question"] = row1[0]
        record["Answer"] = row2[0]

        for record_log in records_log:
            if record_log["Question"] == record["Question"]:
                record["Quotes"] = record_log["Quotes"]
                record["LLMQuestionDifficulty"] = record_log["LLMQuestionDifficulty"]
                break
              
        if "LLMQuestionDifficulty" not in record:
            record["Quotes"] = "N/A"
            record["LLMQuestionDifficulty"] = "N/A"
            
        print(record)
        print("-"*100)
        records.append(record)
    return records


    
if __name__ == "__main__":
    
    result_series = []

    for tab in tabs:
        results = get_survey_results(TEMPLATE_ID, tab)
        result_series.extend(results)

    df = pd.DataFrame(result_series)
    df.to_csv("qna_dataset_table.csv", index=False)
    