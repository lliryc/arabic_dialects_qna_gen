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

SURVEY_ID_1 = "1B3OjMSK7UdCOUW4SDuWwyhbFoA0N49PYJz2PytXtuPE"
SURVEY_ID_2 = "1_kWFhB-qdk_GiWMR-P_wo50KiSo6zyuY-5PVA2r9-t8"
SURVEY_ID_3 = "1v9xpwdL84jCPmg9Au6IyDMzuzqwzNkOU7cApOytq7Ic"

TOTAL = "1Z22OgUxnAVoWPhzECIaQVcs1axiF6x9_udVsY8awZ8w"

survey_ids = [SURVEY_ID_1, SURVEY_ID_2, SURVEY_ID_3]

tabs = ["ar-eg-1", "ar-eg-2", "ar-eg-3", "ar-eg-4", "ar-eg-5"]

def get_qnq_log():
    records = []
    with open("question_logs_egyptian_old.txt", "r") as log_in:
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
                    record["LLMQuestionDifficulty"] = complexity
                    records.append(record)
                    print(record)
            except Exception as e:
                print(e)
                continue
    return records

def save_qnq_log():
    records = []
    with open("question_logs_egyptian.txt", "r") as log_in:
        log_content = log_in.read()
        log_strs = log_content.split("\n\n")
        for log_str in log_strs:
            try:
                log_array = json.loads(log_str)
                for log in log_array:
                    records.append(log)
            except Exception as e:
                print(e)
                continue
    df = pd.DataFrame(records)
    df.to_csv("question_logs_egyptian.csv", index=False)


records_log = get_qnq_log()

def get_survey_results(survey_id, tab):
    global records_log
    creds = service_account.Credentials.from_service_account_file(
        'google_api_credentials2.json', scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=survey_id, range=f"{tab}!B5:I22").execute()
    table_array = result.get('values', [])
    table_len = len(table_array)
    records = []
    for i in range(table_len//2):
        row1 = table_array[2*i]
        row2 = table_array[2*i+1]
        record = {}
        record["Question"] = row1[0]
        record["Answer"] = row2[0]
        record["QuestionRelevancy"] = row1[2]
        record["QuestionDifficulty"] = row1[4]
        for record_log in records_log:
            if record_log["Question"] == record["Question"]:
                record["LLMQuestionDifficulty"] = record_log["LLMQuestionDifficulty"]
                break
        if "LLMQuestionDifficulty" not in record:
            record["LLMQuestionDifficulty"] = "N/A"
        record["IsAnswerCorrect"] = row1[5]
        #record["Comments"] = row1[6]
        print(record)
        print("-"*100)
        records.append(record)
    return records

def iaa_results(df):
    map = {
        'Easy': 0,
        'Moderate': 1,
        'Challenging': 2
    }
    llm_difficulty = [map[difficulty] for difficulty in df['LLMQuestionDifficulty'].values]
    rv0_difficulty = [map[difficulty] for difficulty in df['QuestionDifficulty_0'].values]
    rv1_difficulty = [map[difficulty] for difficulty in df['QuestionDifficulty_1'].values]
    rv2_difficulty = [map[difficulty] for difficulty in df['QuestionDifficulty_2'].values]
    iaa_res_dict = {}
    iaa_res_dict["Krippendorff_alpha_rv0"] = krippendorff.alpha([llm_difficulty, rv0_difficulty], level_of_measurement="ordinal")
    iaa_res_dict["Krippendorff_alpha_rv1"] = krippendorff.alpha([llm_difficulty, rv1_difficulty], level_of_measurement="ordinal")
    iaa_res_dict["Krippendorff_alpha_rv2"] = krippendorff.alpha([llm_difficulty, rv2_difficulty], level_of_measurement="ordinal")
    iaaa_df = pd.DataFrame([iaa_res_dict])
    return iaaa_df
    
if __name__ == "__main__":
    '''
    result_series = []

    for survey_id in survey_ids:
        form_results = []
        for tab in tabs:
            results = get_survey_results(survey_id, tab)
            form_results.extend(results)
        result_series.append(form_results)

    res_cnt = len(result_series)
    final_records = []
    res_len = len(result_series[0])
    for j in range(res_len):
        res_rec = result_series[0][j]
        final_record = {}
        final_record["Question"] = res_rec["Question"]
        final_record["Answer"] = res_rec["Answer"]
        final_record["LLMQuestionDifficulty"] = res_rec["LLMQuestionDifficulty"]
        for i in range(res_cnt):
            final_record[f"IsAnswerCorrect_{i}"] = result_series[i][j]["IsAnswerCorrect"]
            final_record[f"QuestionRelevancy_{i}"] = result_series[i][j]["QuestionRelevancy"]
            final_record[f"QuestionDifficulty_{i}"] = result_series[i][j]["QuestionDifficulty"]
        final_records.append(final_record)
    df = pd.DataFrame(final_records)
    df.to_csv("final_results_table.csv", index=False)
    '''
    '''
    df = pd.read_csv("final_results_table2.csv")
    iaa_df = iaa_results(df)

    iaa_df.to_csv("iaa_results_table.csv", index=False)
    '''
    save_qnq_log()