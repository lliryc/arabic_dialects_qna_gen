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
from agent_question_builder import QuestionBuilder

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEMPLATE_ID = "1FUHdBALXPddxs6aT_S3lF9wTF0kVXYnYPq72zQo7Ytw"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]

def set_cell_value(cell, value):
    creds = None
    try:
        # Load credentials from service account file
        creds = service_account.Credentials.from_service_account_file(
            'google_api_credentials2.json', scopes=SCOPES)

        # Build the Sheets API service
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API to set the value of the cell
        sheet = service.spreadsheets()
        result = sheet.values().update(
            spreadsheetId=TEMPLATE_ID,
            range=f'QnA!{cell}',
            valueInputOption="USER_ENTERED",
            body={"values": [[value]]}
        ).execute()
        
        return result
    except HttpError as err:
        print(f"An error occurred: {err}")
        return None
      
def get_cell_value(service, cell, spreadsheet_id):
    try:
        # Call the Sheets API to get the value of the cell
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=f'QnA!{cell}').execute()
        return result.get('values', [[None]])[0][0]
    except HttpError as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":

    # Load credentials from service account file
    creds = service_account.Credentials.from_service_account_file(
      'google_api_credentials2.json', scopes=SCOPES)

    # Build the Sheets API service


    for i in tqdm(range(100), desc="Questions Generation"):
      
        service = build('sheets', 'v4', credentials=creds)

        original_passage = get_cell_value(service, f"A{i+2}", TEMPLATE_ID)
        
        # Skip the first line of the passage
        if original_passage and '\n\n' in original_passage:
            passage = '\n\n'.join(original_passage.split('\n\n')[1:])
        
        question_builder = QuestionBuilder(passage, "Emirati Arabic (Khaleej) ", "Emirati Arabic (Khaleej) ", "Emirati Arabic (Khaleej)", "United Arab Emirates")
        results = question_builder.build_qna()
        print(results)
        
        easy_question = results[2]["Question"]
        
        easy_answer = results[2]["Answer"]
        if easy_question is not None:
           easy_question_with_answer = f"{easy_question}\n\n\n{easy_answer}"
           set_cell_value(f"M{i+2}", easy_question_with_answer)
        else:
           set_cell_value(f"M{i+2}", "N/A")
       
        moderate_question = results[1]["Question"]
        moderate_answer = results[1]["Answer"]
        if moderate_question is not None:
            moderate_question_with_answer = f"{moderate_question}\n\n\n{moderate_answer}"
            set_cell_value(f"O{i+2}", moderate_question_with_answer)
        else:
            set_cell_value(f"O{i+2}", "N/A")
        
        challenging_question = results[0]["Question"]
        challenging_answer = results[0]["Answer"]
        if challenging_question is not None:
            challenging_question_with_answer = f"{challenging_question}\n\n\n{challenging_answer}"
            set_cell_value(f"Q{i+2}", challenging_question_with_answer)
        else:
            set_cell_value(f"Q{i+2}", "N/A")
            

        time.sleep(random.randint(2, 5))



