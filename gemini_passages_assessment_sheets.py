from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import tqdm
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
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

passage_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are LLM-as-a-judge. You are given a passage in {dialect}. You need to judge if 
            1. passage is toxic or not, 
            2. about sexual content or not.
            3. about violence or not.
            4. about racial content or not.
            5. about other sensitive content or not.
            
            Return the result in the following format:
            {{
                "toxic": true/false,
                "toxic_reason": "reason for the toxic content",
                "sexual": true/false,
                "sexual_reason": "reason for the sexual content",
                "violence": true/false,
                "violence_reason": "reason for the violence content",
                "racial": true/false,
                "racial_reason": "reason for the racial content",
                "other": true/false,
                "other_reason": "reason for the other sensitive content"
            }}
            If you are not sure about the content, return false for all the categories. Return only the result in the above format or you will be penalized $1000 per word.
            """,
        ),
        ("human", "{passage}"),
    ]
)

model_name = "gemini-2.5-pro"

#TEMPLATE_ID = "10LUCJ6xyZjUgn6RM80em1B5duRw2dK5Eizd3EQXI7xY"
TEMPLATE_ID = "1FUHdBALXPddxs6aT_S3lF9wTF0kVXYnYPq72zQo7Ytw"

if __name__ == "__main__":

  llm = ChatGoogleGenerativeAI(
      model="gemini-2.5-pro",
      temperature=0.3,
      max_tokens=None,
      timeout=None,
      max_retries=2,
      api_key=os.getenv("GOOGLE_API_KEY"),
      # other params...
  )

  chain = passage_eval_prompt | llm
  
  
  SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]

def run_json_prompt(prompt, passage, dialect):
    chain = prompt | llm | JsonOutputParser()
    try:  
        res = chain.invoke({"passage": passage, "dialect": dialect})
    except Exception as e:
        print(e)
        time.sleep(random.randint(2, 5))
        return None
   
    return res

def get_cell_value(service, cell, spreadsheet_id):
    try:
        # Call the Sheets API to get the value of the cell
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id, range=f'QnA!{cell}').execute()
        return result.get('values', [[None]])[0][0]
    except HttpError as err:
        print(f"An error occurred: {err}")
        return None

def set_cell_value(service, cell, value):
    try:
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

def set_row_value(service, cell, value, spreadsheet_id):
    try:
        # Call the Sheets API to set the value of the cell
        sheet = service.spreadsheets()
        result = sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=f'QnA!{cell}',
            valueInputOption="USER_ENTERED",
            body={"values": [[value]]}
        ).execute()
        
        return result
    except HttpError as err:
        print(f"An error occurred: {err}")
        return None

def set_eval_result(service, row, result, spreadsheet_id):
    data = []
    result_map = {
        "toxic": "C",
        "toxic_reason": "D",
        "sexual": "E",
        "sexual_reason": "F",
        "violence": "G",
        "violence_reason": "H",
        "racial": "I",
        "racial_reason": "J",
        "other": "K",
        "other_reason": "L"
    }
    for key, value in result.items():
        data.append({'range': f'QnA!{result_map[key]}{row}', 'values': [[str(value)]]})
    try:

        # Call the Sheets API to set the value of the cell
        sheet = service.spreadsheets()
        sheet.values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": "RAW", "data": data}
        ).execute()
    except HttpError as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
  
    # Load credentials from service account file
    creds = service_account.Credentials.from_service_account_file(
      'google_api_credentials2.json', scopes=SCOPES)

    # Build the Sheets API service
    service = build('sheets', 'v4', credentials=creds)

    for i in tqdm(range(100), desc="Censorship"):

        passage = get_cell_value(service, f"A{i+2}", TEMPLATE_ID)
        
        # Skip the first line of the passage
        if passage and '\n\n' in passage:
            passage = '\n\n'.join(passage.split('\n\n')[1:])
        

        result = run_json_prompt(passage_eval_prompt, passage, "Emirati Arabic")
        
        if result is not None:
            set_eval_result(service, i+2, result, TEMPLATE_ID)
        else:
            set_row_value(service, f"C{i+2}", "N/A", TEMPLATE_ID)

        time.sleep(random.randint(5, 10))


  
  

