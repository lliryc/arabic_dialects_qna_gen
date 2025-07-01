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

easy_prompt = ChatPromptTemplate.from_template("""
You need to generate reading comprehension question in Egyptian Arabic based on the passage written in Egyptian Arabic. 
This question should focus on understanding and interpreting the passage, and it must be possible to infer the answers to this question from the passage and not rely on external knowledge or one's opinion.
Question description: a straightforward question based on explicit information in the text. 
The answer is usually found in a single sentence or a clearly stated portion of the text. Little to no interpretation is needed; the task mostly requires finding a specific word or phrase. 
You response must contain the question only or 'N/A' if question can not be created. If you include any other text, you get fined 1000$ per word.

Passage:
{passage}

Question:
""")

moderate_prompt = ChatPromptTemplate.from_template("""
You need to generate reading comprehension question in Egyptian Arabic based on the passage written in Egyptian Arabic. 
This question should focus on understanding and interpreting the passage, and it must be possible to infer the answers to this question from the passage and not rely on external knowledge or one's opinion.
Question description: A question that requires some understanding and interpretation of the text. The reader needs to connect pieces of information or track the order of events. The answer might come from two or more parts of the text. 
You response must contain the question only or 'N/A' if question can not be created. If you include any other text, you get fined 1000$ per word.

Passage:
{passage}

Question:
""")

challenging_prompt = ChatPromptTemplate.from_template("""
Generate one reading-comprehension question in Egyptian Arabic about the given Egyptian Arabic passage.
Constraints:
1. The answer must be fully inferable from the passage; do not depend on external knowledge or personal opinion.
2. Choose exactly one schema from three options:
تحليل (Analysis): Ask about the relationship between ideas in non-adjacent paragraphs (e.g., «ما العلاقة بين …؟»).
استنتاج (Inference): Form a question that requires a logical deduction integrating information from three or more paragraphs.
عدّ (Counting): Ask the reader to count the instances of a specific action, idea, or goal—grouping synonymous or rephrased mentions as the same class—across the entire passage (e.g., «كم وسيلة استخدمها الكاتب لـ…؟»). Avoid counting literal word occurrences.
3.Return only the question text in Egyptian Arabic. If a valid question cannot be created, return exactly N/A.
4.Any extra words beyond the question or "N/A" incur a $1000 penalty per word.

Passage:
{passage}

Question:
""")

#llm_client = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)
llm_client = ChatOpenAI(model="o3", api_key=OPENAI_API_KEY)

TEMPLATE_ID = "10LUCJ6xyZjUgn6RM80em1B5duRw2dK5Eizd3EQXI7xY"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]

def run_prompt(prompt, passage):
    chain = prompt | llm_client
    try:
        res = chain.invoke({"passage": passage})  
    except Exception as e:
        print(e)
        time.sleep(random.randint(2, 5))
        return None
   
    return res.content

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

if __name__ == "__main__":
    files = glob.glob("egy_passages/*.txt")
    files.sort()
    total_files = len(files)
    
    for i, file in tqdm(enumerate(files), total=total_files, desc="Questions Generation"):
        with open(file, "r") as f:
            passage = f.read()
            # Remove content up to and including first double newline
            passage = passage.split('\n\n', 1)[-1]
        question_builder = QuestionBuilder(passage, "Egyptian Arabic", "Egyptian Arabic", "Modern Standard Arabic")
        results = question_builder.build_qna()
        print(results)
        
#        set_cell_value(f"A{i+2}", passage)
        
        easy_question = results[2]["Question"]
        if easy_question is not None:
            set_cell_value(f"M{i+2}", easy_question)
        else:
            set_cell_value(f"M{i+2}", "N/A")
       
        moderate_question = results[1]["Question"]
        if moderate_question is not None:
            set_cell_value(f"O{i+2}", moderate_question)
        else:
            set_cell_value(f"O{i+2}", "N/A")
        
        challenging_question = results[0]["Question"]
        if challenging_question is not None:
            set_cell_value(f"Q{i+2}", challenging_question)
        else:
            set_cell_value(f"Q{i+2}", "N/A")
            

        time.sleep(random.randint(5, 10))



