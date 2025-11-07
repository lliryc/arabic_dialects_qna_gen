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
from agent_question_builder2 import QuestionBuilder
import corrected_passage_restoration_google_sheets as restoration
from camel_tools.utils.dediac import dediac_ar

from corrected_passage_restoration_google_sheets import get_file_ids_from_folder, get_document_tab_names, get_table, get_table_with_background, get_hyperlink, isNone, is_skip_mark, get_corrected_speaker_mark, process_row, write_to_google_sheet, get_corrected_transcription
from gemini_passages_assessment import censorship_check
import copy

load_dotenv()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]

FOLDER_LINK = "https://drive.google.com/drive/folders/1WT-f_fqM3habVbi1cp2QIdbJjdvoPgKn"


TEMPLATE_ID = "1YiwT5720Pqsz8NfnvlFnIf6zHX44Oy_1H6H9FbLs6uY" # Form

def set_cell_value_with_color(sheet_id, cell, value, color_spans = []):

    creds = None

    try:
        # Load credentials from service account file
        creds = service_account.Credentials.from_service_account_file(
            'google_api_credentials2.json', scopes=SCOPES)

        # Build the Sheets API service
        service = build('sheets', 'v4', credentials=creds)

        # Convert cell reference to row/column indices
        match = re.match(r'([A-Z]+)(\d+)', cell)
        if not match:
            print(f"Invalid cell reference: {cell}")
            return None
            
        col_str, row_str = match.groups()
        col_index = 0
        for char in col_str:
            col_index = col_index * 26 + (ord(char) - ord('A') + 1)
        col_index -= 1  # Convert to 0-based index
        row_index = int(row_str) - 1  # Convert to 0-based index

        # Prepare the cell data
        cell_data = {
            "userEnteredValue": {
                "stringValue": value
            }
        }
        
        # Add text format runs
        text_format_runs = []
        current_position = 0
        
        # sort color_spans by start index
        color_spans = list(sorted(color_spans, key=lambda x: x["start"]))
        
        # Fill gaps with default color spans
        filled_spans = []
        current_position = 0
        
        for span in color_spans:
            # Add gap span if there's a gap
            if span["start"] > current_position:
                filled_spans.append({
                    "start": current_position,
                    "end": span["start"],
                    "color": {"red": 0, "green": 0, "blue": 0}
                })
            
            # Add the original span
            filled_spans.append(span)
            current_position = span["end"]
        
        # Add final gap if needed
        if current_position < len(value):
            filled_spans.append({
                "start": current_position,
                "end": len(value),
                "color": {"red": 0, "green": 0, "blue": 0}
            })
        
        # Now process all spans (original + gap fills)
        for span in filled_spans:
            # Ensure start index is valid (>= 0)
            if span["start"] < 0:
                print(f"Warning: Skipping invalid span with negative start index: {span['start']}")
                continue
            text_format_runs.append({
                "startIndex": span["start"],
                "format": {
                    "foregroundColor": span["color"]
                }
            })
        
        cell_data["textFormatRuns"] = text_format_runs

        # Single batch update to set both value and formatting
        # For spanning multiple columns, we need to provide data for each column
        # or use mergeCells to merge the range first
        request = {
            "updateCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_index,
                    "endRowIndex": row_index + 1,
                    "startColumnIndex": col_index,
                    "endColumnIndex": col_index + 1
                },
                "rows": [{
                    "values": [cell_data]
                }],
                "fields": "userEnteredValue,textFormatRuns"
            }
        }
        
        batch_update_body = {
            "requests": [request]
        }
        
        sheet = service.spreadsheets()
        result = sheet.batchUpdate(
            spreadsheetId=TEMPLATE_ID,
            body=batch_update_body
        ).execute()
        
        return result
    except HttpError as err:
        print(f"An error occurred: {err}")
        return None

def set_cell_value(tab_name, cell, value):
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
            range=f'{tab_name}!{cell}',
            valueInputOption="USER_ENTERED",
            body={"values": [[value]]}
        ).execute()
        
        return result
    except HttpError as err:
        print(f"An error occurred: {err}")
        return None

def get_censorship_list(censorship_result):
    
    labels = ["toxic", "sexual", "violence", "racial", "other"]
    
    output = []
    
    for label in labels:
        res = "No"
        if label in censorship_result and censorship_result[label]:
            res = "Yes"
            if label + "_reason" in censorship_result:
                res += f". ({censorship_result[label + '_reason']})"
        output.append(res)
    return output

def set_merged_cell_value(sheet_id, cell_range, value, color_spans=None):
    """
    Set value for a merged cell range (e.g., "B2:F3")
    
    Args:
        sheet_id: The ID of the sheet
        cell_range: Range in A1 notation (e.g., "B2:F3") - should be a merged cell
        value: Single value to set in the merged cell
        color_spans: Optional list of color spans for text formatting
    """
    creds = None
    try:
        # Load credentials from service account file
        creds = service_account.Credentials.from_service_account_file(
            'google_api_credentials2.json', scopes=SCOPES)

        # Build the Sheets API service
        service = build('sheets', 'v4', credentials=creds)

        # Parse the range to get start and end cells
        range_match = re.match(r'([A-Z]+)(\d+):([A-Z]+)(\d+)', cell_range)
        if not range_match:
            print(f"Invalid range format: {cell_range}. Expected format like 'B2:F3'")
            return None
            
        start_col_str, start_row_str, end_col_str, end_row_str = range_match.groups()
        
        # Convert column letters to indices
        def col_to_index(col_str):
            col_index = 0
            for char in col_str:
                col_index = col_index * 26 + (ord(char) - ord('A') + 1)
            return col_index - 1  # Convert to 0-based index
        
        start_col_index = col_to_index(start_col_str)
        end_col_index = col_to_index(end_col_str)
        start_row_index = int(start_row_str) - 1  # Convert to 0-based index
        end_row_index = int(end_row_str)  # Keep as 1-based for end index
        
        # Debug information
        print(f"Debug: sheet_id={sheet_id}, cell_range={cell_range}")
        print(f"Debug: start_col={start_col_str}({start_col_index}), start_row={start_row_str}({start_row_index})")
        print(f"Debug: end_col={end_col_str}({end_col_index}), end_row={end_row_str}({end_row_index})")
        print(f"Debug: value length={len(str(value))}")
        
        # Prepare the cell data (single value for merged cell)
        cell_data = {
            "userEnteredValue": {
                "stringValue": str(value)
            }
        }
        
        # Add text formatting if color_spans provided
        if color_spans and isinstance(value, str):
            text_format_runs = []
            current_position = 0
            
            # Sort color_spans by start index
            sorted_spans = list(sorted(color_spans, key=lambda x: x["start"]))
            
            # Fill gaps with default color spans
            filled_spans = []
            current_position = 0
            
            for span in sorted_spans:
                # Add gap span if there's a gap
                if span["start"] > current_position:
                    filled_spans.append({
                        "start": current_position,
                        "end": span["start"],
                        "color": {"red": 0, "green": 0, "blue": 0}
                    })
                
                # Add the original span
                filled_spans.append(span)
                current_position = span["end"]
            
            # Add final gap if needed
            if current_position < len(value):
                filled_spans.append({
                    "start": current_position,
                    "end": len(value),
                    "color": {"red": 0, "green": 0, "blue": 0}
                })
            
            # Create text format runs
            for span in filled_spans:
                # Ensure start index is valid (>= 0)
                if span["start"] < 0:
                    print(f"Warning: Skipping invalid span with negative start index: {span['start']}")
                    continue
                text_format_runs.append({
                    "startIndex": span["start"],
                    "format": {
                        "foregroundColor": span["color"]
                    }
                })
            
            if text_format_runs:
                cell_data["textFormatRuns"] = text_format_runs

        # Create the batch update request for merged cell
        # For merged cells, we only need to set the value in the top-left cell
        # and the merge will handle the rest
        request = {
            "updateCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": start_row_index,
                    "endRowIndex": start_row_index + 1,
                    "startColumnIndex": start_col_index,
                    "endColumnIndex": start_col_index + 1
                },
                "rows": [{
                    "values": [cell_data]
                }],
                "fields": "userEnteredValue,textFormatRuns"
            }
        }
        
        batch_update_body = {
            "requests": [request]
        }
        
        sheet = service.spreadsheets()
        result = sheet.batchUpdate(
            spreadsheetId=TEMPLATE_ID,
            body=batch_update_body
        ).execute()
        
        return result
    except HttpError as err:
        print(f"An error occurred: {err}")
        return None
  

color_pallete = [
  {"red": 0.000, "green": 0.451, "blue": 0.741},  # Dark Blue (#0073BD)
  {"red": 0.094, "green": 0.533, "blue": 0.247},  # Forest Green (#18883F)
  {"red": 0.780, "green": 0.490, "blue": 0.000},  # Burnt Orange (#C77D00)
  {"red": 0.545, "green": 0.000, "blue": 0.545},  # Dark Magenta (#8B008B)
  {"red": 0.165, "green": 0.318, "blue": 0.616},  # Indigo Blue (#2A518D)
  {"red": 0.647, "green": 0.165, "blue": 0.165},  # Brick Red (#A52A2A)
  {"red": 0.345, "green": 0.184, "blue": 0.494},  # Deep Purple (#582F7E)
  {"red": 0.459, "green": 0.459, "blue": 0.000},  # Olive (#757500)
  {"red": 0.180, "green": 0.400, "blue": 0.400}   # Dark Teal (#2E6666)
]

def filter_results(results, passage):
    filtered_results = []
    for result in results:
        if isinstance(result, str):
            continue
        if result["Question"] is None or result["Answer"] is None or result["Quotes"] is None:
            continue
        if len(result["Quotes"]) == 0:
            continue
        if result["Question"] == "N/A" or result["Answer"] == "N/A" or result["Quotes"][0] == "N/A":
            continue
        if not any([quote["text"] in passage for quote in result["Quotes"]]):
            continue
        if not all(quote["start_char"] != -1 and quote["end_char"] != -1 and quote["start_char"] >= 0 and quote["end_char"] >= 0 for quote in result["Quotes"]):
            continue
        filtered_results.append(result)
        
    return filtered_results

def custom_sort_key(result):

    if result.get("Quotes") and len(result["Quotes"]) > 0:
        quote_start = result["Quotes"][0].get("start_char", 0)
    else:
        quote_start = int(1e9)

    # Return a tuple for multi-criteria sorting
    # Lower values come first in sorting
    return quote_start

def build_spans(quotes, passage: str):
    global color_pallete
    output_color_spans = []
    for i, quote_list in enumerate(quotes):
        for quote in quote_list:
            quote_text = quote["text"]
            
            # Debug: Check if quote is found with 'in' operator
            found_with_in = quote_text in passage
            start = passage.find(quote_text)
            
            if start == -1:
                # Quote not found in passage, skip this span
                print(f"Warning: Quote text not found in passage: '{quote_text[:50]}...'")
                print(f"  Found with 'in' operator: {found_with_in}")
                print(f"  Quote length: {len(quote_text)}")
                print(f"  Passage length: {len(passage)}")
                if found_with_in:
                    print(f"  WARNING: Quote found with 'in' but not with 'find()' - this shouldn't happen!")
                    # Try to find the quote with different approaches
                    print(f"  Trying to find with stripped text...")
                    stripped_quote = quote_text.strip()
                    if stripped_quote != quote_text:
                        start = passage.find(stripped_quote)
                        if start != -1:
                            print(f"  Found with stripped text at position {start}")
                        else:
                            print(f"  Still not found with stripped text")
                continue
            end = start + len(quote_text)
            output_color_spans.append({"start": start, "end": end, "color": color_pallete[i % len(color_pallete)]})
    output_color_spans = list(sorted(output_color_spans, key=lambda x: x["start"]))
    return output_color_spans

if __name__ == "__main__":
  
    file_ids = get_file_ids_from_folder(FOLDER_LINK)
    
    file_ids = list(sorted(file_ids, key=lambda x: int(x[1].split("_")[0])))
    
    doc_id = file_ids[0][0]
    
    for j in range(0, 5):
    
      tab_info = get_document_tab_names(doc_id)
      
      tab_name = tab_info[j][0]
      tab_id = tab_info[j][1]
      
      passage = get_corrected_transcription(doc_id, tab_name)
      
      # Reset global variables after getting the transcription
      restoration.passage_paragraphs = []
      restoration.speaker_mark = None
      restoration.paragraph_mark = None
      
      form_tab_info = get_document_tab_names(TEMPLATE_ID)
      
      form_tab_name = form_tab_info[j][0]
      form_tab_id = form_tab_info[j][1]
      

      #result = set_merged_cell_value(form_tab_id, "B2:F3", passage, color_spans = [{"start": 0, "end": 100, "color": {"red": 0.353, "green": 0.098, "blue": 0.604}}])
      #print(f"Passage test result: {result}")
      
      print(passage)
      
      total_results = []
      with open(f"question_logs_emirati.txt", "w") as log_out:
        for i in tqdm(range(12), desc="Building questions"):
            previous_questions = copy.deepcopy(total_results)
            question_builder = QuestionBuilder(passage, "Khaleej_Arabic_dialect (اللهجة الخليجية)", "United Arab Emirates", previous_questions=previous_questions)
            results = question_builder.build_qna()
            for result in results:
              result["Passage"] = passage
            results_str = json.dumps(results, indent=4, ensure_ascii=False)
            log_out.write(results_str)
            log_out.write("\n")
            log_out.write("\n")
            filtered_results = filter_results(results, passage)
            total_results.extend(filtered_results)
            print("-"*100)
      
      total_results = filter_results(total_results, passage)
      
      total_results = sorted(total_results, key=lambda x: custom_sort_key(x))
      
      if len(total_results) < 9:
          print(f"Not enough questions found: {len(total_results)}") 
          exit()
          
      total_results = total_results[:9]
      
      quotes = [result["Quotes"] for result in total_results]
      
      color_spans = build_spans(quotes, passage)
      
      for i, result in enumerate(total_results):
          set_cell_value(form_tab_name, f"B{2*i+5}", dediac_ar(result["Question"]))
          set_cell_value(form_tab_name, f"B{2*i+6}", dediac_ar(result["Answer"]))
          
      set_merged_cell_value(form_tab_id, "B2:F3", passage, color_spans=color_spans)
          
      #print("-"*100)
      #print("Total results:")
      #print(total_results)
      
      #censorship_result = censorship_check(text, "Syrian Arabic)")
      
      #censorship_list = get_censorship_list(censorship_result)
      
      #print(censorship_list)
      
      #for i, item in enumerate(censorship_list):
      #    set_cell_value(form_tab_name, f"C{i+7}", item)
    