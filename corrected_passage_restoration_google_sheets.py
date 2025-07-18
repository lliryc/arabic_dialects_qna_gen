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
import time
import numpy as np
from uuid import uuid4


load_dotenv()


def align_table(array, fill_to):
    """
    Align a 2D array by filling shorter rows with None values to match the specified length.
    
    Args:
        array: 2D array (non-rectangular)
        fill_to: target length for all rows
    
    Returns:
        2D array with all rows padded to the specified length
    """
    aligned_array = []
    for row in array:
        # Create a new row with the original values plus None padding
        aligned_row = row + [None] * (fill_to - len(row))
        aligned_array.append(aligned_row)
    return aligned_array

def align_table_with_colors(array, fill_to):
    """
    Align a 2D array with color data by filling shorter rows with None values to match the specified length.
    
    Args:
        array: 2D array where each cell is a tuple (value, background_color)
        fill_to: target length for all rows
    
    Returns:
        2D array with all rows padded to the specified length
    """
    aligned_array = []
    for row in array:
        # Create a new row with the original values plus (None, {}) padding
        aligned_row = row + [(None, {})] * (fill_to - len(row))
        aligned_array.append(aligned_row)
    return aligned_array

target_sheets = {
  "Emirati": "1FUHdBALXPddxs6aT_S3lF9wTF0kVXYnYPq72zQo7Ytw",
}

original_folder_links = {
  "Emirati": "https://drive.google.com/drive/folders/1WT-f_fqM3habVbi1cp2QIdbJjdvoPgKn",
}


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive"
]

passage_paragraphs = []

speaker_mark = None

paragraph_mark = None

def get_file_ids_from_folder(folder_link):
  # Extract folder ID from the full URL
  if 'folders/' in folder_link:
    folder_id = folder_link.split('folders/')[-1].split('?')[0].split('/')[0]
  else:
    folder_id = folder_link  # Assume it's already just the ID
  
  creds = service_account.Credentials.from_service_account_file(
    'google_api_credentials2.json', scopes=SCOPES)
  
  service = build("drive", "v3", credentials=creds)
  files = service.files().list(q=f"parents='{folder_id}'").execute()
  return [(file['id'], file['name']) for file in files.get('files', [])]

def get_document_tab_names(doc_id):
  """Get sheet names (tabs) from a Google Sheet"""
  creds = service_account.Credentials.from_service_account_file(
    'google_api_credentials2.json', scopes=SCOPES)
  
  service = build("sheets", "v4", credentials=creds)
  sheet = service.spreadsheets()
  
  try:
    # Get the spreadsheet metadata which includes sheet names
    result = sheet.get(spreadsheetId=doc_id).execute()
    sheet_names = [sheet['properties']['title'] for sheet in result.get('sheets', [])]
    return sheet_names
  except HttpError as error:
    print(f"HTTP Error {error.resp.status}: {error}")
    return []
  except Exception as e:
    print(f"Unexpected error: {e}")
    return []


def get_table(service,doc_id, tab_name, cell_range):
  result = service.spreadsheets().values().get(spreadsheetId=doc_id, range=f"{tab_name}!{cell_range}").execute()
  res = result.get('values', None)
  if res is None:
    return None
  return align_table(res, 5)

def get_table_with_background(service,doc_id, tab_name, cell_range):
  """Get table data with background color information for each cell"""
  try:
    # Get cell data with formatting information including background colors
    result = service.spreadsheets().get(
      spreadsheetId=doc_id,
      ranges=f"{tab_name}!{cell_range}",
      fields="sheets(data(rowData(values(userEnteredFormat/backgroundColor,formattedValue))))"
    ).execute()
    
    sheets = result.get('sheets', [])
    if not sheets:
      return None
    
    data = sheets[0].get('data', [])
    if not data:
      return None
    
    table_with_colors = []
    for row_data in data[0].get('rowData', []):
      row = []
      for cell in row_data.get('values', []):
        # Get the cell value
        value = cell.get('formattedValue', '')
        
        # Get the background color
        user_format = cell.get('userEnteredFormat', {})
        background_color = user_format.get('backgroundColor', {})
        
        # Create a tuple with (value, background_color)
        cell_data = (value, background_color)
        row.append(cell_data)
      
      table_with_colors.append(row)
    
    # Align the table to 5 columns
    return align_table_with_colors(table_with_colors, 5)
    
  except Exception as e:
    print(f"Error getting table with background colors: {e}")
    return None

def get_hyperlink(service, doc_id, tab_name, cell_range):
  """Get hyperlink URL from a cell in Google Sheets"""
  try:
    # Get the cell data with hyperlink information
    result = service.spreadsheets().get(
      spreadsheetId=doc_id,
      ranges=f"{tab_name}!{cell_range}",
      fields="sheets(data(rowData(values(hyperlink,formattedValue))))"
    ).execute()
    
    # Extract hyperlink from the response
    sheets = result.get('sheets', [])
    if not sheets:
      return None
    
    data = sheets[0].get('data', [])
    if not data:
      return None
    
    row_data = data[0].get('rowData', [])
    if not row_data:
      return None
    
    values = row_data[0].get('values', [])
    if not values:
      return None
    
    # Get the hyperlink URL
    hyperlink = values[0].get('hyperlink')
    return hyperlink
    
  except Exception as e:
    print(f"Error getting hyperlink: {e}")
    return None


def isNone(value):
  if value is None:
    return True
  if value.strip() == "":
    return True
  return False

def is_skip_mark(color_table, row_number):
  if color_table is None:
    return False
  if color_table[row_number][3][0] is None:
    return False
  if (color_table[row_number][3][0]).strip() != "#":
    return False
  
  color_dict = color_table[row_number][3][1]
  
  if 'red' not in color_dict:
    return False
  
  if np.argmax([color_dict.get('red'), color_dict.get('green', 0), color_dict.get('blue', 0)]) == 0:
    return True
  
  return False

def get_corrected_speaker_mark(table, row_number):
  
  review_cell_value = table[row_number][3]
  
  if not isNone(review_cell_value) and review_cell_value.strip() != "#":
    if  'المتحدث' in review_cell_value.lower():
      return f"[* - *] ** {review_cell_value.strip()} **"
    else:
      return None
  
  return None

SKIPPED_TOKEN = "{" + "تم تخطيه" + "}"

def process_row(table, color_table, row_number):
  """Extract value from column A at specified row number"""
  global speaker_mark, paragraph_mark, passage_paragraphs
  
  first_cell_value = table[row_number][0]
  
  if not isNone(first_cell_value):
    if  'المتحدث' in first_cell_value.lower():
      speaker_mark = first_cell_value
      passage_paragraphs.append([speaker_mark])
      passage_paragraphs.append([])
      paragraph_mark = None
    else:
      if len(passage_paragraphs) == 0 or len(passage_paragraphs[-1]) > 0:
        passage_paragraphs.append([])
      paragraph_mark = first_cell_value
  
  corrected_speaker_mark = get_corrected_speaker_mark(table, row_number)
  
  if not isNone(corrected_speaker_mark):  
    speaker_mark = corrected_speaker_mark
    passage_paragraphs.append([speaker_mark])
    passage_paragraphs.append([])
    paragraph_mark = str(uuid4())

  
  if isNone(speaker_mark):
    return
  
  if isNone(paragraph_mark):
    return
  
  skip_mark = is_skip_mark(color_table, row_number)
  
  if skip_mark:
    if len(passage_paragraphs) > 0 and len(passage_paragraphs[-1]) == 0:
      passage_paragraphs.pop()
      
    if len(passage_paragraphs) == 0 or passage_paragraphs[-1][0] != SKIPPED_TOKEN:
      passage_paragraphs.append([SKIPPED_TOKEN])
      passage_paragraphs.append([])
    print(f"Skipping row {row_number}")
    return
  
  second_cell_value = table[row_number][2]
  
  original_value = table[row_number][1]
  
  if isNone(second_cell_value) and isNone(original_value):
    paragraph_mark = None
  
  if isNone(second_cell_value):
    return
  
  second_cell_value = second_cell_value.strip()
  
  tokens = second_cell_value.split()
  
  for token in tokens:
    passage_paragraphs[-1].append(token)
    
  return

def write_to_google_sheet(doc_id, tab_name, text, cell_range):
  creds = service_account.Credentials.from_service_account_file(
    'google_api_credentials2.json', scopes=SCOPES)
  
  service = build("sheets", "v4", credentials=creds)
  sheet = service.spreadsheets()
  
  sheet.values().update(
    spreadsheetId=doc_id,
    range=f"{tab_name}!{cell_range}",
    valueInputOption="USER_ENTERED",
    body={"values": [[text]]}
  ).execute()

def get_corrected_transcription(doc_id, tab_name):
  
  creds = service_account.Credentials.from_service_account_file(
    'google_api_credentials2.json', scopes=SCOPES)
  
  service = build("sheets", "v4", credentials=creds)
  sheet = service.spreadsheets()
  
  hyperlink = get_hyperlink(service, doc_id, tab_name, 'A2')
  
  table = get_table(service, doc_id, tab_name, 'A4:E1000')
  
  color_table = get_table_with_background(service, doc_id, tab_name, 'A4:E1000')
  
  if table is None:
    raise Exception(f"Table is None for {doc_id} {tab_name}")
  
  for row_number in range(len(table)):
    process_row(table, color_table, row_number)
    
  text = f"{hyperlink}\n\n"
  
  for i in range(len(passage_paragraphs)):
    
    words = passage_paragraphs[i]
    
    if len(words) == 0:
      continue
    
    if i + 1 < len(passage_paragraphs):
      words_next = passage_paragraphs[i + 1]
    else:
      words_next = None
      
    if len(words) > 0 and '**' in words[0] and (words_next is None or len(words_next) == 0):
      continue
    content = " ".join(words)

    text += f"{content}\n\n"
  
  return text


if __name__ == "__main__":
  # Test multiple document IDs to find which ones are accessible
  google_sheets_file_ids = get_file_ids_from_folder(original_folder_links['Emirati'])
  
  google_sheets_file_ids = list(sorted(google_sheets_file_ids, key=lambda x: int(x[1].split('_')[0])))
  
  target_sheet_id = target_sheets['Emirati']
  
  target_row_id = 2
  
  for google_sheet_file_id, google_sheet_file_name in google_sheets_file_ids:
    
    print( "Processing: ", google_sheet_file_id, " - ", google_sheet_file_name)
    
    tab_names = get_document_tab_names(google_sheet_file_id)
    
    print(tab_names)
    
    #tab_names = list(sorted(tab_names))
    
    for tab_name in tab_names:
      
      passage_paragraphs.clear()

      speaker_mark = None
      
      paragraph_mark = None
      
      if "status" in tab_name.lower():
        continue
      
      print( "Processing: ", tab_name)
      
      corrected_transcription = get_corrected_transcription(google_sheet_file_id, tab_name)
      
      print( "Corrected transcription: ")
      
      print(corrected_transcription)
      
      write_to_google_sheet(target_sheet_id, "QnA", corrected_transcription, f"A{target_row_id}")
      
      target_row_id += 1
      # break
      time.sleep(3)
      
    # break

