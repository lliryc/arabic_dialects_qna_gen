from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import json
from langchain_core.output_parsers import JsonOutputParser


challenging_judgement_prompt = ChatPromptTemplate.from_template("""You are evaluating reading comprehension questions in {question_language} based on the text below written in {passage_language}. 
The questions allow usage of knowledge from the text as well as high school level knowledge to answer. 
Rate complexity based on how many distinct reasoning/inference steps are needed to answer correctly:

1: Minimal reasoning (direct retrieval)
2: Simple inference (one reasoning step)
3: Moderate inference (two reasoning steps)
4: Complex inference (three reasoning steps)
5: Highly challenging inference (four or more reasoning steps)

Also, label these dimensions (True/False):

- Opinionated: Subjective, requires personal opinion.
- Biased: Contains bias/prejudice.
- Non-answerable: Cannot be answered from text with high school level knowledge support.
- Irrelevant: Unrelated to text content.

Include recommendations:

- Critical: If any of Opinionated/Biased/NonAnswerable/Irrelevant are true, describe briefly how to correct it.
- NiceToHave: Suggest briefly how to increase complexity by adding inference/reasoning steps.

Output your evaluation strictly as the following JSON schema:

{{
  "Complexity": 1|2|3|4|5,                 // integer [1–5]
  "Complexity_reason": "string",          // string, reason for the complexity
  "Opinionated": false|true,            // boolean
  "Opinionated_reason": "string",      // string, reason for the opinionated
  "Biased": false|true,                 // boolean
  "Biased_reason": "string",          // string, reason for the biased
  "NonAnswerable": false|true,          // boolean
  "NonAnswerable_reason": "string",      // string, reason for the non-answerable
  "Irrelevant": false|true,             // boolean
  "Irrelevant_reason": "string",             // string, reason for the irrelevant
  "Recommendations": {{
    "Critical": "string",                // string, leave empty if no critical issues
    "NiceToHave": "string"               // string, suggestion to increase complexity
  }}
}}


Return only the JSON object, nothing else. Otherwise you will be penalized $1000 per word.

Text:
--------------------------------
{text}
--------------------------------

Question:
{question}
""")

common_judgement_prompt = ChatPromptTemplate.from_template("""You are evaluating reading comprehension questions in {question_language} based on the text below written in {passage_language}. 
The questions allow usage of knowledge from the text as well as high school level knowledge to answer. 

Rate its complexity using a 5-point Likert scale:

1 (Trivial): Answer obvious; no inference required; may directly repeat text.
2 (Easy): Little inference; minimal paraphrasing; slight thinking needed.
3 (Moderate): Requires clear inference, paraphrasing, and integration of ideas.
4 (Hard): Demands multiple inference steps, significant paraphrasing, or deeper reasoning.
5 (Very Hard): Highly complex inference; extensive paraphrasing; multiple challenging reasoning steps.

Assess additional dimensions (True/False):

- Opinionated: Subjective, requires personal opinion.
- Biased: Contains bias/prejudice.
- NonAnswerable: Cannot be answered from text with high school level knowledge support.
- Irrelevant: Unrelated to text content.

Include recommendations:

Critical: If any of Opinionated/Biased/NonAnswerable/Irrelevant are true, briefly describe how to correct it.

NiceToHave: Suggest briefly how to increase complexity.

Provide evaluation strictly in JSON:

{{
  "Complexity": 1|2|3|4|5,                 // integer [1–5]
  "Complexity_reason": "string",          // string, reason for the complexity
  "Opinionated": false|true,            // boolean
  "Opinionated_reason": "string",      // string, reason for the opinionated
  "Biased": false|true,                 // boolean
  "Biased_reason": "string",          // string, reason for the biased
  "NonAnswerable": false|true,          // boolean
  "Irrelevant": false,             // boolean
  "Recommendations": {{
    "Critical": "string",                // string; empty if no critical issues
    "NiceToHave": "string"               // string; suggestion to increase complexity
  }}
}}

""")


class LLMAsAJudge:
  def __init__(self, passage, passage_language, question_language):
    self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    self.passage = passage
    self.passage_language = passage_language
    self.question_language = question_language
  
  def run_challenging_eval_prompt(self, question):
    chain = challenging_judgement_prompt | self.llm | JsonOutputParser()
    try:
        res = chain.invoke({"text": self.passage, "passage_language": self.passage_language, "question_language": self.question_language, "question": question})
    except Exception as e:
        print(e)
        return None
    return res
  
  def run_common_eval_prompt(self, question):
      chain = common_judgement_prompt | self.llm | JsonOutputParser()
      try:
          res = chain.invoke({"text": self.passage, "passage_language": self.passage_language, "question_language": self.question_language, "question": question})
      except Exception as e:
          print(e)
          return None
      return res
