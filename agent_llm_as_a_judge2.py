from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import json
from langchain_core.output_parsers import JsonOutputParser


combined_judgement_prompt = ChatPromptTemplate.from_template("""
You are {passage_language} native speaker.
You are evaluating a reading comprehension question and its answer based on the passage below, which is written in {passage_language}.
Your task is to set of critical criteria and nice-to-have recommendations.


Label these quality and formatting dimensions (True/False):
- IsNonOpinionated: The question is objective and not based on personal opinion.
- UnambiguousAnswer: The answer is clear and does not allow multiple interpretations.
- IsUnbiased: The question is free from prejudice.
- IsAnswerable: The question can be answered from the text with the allowed outside knowledge.
- IsRelevant: The question is clearly related to the passage's content.
- IsInThirdPerson: The question is phrased strictly in the third person.
- QuestionFreeFromLinguisticOrGrammarTerms: Grammar or linguistic terms are not used in the question.
- IsShortQuestion: The question is 20 words or fewer.
- IsPreciseAnswer: The answer does not contain unnecessary details or irrelevant information.
- IsIn{passage_language}: The question, answer and quotes must be in {passage_language}.


Critical: If any of the checks for boolean dimensions are false, describe how to fix it. 
NiceToHave: Suggest briefly how to increase the reasoning complexity of the question.

Output your evaluation strictly as the following JSON schema. Return only the JSON object, nothing else.

Provide evaluation strictly in JSON:

{{
  "IsNonOpinionated": true|false,
  "IsNonOpinionated_reason": "string",
  "UnambiguousAnswer": true|false,
  "UnambiguousAnswer_reason": "string",
  "IsUnbiased": true|false,
  "IsUnbiased_reason": "string",
  "IsAnswerable": true|false,
  "IsAnswerable_reason": "string",
  "IsRelevant": true|false,
  "IsRelevant_reason": "string",
  "IsInThirdPerson": true|false,
  "IsInThirdPerson_reason": "string",
  "QuestionFreeFromLinguisticOrGrammarTerms": true|false,
  "QuestionFreeFromLinguisticOrGrammarTerms_reason": "string",
  "IsShortQuestion": true|false,
  "IsShortQuestion_reason": "string",
  "IsPreciseAnswer": true|false,
  "IsPreciseAnswer_reason": "string",
  "IsIn{passage_language}": true|false,
  "IsIn{passage_language}_reason": "string",
  
  "Recommendations": {{
    "Critical": "string",                // string; empty if no critical issues
    "NiceToHave": "string"               // string; suggestion to increase complexity
  }}
}}

Return only the JSON object, nothing else. Otherwise you will be penalized $1000 per word.

Passage:
--------------------------------
{passage}
--------------------------------

Question:
{question}

Answer:
{answer}

Quotes:
{quotes}
""")


class LLMAsAJudge:
  def __init__(self, passage, passage_language, country):
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
    self.country = country
    
  def run_combined_eval_prompt(self, question, answer, quotes):
    chain = combined_judgement_prompt | self.llm | JsonOutputParser()
    try:
        res = chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                            "question": question, "answer": answer, "quotes": quotes, 
                            "country": self.country})
    except Exception as e:
        print(e)
        return None
    return res
  
  def run_combined_eval_prompt(self, question, answer, quotes):
      chain = combined_judgement_prompt | self.llm | JsonOutputParser()
      try:
          res = chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                              "question": question, "answer": answer, "quotes": quotes
                              })
      except Exception as e:
          print(e)
          return None
      return res
