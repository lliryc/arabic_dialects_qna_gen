from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import json
from langchain_core.output_parsers import JsonOutputParser


challenging_judgement_prompt = ChatPromptTemplate.from_template("""
You are evaluating a reading comprehension question in {question_language} and its answer in {answer_language} based on the passage below, which is written in {passage_language}.
The question should be answerable using reasoning based on the passage, possibly supported by high school level knowledge from {country}.

Rate complexity based on how many distinct reasoning/inference steps are needed to answer correctly:
1: Minimal reasoning (direct retrieval or single simple connection)
2: Simple inference (one clear reasoning step)
3: Moderate inference (two reasoning steps)
4: Complex inference (three reasoning steps)
5: Highly challenging inference (four or more reasoning steps)

Also, label these quality and formatting dimensions (True/False):
- IsNonOpinionated: The question is objective and not based on personal opinion.
- UnambiguousAnswer: The answer is clear and does not allow multiple interpretations.
- IsUnbiased: The question is free from prejudice.
- IsAnswerable: The question can be answered from the text with the allowed outside knowledge.
- IsRelevant: The question is clearly related to the passage's content.
- AnswerNotInSpan: The answer is correctly derived through inference, not just copied from a single text span.
- IsInThirdPerson: The question is phrased strictly in the third person.
- NoHighLexicalOverlap: The question rephrases concepts and avoids copying exact phrases from the passage.
- NoSpecializedExternalKnowledge: The question might require high school level knowledge from {country} to answer. No specialized university-level knowledge, advanced academic concepts, or expert-level details.
- IsShortQuestion: The question is 16 words or fewer.


Include recommendations:
Critical: If any of the checks for boolean dimensions are false, describe how to fix it. 
NiceToHave: Suggest briefly how to increase the reasoning complexity of the question.

Output your evaluation strictly as the following JSON schema. Return only the JSON object, nothing else.

{{
  "Complexity": 1|2|3|4|5,
  "Complexity_reason": "string",
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
  "AnswerNotInSpan": true|false,
  "AnswerNotInSpan_reason": "string",
  "IsInThirdPerson": true|false,
  "IsInThirdPerson_reason": "string",
  "NoHighLexicalOverlap": true|false,
  "NoHighLexicalOverlap_reason": "string",
  "NoSpecializedExternalKnowledge": true|false,
  "NoSpecializedExternalKnowledge_reason": "string",
  "IsShortQuestion": true|false,
  "IsShortQuestion_reason": "string",
  "Recommendations": {{
    "Critical": "string",
    "NiceToHave": "string"
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
  "Ambiguous": false|true,            // boolean
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

moderate_judgement_prompt = ChatPromptTemplate.from_template("""
You are evaluating a reading comprehension question in {question_language} and its answer in {answer_language} based on the passage below, which is written in {passage_language}.
Your task is to provide a structured analysis of the question and answer using a rating grade of answer complexity, set of critical criteria and nice-to-have recommendations.

Rate getting answer complexity using a 3-point scale:
1 (Trivial): Answer obvious; no inference required; may directly repeat text.
2 (Easy): Little inference; minimal paraphrasing; slight thinking needed.
3 (Moderate): Requires clear inference, paraphrasing, and integration of ideas.

Also, label these quality and formatting dimensions (True/False):
- IsNonOpinionated: The question is objective and not based on personal opinion.
- UnambiguousAnswer: The answer is clear and does not allow multiple interpretations.
- IsUnbiased: The question is free from prejudice.
- IsAnswerable: The question can be answered from the text with the allowed outside knowledge.
- IsRelevant: The question is clearly related to the passage's content.
- IsInThirdPerson: The question is phrased strictly in the third person.
- IsNotVerbatimAnswer: It is impossible to use span from the passage to answer the question. The answer is uniquely inferable from passage content but never verbatim.
- QuestionFreeFromLinguisticOrGrammarTerms: Grammar or linguistic terms are not used in the question.
- IsShortQuestion: The question is 16 words or fewer.
- IsPreciseAnswer: The answer does not contain unnecessary details or irrelevant information.
- IsShortAnswer: The answer is less than 4 words.


Critical: If any of the checks for boolean dimensions are false, describe how to fix it. 
NiceToHave: Suggest briefly how to increase the reasoning complexity of the question.

Output your evaluation strictly as the following JSON schema. Return only the JSON object, nothing else.

Provide evaluation strictly in JSON:

{{
  "Complexity": 1|2|3,                 // integer [1–3]
  "Complexity_reason": "string",          // string, reason for the complexity
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
  "IsNotVerbatimAnswer": true|false,
  "IsNotVerbatimAnswer_reason": "string",
  "QuestionFreeFromLinguisticOrGrammarTerms": true|false,
  "QuestionFreeFromLinguisticOrGrammarTerms_reason": "string",
  "IsShortQuestion": true|false,
  "IsShortQuestion_reason": "string",
  "IsPreciseAnswer": true|false,
  "IsPreciseAnswer_reason": "string",
  "IsShortAnswer": true|false,
  "IsShortAnswer_reason": "string",

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
""")


class LLMAsAJudge:
  def __init__(self, passage, passage_language, question_language, answer_language, country):
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
    self.answer_language = answer_language
    self.country = country
    
  def run_challenging_eval_prompt(self, question, answer):
    chain = challenging_judgement_prompt | self.llm | JsonOutputParser()
    try:
        res = chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                            "question_language": self.question_language, "question": question, 
                            "answer": answer, "answer_language": self.answer_language, 
                            "country": self.country})
    except Exception as e:
        print(e)
        return None
    return res
  
  def run_moderate_eval_prompt(self, question, answer):
      chain = moderate_judgement_prompt | self.llm | JsonOutputParser()
      try:
          res = chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                              "question_language": self.question_language, "question": question, 
                              "answer_language": self.answer_language, "answer": answer
                              })
      except Exception as e:
          print(e)
          return None
      return res
