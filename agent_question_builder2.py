from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import json
from langchain_core.output_parsers import JsonOutputParser
from agent_llm_as_a_judge2 import LLMAsAJudge

load_dotenv()

combined_initial_prompt = ChatPromptTemplate.from_template("""
You are a linguist and {passage_language} native speaker.
You need to generate reading comprehension non-opinionated question with precise, unambiguous answer, along with a list of exact quotes from the passage that were used to form the answer.
The questions should be based on understanding and interpreting the text in the passage.
A question can sometimes be derived partly through common knowledge in {country} that is not explicitly present in the passage.
It is possible to either find the answer to this question directly in the text or to infer the answer from the overall meaning of the passage.
In order to answer an inferential question, pieces of information in the passage may need to be connected, or the sequence of events may need to be tracked.
The answers to the generated questions must undergo all the following cases:

1.The answer must be found directly in the passage. It should match the passage closely in meaning but does not need to be copied verbatim. Minor adjustments are allowed for pronoun consistency (e.g., change “I did” to “he did” or “she did”), as well as slight rephrasing or morphological changes.

2. The question must be phrased strictly in the third person.

3.The answer must be inferred by piecing together information from different parts of the passage.

4.The answer must be inferred by interpreting the overall meaning of the passage.


You must not use a question similar to the previous questions. The answer to the question should not be the same as the answer to previous questions.

Identify and extract the full quote(s) where the answer to the question is based.
The quotes must be exact, and the character indices of each quote within the passage must be provided.
The quotes must not include any extra details that are not explicitly in the answers
Formulate each question strictly in the third person.
Questions and answers must be generated using ONLY the {passage_language}.
Your response must contain the question and answer only, or “N/A” if the question cannot be created.
Keep each question short and concise.
The answer must be short.

IMPORTANT: The question, answer and extracted sentence must be in {passage_language}.Do not use MSA or other dialects, otherwise you will be penalized $1000 per word.

Return your response in the following JSON format:
{{
  "Question": "<your question in {passage_language} or N/A>",
  "Answer": "<your answer in {passage_language} or N/A>",
  "Quotes":[
        {{
          "text": "<your quote in {passage_language} or N/A>",
          "start_char": <the starting character index of the quote in the passage as an integer, or -1 if N/A>,
          "end_char": <the ending character index of the quote in the passage as an integer, or -1 if N/A>
        }}
  ]
}}

Passage:
--------------------------------
{passage}
--------------------------------

Previous Questions:
{previous_questions}
""")

combined_improvement_prompt = ChatPromptTemplate.from_template("""
You are a linguist and {passage_language} native speaker.
Revise the original_question and original_answer based on the provided judge feedback.
Your primary mandate is to resolve all issues in Recommendations.Critical. 

Systematically correct every dimension flagged as false in the feedback, using the associated _reason fields to guide your edits. 
You need to generate a new version of reading comprehension non-opinionated question with precise, unambiguous answer, along with a list of exact quotes from the passage that were used to form the answer.
The questions should be based on understanding and interpreting the text in the passage.
A question can sometimes be derived partly through common knowledge in {country} that is not explicitly present in the passage.
It is possible to either find the answer to this question directly in the text or to infer the answer from the overall meaning of the passage.
In order to answer an inferential question, pieces of information in the passage may need to be connected, or the sequence of events may need to be tracked.
The answers to the generated questions must undergo all the following cases:

1.The answer must be found directly in the passage. It should match the passage closely in meaning but does not need to be copied verbatim. Minor adjustments are allowed for pronoun consistency (e.g., change “I did” to “he did” or “she did”), as well as slight rephrasing or morphological changes.

2. The question must be phrased strictly in the third person.

3.The answer must be inferred by piecing together information from different parts of the passage.

4.The answer must be inferred by interpreting the overall meaning of the passage.

Identify and extract the full quote(s) where the answer to the question is based.
The quotes must be exact, and the character indices of each quote within the passage must be provided.
The quotes must not include any extra details that are not explicitly in the answers
Formulate each question strictly in the third person.
Questions and answers must be generated using ONLY the {passage_language}.
Your response must contain the question and answer only, or “N/A” if the question cannot be created.
Keep each question short and concise.
The answer must be short.

IMPORTANT: The question, answer and extracted sentence must be in {passage_language}.Do not use MSA or other dialects, otherwise you will be penalized $1000 per word.

Return your response in the following JSON format:
{{
  "Question": "<your question in {passage_language} or N/A>",
  "Answer": "<your answer in {passage_language} or N/A>",
  "Quotes":[
        {{
          "text": "<your quote in {passage_language} or N/A>",
          "start_char": <the starting character index of the quote in the passage as an integer, or -1 if N/A>,
          "end_char": <the ending character index of the quote in the passage as an integer, or -1 if N/A>
        }}
  ]
}}

Passage:
--------------------------------
{passage}
--------------------------------

Original Question: 
{original_question}

Original Answer: 
{original_answer}

Original Quotes:
{original_quotes}

Judge Feedback:
{judge_feedback}
""")



class QuestionBuilder:
    def __init__(self, passage, passage_language, country, previous_questions = []):
        # Set basic attributes first
        self.passage = passage
        self.passage_language = passage_language
        self.question_answer_pairs = []
        self.previous_questions = previous_questions
        self.country = country
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        
        # Initialize judge with error handling
        try:
            self.judge = LLMAsAJudge(passage, passage_language, country)
        except Exception as e:
            print(f"Warning: Failed to initialize LLMAsAJudge: {e}")
            self.judge = None
    

    def build_combined_qna_in_multiple_steps(self):
      if self.judge is None:
          print("Warning: Judge not initialized, skipping combined question generation")
          return None, None
          
      print("Building combined question in multiple steps...")
      print("--------------------------------")
      
      initial_chain = combined_initial_prompt | self.llm | JsonOutputParser()
      improvement_chain = combined_improvement_prompt | self.llm | JsonOutputParser()
      try:
          res = initial_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, "country": self.country,
                                      "previous_questions": self.previous_questions})
      except Exception as e:
          print(e)
          return None
        
      if res is None:
        print("No question generated")
        return None
      
      print("Question-Generation Agent: ", res)
      question = res["Question"]
      answer = res["Answer"]
      quotes = res["Quotes"]
      
      iteration_limit = 3
      
      final_question = None
      final_answer = None
      final_quotes = None
      
      for _ in range(iteration_limit):
        judgement = self.judge.run_combined_eval_prompt(question, answer, quotes)
        if judgement is None:
          print("No judgement generated")
          return None
        print("Judgement Agent: ", judgement)
        
        check_passed = True

        for key, value in judgement.items():
          
          if key.endswith("_reason"):
            continue
          
          if type(value) == bool and value == False:
            check_passed = False
            break

        if check_passed:
          
          final_question = question
          final_answer = answer
          final_quotes = quotes
          
          break
          
        critical_recommendation = judgement["Recommendations"]["Critical"]
        nice_to_have_recommendation = judgement["Recommendations"]["NiceToHave"]
        
        if critical_recommendation == "None" and nice_to_have_recommendation == "None":
          print("No improvement needed")
          break
        print("Improving question...")

        try:
            res = improvement_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                                            "country": self.country,
                                            "original_question": question, "original_answer": answer,
                                            "original_quotes": quotes,
                                            "judge_feedback": judgement})  
        except Exception as e:
            print(e)
            return None
          
        if res is None:
          print("No improvement generated")
          break
        print("Question-Improvement Agent: ", res)
        
        question = res["Question"]
        answer = res["Answer"]
        quotes = res["Quotes"]
        
      print(f"Final question: {final_question}")
      print(f"Final answer: {final_answer}")
      print(f"Final quotes: {final_quotes}")
      return final_question, final_answer, final_quotes
    
    def build_qna(self):

      results = []

      combined_question, combined_answer, combined_quotes = self.build_combined_qna_in_multiple_steps()
      combined_question_obj = {"Question": combined_question, "Answer": combined_answer, "Quotes": combined_quotes, "LLMQuestionDifficulty": "Combined"}
      results.append(combined_question_obj)
      self.previous_questions.append(combined_question_obj)
      
      print("Question-Builder: ", results)
      
      return results
    