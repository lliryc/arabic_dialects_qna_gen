from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import json
from langchain_core.output_parsers import JsonOutputParser
from agent_llm_as_a_judge import LLMAsAJudge

load_dotenv()

question_types = ["easy", "moderate", "challenging"]



challenging_initial_prompt = ChatPromptTemplate.from_template("""You are a Logic & Reasoning teacher.  
Generate difficult follow-up non-opinionated question in {question_language} with proper answer in {answer_language}, based on the text below written in {passage_language}.  
Don't provide context as it is already given in the text. 
Question must be one sentence only and not have introductory or "given" section. 
Keep the question short and concise (no more than 16 words).
Answer must be short (no more than 3 words). 
To answer a question, reader must perform reasoning steps, using the knowledge from the text and that is typically expected from a high school graduate.
Do not rely on specialized university-level knowledge, advanced academic concepts, or expert-level details.

Return your response in the following JSON format:
{{
  "Question": "your question in {question_language} or N/A",
  "Answer": "your answer in {answer_language} or N/A"
}}

Text:
--------------------------------
{passage}
--------------------------------
""")

challenging_improvement_prompt = ChatPromptTemplate.from_template("""You are a Logic & Reasoning teacher.  
Improve the question below to make it more challenging.

Question:
{question}
""")

improvement_prompt = ChatPromptTemplate.from_template("""You are a Logic & Reasoning teacher.  
Improve difficulty of your reading comprehension question in {question_language} with proper answer in {answer_language} based on the text written in {passage_language} following the recommendation below. Keep the question short and concise (no more than 16 words).
Use only knowledge from the text to answer the question and high school level knowledge. No specialized university-level knowledge, advanced academic concepts, or expert-level details.
Answer must be short (no more than 3 words).
  
Return your response in the following JSON format:
{{
  "Question": "your difficult follow-up question or N/A",
  "Answer": "your short answer or N/A"
}}

Text:
--------------------------------
{passage}
--------------------------------

Question:
{question}

Recommendation:
--------------------------------
Critical:
{critical_recommendation}

Nice to have:
{nice_to_have_recommendation}
--------------------------------
""")

easy_initial_prompt = ChatPromptTemplate.from_template("""
You need to generate reading comprehension non-opinionated question in {question_language} with proper answer in {answer_language} based on the text written in {passage_language}. 
This question should focus on understanding and interpreting the passage, and it must be possible to infer the answers to this question from the text and not rely on external knowledge or one's opinion.
Question description: a straightforward question based on explicit information in the text. 
The answer is usually found in a single sentence or a clearly stated portion of the text. Little to no interpretation is needed; the task mostly requires finding a specific word or phrase. 
IMPORTANT: You response must contain the question and answer only or 'N/A' if question can not be created. If you include any other text, you get fined 1000$ per word.
IMPORTANT: You must not use the similar question as the existing questions. if you do, you get fined 1000$ per word.
Keep the question short and concise (no more than 16 words).
Answer must be short (no more than 3 words). 

Text:
{passage}

Return your response in the following JSON format:
{{
  "Question": "your question in {question_language} or N/A",
  "Answer": "your answer in {answer_language} or N/A"
}}

Existing questions:
{existing_questions}
""")

moderate_initial_prompt = ChatPromptTemplate.from_template("""
You need to generate reading comprehension non-opinionated question in {question_language} with proper answer in {answer_language} based on the passage written in {passage_language}. 
This question should focus on understanding and interpreting the passage, and it must be possible to infer the answers to this question from the passage and not rely on external knowledge or one's opinion.
Question description: A question that requires some understanding and interpretation of the text.
The reader needs to connect pieces of information or track the order of events.
IMPORTANT: The question must be paraphrased and should NOT use the exact same words or phrases directly from the passage. Rephrase the content in your own words while maintaining the same meaning.
IMPORTANT: You must not use the similar question as the existing questions. if you do, you get fined 1000$ per word.
You response must contain the question and answer only or 'N/A' if question can not be created. If you include any other text, you get fined 1000$ per word.
Keep the question short and concise (no more than 16 words).
Answer must be short (no more than 3 words). 

Passage:
{passage}

Return your response in the following JSON format:
{{
  "Question": "your question in {question_language} or N/A",
  "Answer": "your answer in {answer_language} or N/A"
}}

Existing questions:
{existing_questions}
""")




class QuestionBuilder:
    def __init__(self, passage, passage_language, question_language, answer_language):
        # Set basic attributes first
        self.passage = passage
        self.passage_language = passage_language
        self.question_language = question_language
        self.answer_language = answer_language
        self.question_answer_pairs = []
        self.existing_questions = []
        
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
            self.judge = LLMAsAJudge(passage, passage_language, question_language)
        except Exception as e:
            print(f"Warning: Failed to initialize LLMAsAJudge: {e}")
            self.judge = None
    
    def build_qna_in_single_step(self, question_type):
        if question_type == "easy":
            chain = easy_initial_prompt | self.llm | JsonOutputParser()
        elif question_type == "moderate":
            chain = moderate_initial_prompt | self.llm | JsonOutputParser()

        else:
            raise ValueError(f"Invalid question type: {question_type}")
           
        try:
            res = chain.invoke({"passage": self.passage, "passage_language": self.passage_language, "question_language": self.question_language, "answer_language": self.answer_language, "existing_questions": self.existing_questions})
        except Exception as e:
            print(e)
            return None
        return res["Question"], res["Answer"]
    
    def build_qna_in_multiple_steps(self):
      if self.judge is None:
          print("Warning: Judge not initialized, skipping challenging question generation")
          return None, None
          
      max_complexity = -1
      print("Building challenging question in multiple steps...")
      print("Text:")
      print(self.passage)
      print("--------------------------------")
      
      initial_chain = challenging_initial_prompt | self.llm | JsonOutputParser()
      improvement_chain = improvement_prompt | self.llm | JsonOutputParser()
      try:
          res = initial_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, "question_language": self.question_language, "answer_language": self.answer_language})
      except Exception as e:
          print(e)
          return None
        
      if res is None:
        print("No question generated")
        return None
      
      print("Question-Generation Agent: ", res)
      question = res["Question"]
      answer = res["Answer"]
      iteration_limit = 3
      
      final_question = None
      final_answer = None
      
      for i in range(iteration_limit):
        judgement = self.judge.run_challenging_eval_prompt(question)
        if judgement is None:
          print("No judgement generated")
          return None
        print("Judgement Agent: ", judgement)
        
        complexity = int(judgement["Complexity"])
        complexity_reason = judgement["Complexity_reason"]
        opinionated = judgement["Opinionated"]
        opinionated_reason = judgement["Opinionated_reason"]
        biased = judgement["Biased"]
        biased_reason = judgement["Biased_reason"]
        non_answerable = judgement["NonAnswerable"]
        non_answerable_reason = judgement["NonAnswerable_reason"]
        irrelevant = judgement["Irrelevant"]
        irrelevant_reason = judgement["Irrelevant_reason"]
        critical_recommendation = judgement["Recommendations"]["Critical"]
        nice_to_have_recommendation = judgement["Recommendations"]["NiceToHave"]
        
        if complexity > max_complexity and not opinionated and not biased and not non_answerable and not irrelevant:
          max_complexity = complexity
          final_question = question
          final_answer = answer
        
        if critical_recommendation == "None" and nice_to_have_recommendation == "None":
          print("No improvement needed")
          break
        print("Improving question...")

        try:
            res = improvement_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, "question": question, "question_language": self.question_language, "answer_language": self.answer_language, "critical_recommendation": critical_recommendation, "nice_to_have_recommendation": nice_to_have_recommendation})  
        except Exception as e:
            print(e)
            return None
          
        if res is None:
          print("No improvement generated")
          break
        print("Question-Improvement Agent: ", res)
        question = res["Question"]
        answer = res["Answer"]
        
      print(f"Final question: {final_question}")
      print(f"Final answer: {final_answer}")
      return final_question, final_answer
    
    def build_qna(self):

        challenging_question, challenging_answer = self.build_qna_in_multiple_steps()
        results = []
        results.append({"Question": challenging_question, "Answer": challenging_answer})
        self.existing_questions.append(challenging_question)
        
        moderate_question, moderate_answer = self.build_qna_in_single_step("moderate")
        results.append({"Question": moderate_question, "Answer": moderate_answer})
        self.existing_questions.append(moderate_question)
        
        easy_question, easy_answer = self.build_qna_in_single_step("easy")
        results.append({"Question": easy_question, "Answer": easy_answer})
        self.existing_questions.append(easy_question)
        
        print("Question-Builder: ", results)
        
        return results
    