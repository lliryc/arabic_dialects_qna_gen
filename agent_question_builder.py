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



challenging_initial_prompt = ChatPromptTemplate.from_template("""
You need to generate a reading comprehension non-opinionated question in {question_language} along with a short, precise, unambiguous  answer in {answer_language} based on the passage written in {passage_language}. 
Formulate the question strictly in the third person.
This question should be based on understanding and interpreting the passage. 
It must be possible to infer the answers to this question from the passage.
In order to answer the question, pieces of information in the passage might need to be connected or the order of events need to be tracked.

IMPORTANT:The correct answer must be derived through inference or reasoning, but never verbatim copy from a single text span. It can be derived partly through external knowledge of high school level in {country} that is not present in the passage.

Try to minimise the use of the exact same words or phrases directly from the passage when generating the question. The question should be a rephrasing of parts of the passage content that maintains the original meaning.
You must not use a question similar to the previous questions. The answer to the question should not be the same as the answer to previous questions.
Your response must contain the question and answer only or 'N/A' if the question can not be created.
Keep the question short and concise (no more than 16 words).
The answer must be short, precise and unambiguous. 

Return your response in the following JSON format:
{{
  "Question": "your question in {question_language} or N/A",
  "Answer": "your answer in {answer_language} or N/A"
}}

Passage:
--------------------------------
{passage}
--------------------------------
""")

challenging_improvement_prompt = ChatPromptTemplate.from_template("""
Revise the original_question and original_answer based on the provided judge feedback.
Your primary mandate is to resolve all issues in Recommendations.Critical. 
Systematically correct every dimension flagged as false in the feedback, using the associated _reason fields to guide your edits. 
After addressing all critical errors, implement the Recommendations.NiceToHave to increase the question's reasoning complexity.
Ensure all boolean dimensions from the feedback are True for new version of question and answer. If not, make changes to the question and answer to satisfy all boolean dimensions.
The final question must be written in {question_language} and answer written in {answer_language}.

Here is a judgement criteria description.

Complexity based on how many distinct reasoning/inference steps are needed to answer correctly:
1: Minimal reasoning (direct retrieval or single simple connection)
2: Simple inference (one clear reasoning step)
3: Moderate inference (two reasoning steps)
4: Complex inference (three reasoning steps)
5: Highly challenging inference (four or more reasoning steps)

Boolean dimensions (True/False):
- IsNonOpinionated: The question is objective and not based on personal opinion.
- UnambiguousAnswer: The answer is clear and does not allow multiple interpretations.
- IsUnbiased: The question is free from prejudice.
- IsAnswerable: The question can be answered from the text with the allowed outside knowledge.
- IsRelevant: The question is clearly related to the passage's content.
- AnswerNotInSpan: The answer is correctly derived through inference, NOT a verbatim copy from a single text span.
- IsInThirdPerson: The question is phrased strictly in the third person.
- NoHighLexicalOverlap: The question rephrases concepts and avoids copying exact phrases from the passage.
- NoSpecializedExternalKnowledge: The question might require high school level knowledge from {country} to answer. No specialized university-level knowledge, advanced academic concepts, or expert-level details.
- IsShortQuestion: The question is 15 words or fewer.
- IsShortAndPreciseAnswer: The answer is short (no more than 4 words) and does not contain unnecessary details or irrelevant information.


Return your response in the following JSON format. Keep answer precise, unambiguous and short or you will be penalized $1000 per word. If the question is impossible to fix, return 'N/A'.
{{
  "Question": "The improved question in {question_language} or N/A",
  "Answer": "The improved precise, unambiguous and short answer in {answer_language} or N/A"
}}

INPUT: 

Passage:
----------------------------------
{passage}
----------------------------------

Original Question: 
{original_question}

Original Answer: 
{original_answer}

Judge Feedback:
{judge_feedback}

""")

easy_initial_prompt = ChatPromptTemplate.from_template("""
You need to generate a reading comprehension non-opinionated question in {question_language} along with a precise, unambiguous answer in {answer_language} based on the passage written in {passage_language}. 
Formulate the question strictly in the third person.
This question should be based on understanding and interpreting the passage, and it must be possible to find the answer to this question in the text and not rely on inference, on external knowledge or on one's own opinion.
The answer must be found in one text span of the passage.
You must not use a question similar to the previous questions. The answer to the question should not be the same as the answer to previous questions.
Your response must contain the question and answer only or 'N/A' if the question can not be created.
Keep the question short and concise (no more than 16 words).
The answer must be short.

Passage:
{passage}

Return your response in the following JSON format:
{{
  "Question": "your question in {question_language} or N/A",
  "Answer": "your answer in {answer_language} or N/A"
}}

Previous questions:
{previous_questions}
""")

moderate_initial_prompt = ChatPromptTemplate.from_template("""
You need to generate a reading comprehension non-opinionated question in {question_language} along with a precise, unambiguous  answer in {answer_language} based on the passage written in {passage_language}. 
Formulate the question strictly in the third person.
This question should be based on understanding and interpreting the passage. 
It must be possible to infer the answers to this question from the passage.
In order to answer the question, pieces of information in the passage might need to be connected or the order of events need to be tracked.

IMPORTANT: The correct answer must be a different grammatical form/forms of a word/words from a passage. Use morphological inflectional, derivational or other word transformations (e.g., convert noun  to adjective, noun to  demonym, verb  to  noun, comparative form to superlative form of adjective/adverb and vice versa). Do NOT mention linguistic or grammar terms in the question.  Frame the question as normal comprehension, NOT as a linguistic or grammatical question. Ensure the transformed answer is uniquely inferable from passage content but never verbatim.

Try to minimise the use of the exact same words or phrases directly from the passage when generating the question. The question should be a rephrasing of parts of the passage content that maintains the original meaning.
You must not use a question similar to the previous questions. The answer to the question should not be the same as the answer to previous questions.
Your response must contain the question and answer only or 'N/A' if the question can not be created.
Keep the question short and concise (no more than 16 words).
The answer must be short. 

Passage:
--------------------------------
{passage}
--------------------------------

Return your response in the following JSON format:
{{
  "Question": "your question in {question_language} or N/A",
  "Answer": "your answer in {answer_language} or N/A"
}}

Previous questions:
{previous_questions}
""")


moderate_improvement_prompt = ChatPromptTemplate.from_template("""
Revise the original_question and original_answer based on the provided judge feedback.
Your primary mandate is to resolve all issues in Recommendations.Critical. 
Systematically correct every dimension flagged as false in the feedback, using the associated _reason fields to guide your edits. 
After addressing all critical errors, implement the Recommendations.NiceToHave to increase the question's reasoning complexity.
The final reading comprehension question-answer pair must be non-opinionated, have a short, unambiguous answer derivable or inferred from the passage.

Return your response in the following JSON format. If the question is impossible to fix, return 'N/A'.
{{
  "Question": "The improved question in {question_language} or N/A",
  "Answer": "The improved answer in {answer_language} or N/A"
}}

INPUT: 

Passage:
----------------------------------
{passage}
----------------------------------

Original Question: 
{original_question}

Original Answer: 
{original_answer}

Judge Feedback:
{judge_feedback}

""")


class QuestionBuilder:
    def __init__(self, passage, passage_language, question_language, answer_language, country):
        # Set basic attributes first
        self.passage = passage
        self.passage_language = passage_language
        self.question_language = question_language
        self.answer_language = answer_language
        self.question_answer_pairs = []
        self.previous_questions = []
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
            self.judge = LLMAsAJudge(passage, passage_language, question_language, answer_language, country)
        except Exception as e:
            print(f"Warning: Failed to initialize LLMAsAJudge: {e}")
            self.judge = None
    
    def build_easy_qna_in_single_step(self):

        chain = easy_initial_prompt | self.llm | JsonOutputParser()
         
        try:
            res = chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                                "question_language": self.question_language, "answer_language": self.answer_language, 
                                "previous_questions": self.previous_questions})
        except Exception as e:
            print(e)
            return None
        return res["Question"], res["Answer"]
    
    def build_challenging_qna_in_multiple_steps(self):
      if self.judge is None:
          print("Warning: Judge not initialized, skipping challenging question generation")
          return None, None
          
      max_complexity = -1
      print("Building challenging question in multiple steps...")
      print("Text:")
      print(self.passage)
      print("--------------------------------")
      
      initial_chain = challenging_initial_prompt | self.llm | JsonOutputParser()
      improvement_chain = challenging_improvement_prompt | self.llm | JsonOutputParser()
      try:
          res = initial_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                                      "question_language": self.question_language, "answer_language": self.answer_language, 
                                      "country": self.country})
      except Exception as e:
          print(e)
          return None
        
      if res is None:
        print("No question generated")
        return None
      
      print("Question-Generation Agent: ", res)
      question = res["Question"]
      answer = res["Answer"]
      iteration_limit = 2
      extended_iteration_limit = 3
      
      final_question = None
      final_answer = None
      
      for i in range(iteration_limit + extended_iteration_limit):
        judgement = self.judge.run_challenging_eval_prompt(question, question)
        if judgement is None:
          print("No judgement generated")
          return None
        print("Judgement Agent: ", judgement)
        
        complexity = int(judgement["Complexity"])
        check_passed = True
        
        for key, value in judgement.items():
          if key.endswith("_reason"):
            continue
          if type(value) == bool and value == False:
            check_passed = False
            break
        
        if complexity > max_complexity and check_passed:
          max_complexity = complexity
          final_question = question
          final_answer = answer
          
        critical_recommendation = judgement["Recommendations"]["Critical"]
        nice_to_have_recommendation = judgement["Recommendations"]["NiceToHave"]
        
        if critical_recommendation == "None" and nice_to_have_recommendation == "None":
          print("No improvement needed")
          break
        
        if final_question is not None and i >= iteration_limit:
          print("No improvement needed")
          break
        
        print("Improving question...")

        try:
            res = improvement_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                                            "question_language": self.question_language,
                                            "answer_language": self.answer_language, "country": self.country,
                                            "original_question": question, "original_answer": answer,
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
        
      print(f"Final question: {final_question}")
      print(f"Final answer: {final_answer}")
      return final_question, final_answer
    
    def build_moderate_qna_in_multiple_steps(self):
      if self.judge is None:
          print("Warning: Judge not initialized, skipping challenging question generation")
          return None, None
          
      max_complexity = -1
      print("Building medium question in multiple steps...")
      print("Text:")
      print(self.passage)
      print("--------------------------------")
      
      initial_chain = moderate_initial_prompt | self.llm | JsonOutputParser()
      improvement_chain = moderate_improvement_prompt | self.llm | JsonOutputParser()
      try:
          res = initial_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                                      "question_language": self.question_language, 
                                      "answer_language": self.answer_language,
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
      iteration_limit = 2
      
      final_question = None
      final_answer = None
      
      for _ in range(iteration_limit):
        judgement = self.judge.run_moderate_eval_prompt(question, answer)
        if judgement is None:
          print("No judgement generated")
          return None
        print("Judgement Agent: ", judgement)
        
        complexity = int(judgement["Complexity"])
        check_passed = True

        for key, value in judgement.items():
          if key.endswith("_reason"):
            continue
          if type(value) == bool and value == False:
            check_passed = False
            break

        if complexity > max_complexity and check_passed:
          max_complexity = complexity
          final_question = question
          final_answer = answer
          
        critical_recommendation = judgement["Recommendations"]["Critical"]
        nice_to_have_recommendation = judgement["Recommendations"]["NiceToHave"]
        
        if critical_recommendation == "None" and nice_to_have_recommendation == "None":
          print("No improvement needed")
          break
        print("Improving question...")

        try:
            res = improvement_chain.invoke({"passage": self.passage, "passage_language": self.passage_language, 
                                            "question_language": self.question_language,
                                            "answer_language": self.answer_language,
                                            "original_question": question, "original_answer": answer,
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
        
      print(f"Final question: {final_question}")
      print(f"Final answer: {final_answer}")
      return final_question, final_answer
    
    def build_qna(self):

        challenging_question, challenging_answer = self.build_challenging_qna_in_multiple_steps()
        results = []
        #results.append({"Question": "N/A", "Answer": "N/A"})
        results.append({"Question": challenging_question, "Answer": challenging_answer})     
        self.previous_questions.append(challenging_question)
        
        moderate_question, moderate_answer = self.build_moderate_qna_in_multiple_steps()
        results.append({"Question": moderate_question, "Answer": moderate_answer})
        self.previous_questions.append(moderate_question)
        
        easy_question, easy_answer = self.build_easy_qna_in_single_step()
        results.append({"Question": easy_question, "Answer": easy_answer})
        self.previous_questions.append(easy_question)
        
        print("Question-Builder: ", results)
        
        return results
    