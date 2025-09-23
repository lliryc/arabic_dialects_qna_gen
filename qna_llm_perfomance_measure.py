from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import time
import json
import pandas as pd
from difflib import SequenceMatcher
from langchain_core.output_parsers import JsonOutputParser
from camel_tools.utils.dediac import dediac_ar
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.tokenizers.word import simple_word_tokenize
from sentence_transformers import SentenceTransformer

embeddings_model = SentenceTransformer("google/embeddinggemma-300m")

db = MorphologyDB.builtin_db('calima-egy-r13')
analyzer = Analyzer(db)

load_dotenv()

llm1 = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

llm2 = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY"),
)


reading_comprehension_easy_answer_prompt = ChatPromptTemplate.from_template("""
You are a {passage_language} native speaker and expert reading comprehension assistant.
You are given a passage and a reading comprehension question. Your task is to generate a precise, accurate answer in {passage_language} based solely on the information provided in the passage written in {passage_language}.

Instructions:
1. Read the passage carefully and understand its content
2. Analyze the question to determine what information is being asked for
3. Find the relevant information in the passage that answers the question
4. Generate a concise, accurate answer based on the passage content
5. The answer must be copied verbatim as a short quote from the passage. Do not paraphrase or rephrase.
6. If the answer is not available in the passage, respond with "N/A"
7. The answer must be in {passage_language}

Passage:
--------------------------------
{passage}
--------------------------------

Question: {question}

Return your answer in the following JSON format:
{{
  "answer": "<your answer in {passage_language} or N/A>"
}}

Return only the JSON object, nothing else.
""")

reading_comprehension_moderate_answer_prompt = ChatPromptTemplate.from_template("""
You are a {passage_language} native speaker and expert reading comprehension assistant.
You are given a passage and a reading comprehension question. Your task is to generate a precise, accurate answer in {passage_language} based on the information provided in the passage written in {passage_language}.

Instructions:
1. Read the passage carefully and understand its content
2. Analyze the question to determine what information is being asked for
3. Find the relevant word(s) in the passage that answer the question
4. Apply the appropriate morphological transformation to the word(s) from the passage
5. The answer must be a different grammatical form of a word/words from the passage (e.g., convert noun to adjective, noun to demonym, verb to noun, comparative form to superlative form and vice versa)
6. Frame your answer as normal comprehension, NOT as a linguistic or grammatical transformation
8. Ensure the transformed answer is uniquely inferable from passage content but never verbatim
9. If the answer cannot be derived from the passage, respond with "N/A"
10. The answer must be in {passage_language}


Passage:
--------------------------------
{passage}
--------------------------------

Question: {question}

Return your answer in the following JSON format:
{{
  "answer": "<your transformed answer in {passage_language} or N/A>"
}}

Return only the JSON object, nothing else.
"""
)

reading_comprehension_challenging_answer_prompt = ChatPromptTemplate.from_template("""
You are a Logic & Reasoning expert and {passage_language} native speaker.
You are given a passage and a challenging reading comprehension question. Your task is to generate a precise, accurate answer in {passage_language} based on the information provided in the passage written in {passage_language}.

Instructions:
1. Read the passage carefully and understand its content
2. Analyze the question to determine what information is being asked for
3. Connect multiple pieces of information from the passage or track the order of events
4. Apply logical reasoning and inference to derive the answer
5. The answer must be derived through inference or reasoning, but never a verbatim copy from a single text span
6. You may use high school level knowledge from {country} that is not present in the passage
7. The answer must be short, precise and unambiguous (no more than 4 words)
8. If the answer cannot be derived from the passage, respond with "N/A"
9. The answer must be in {passage_language}

Reasoning Requirements:
- Connect multiple pieces of information from the passage
- Apply cause-and-effect reasoning where appropriate
- Use logical deduction to reach conclusions
- Track sequences or order of events if relevant
- Make inferences that go beyond direct text retrieval
- Ensure your reasoning is well-supported by passage content

IMPORTANT: The answer must be in {passage_language}. Otherwise you will be penalized $1000 per word.

Passage:
--------------------------------
{passage}
--------------------------------

Question: {question}

Return your answer in the following JSON format:
{{
  "answer": "<your reasoned answer in {passage_language} or N/A>"
}}

Return only the JSON object, nothing else.
"""
)

question_difficulty_to_prompt = {
    "Easy": reading_comprehension_easy_answer_prompt,
    "Moderate": reading_comprehension_moderate_answer_prompt,
    "Challenging": reading_comprehension_challenging_answer_prompt
}

class ReadingComprehensionAnswerGenerator:
    def __init__(self, passage, passage_language, question_difficulty, country="any", llm_provider="gemini"):
        """
        Initialize the reading comprehension answer generator.
        
        Args:
            passage (str): The text passage to answer questions from
            passage_language (str): The language of the passage (e.g., "Arabic", "English")
            question_difficulty (str): The difficulty level ("Easy", "Moderate", "Challenging")
            country (str): The country context for external knowledge (default: "any")
            llm_provider (str): Either "gemini" or "chatgpt"
        """
        self.passage = passage
        self.passage_language = passage_language
        self.question_difficulty = question_difficulty
        self.country = country
        
        if llm_provider.lower() == "chatgpt":
            self.llm = llm2  # ChatGPT
        else:
            self.llm = llm1  # Gemini (default)
            
    def generate_answer(self, question):
        """
        Generate an answer to a reading comprehension question.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: Generated answer based on the passage
        """
        # Select the appropriate prompt based on question difficulty
        prompt = question_difficulty_to_prompt.get(self.question_difficulty)
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            # Prepare the input parameters
            input_params = {
                "passage": self.passage,
                "passage_language": self.passage_language,
                "question": question
            }
            
            # Add country parameter for challenging questions
            if self.question_difficulty == "Challenging":
                input_params["country"] = self.country
            
            result = chain.invoke(input_params)
            return result.get("answer", None)
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None
    
def generate_answers(df_qna_egyptian):
  
  generated_answers = []
  
  for _, row in tqdm(df_qna_egyptian.iterrows(), total=len(df_qna_egyptian), desc="Generating answers"):
    
    passage = row["Passage"]
    question = row["Question"]
    difficulty = row["LLMQuestionDifficulty"]
    
    if difficulty not in ["Easy", "Moderate", "Challenging"]:
      generated_answers.append("N/A")
      continue
    
    generator = ReadingComprehensionAnswerGenerator(
      passage=passage,
      passage_language="Egyptian Arabic (العامية المصرية)",
      question_difficulty=difficulty,
      country="Egypt",
      llm_provider="gemini"
    )
    
    generated_answer = generator.generate_answer(question)
    
    generated_answers.append(generated_answer)

  return generated_answers


def evaluate_answers(df_qna_egyptian):
  for _, row in tqdm(df_qna_egyptian.iterrows(), total=len(df_qna_egyptian), desc="Evaluating answers"):
    question = row["Question"]
    answer = row["Answer"]
    generated_answer = row["GeneratedAnswer"]
    print(question, answer, generated_answer)
    
def evaluate_easy_answer(passage, question, question_difficulty, answer, spans, generated_answer):
  if question_difficulty != "Easy":
    return 0
  return SequenceMatcher(None, a=answer, b=generated_answer).ratio()

def evaluate_moderate_answer(passage, question, question_difficulty, answer, spans, generated_answer):
  if question_difficulty != "Moderate":
    return 0
  spans = spans.replace("'", '"')
  spans_json = json.loads(spans)
  spans = []
  for span_obj in spans_json:
    spans.append(span_obj["text"])
  span = " ".join(spans)
  span_tokens = simple_word_tokenize(span)
  span_lemmas_set = set()
  for token in span_tokens:
    analyses = analyzer.analyze(token)
    if len(analyses) > 0:
      analysis = analyses[0]
      span_lemmas_set.add(analysis.get('lex', ''))
  
  generated_answer_tokens = simple_word_tokenize(generated_answer)
  generated_answer_lemmas_set = set()
  for token in generated_answer_tokens:
    analyses = analyzer.analyze(token)
    if len(analyses) > 0:
      analysis = analyses[0]
      generated_answer_lemmas_set.add(analysis.get('lex', ''))
  
  if len(span_lemmas_set.intersection(generated_answer_lemmas_set)) == 0:
    return 0
  else:
    return 1
  
def evaluate_challenging_answer(passage, question, question_difficulty, answer, spans, generated_answer):
  if question_difficulty != "Challenging":
    return 0
  generated_answer_embedding = embeddings_model.encode_query(generated_answer)
  answer_embeddings = embeddings_model.encode_document([answer])
  
  similarities = embeddings_model.similarity(generated_answer_embedding, answer_embeddings)
  
  # convert Tensor to float
  similarities = similarities.tolist()
  
  return similarities[0][0]

def build_generated_answers_table():

  df_qna_egyptian = pd.read_csv("qna_dataset_table.csv")
  df_qna_egyptian["GeneratedAnswer"] = generate_answers(df_qna_egyptian)
  df_qna_egyptian.to_csv("qna_dataset_table_with_generated_answers.csv", index=False)

def build_evaluations_table():
  df_qna_egyptian = pd.read_csv("qna_dataset_table_with_generated_answers.csv")
  df_qna_egyptian["Easy_LCS_Evaluation"] = df_qna_egyptian.apply(lambda row: evaluate_easy_answer(row["Passage"], row["Question"],row["LLMQuestionDifficulty"], row["Answer"], row["Quotes"], row["GeneratedAnswer"]), axis=1)
  df_qna_egyptian["Moderate_Lemmas_Intersection_Evaluation"] = df_qna_egyptian.apply(lambda row: evaluate_moderate_answer(row["Passage"], row["Question"], row["LLMQuestionDifficulty"], row["Answer"], row["Quotes"], row["GeneratedAnswer"]), axis=1)
  df_qna_egyptian["Challenging_Similarity_Evaluation"] = df_qna_egyptian.apply(lambda row: evaluate_challenging_answer(row["Passage"], row["Question"], row["LLMQuestionDifficulty"], row["Answer"], row["Quotes"], row["GeneratedAnswer"]), axis=1)
  df_qna_egyptian.to_csv("qna_dataset_table_with_generated_answers_and_evaluations.csv", index=False)

# Example usage
if __name__ == "__main__":
  build_evaluations_table()
      