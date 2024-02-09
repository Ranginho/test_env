from google.colab import files
from transformers import AutoTokenizer, AutoModel
import torch
import torch.optim as optim
import torch.nn as nn
import tika
import os
import openai
from openai import OpenAI
from tika import parser
from deep_translator import GoogleTranslator
import sys

openai.api_key = "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"
os.environ['OPENAI_API_KEY'] = "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"

tika.initVM()

def read_file(file_path):
    """ Reads any type of document (.pdf .doc .docx)
      Args:
        file_path(str): path of file
      Returns:
        content(str): content of file
    """

    parsed_file = parser.from_file(file_path)
    content = parsed_file['content']

    return content

def preprocess_resume(resume):
    resume = resume.replace('\n', '')
    resume = resume.replace('\uf02a', '')
    resume = resume.replace('\uf0b7', '')
    resume = resume.replace('\uf0a8', '')
    resume = resume.replace('\uf0d8', '')
    resume = resume.replace('\uf041', '')
    resume = resume.replace('\uf095', '')
    resume = resume.replace('\uf0e0', '')
    resume = resume.replace('\uf0a7', '')
    resume = resume.replace('\uf020', '')
    resume = resume.replace('\xa0', '')
    resume = resume.replace('\x0c', '')
    resume = resume.replace('\x00', '')
    resume = resume.replace('u0000', '')
    resume = resume.replace('\\', '')
    resume = resume.replace('•', '')
    resume = resume.replace('▪', '')
    resume = resume.replace('-', '')
    resume = resume.replace('  ', '')
    resume = resume.replace('"', '')

    return resume

def get_openai_response(resume):
  prompt = f"""Your goal is write a resume summary, max 50 words.
  Here are some important rules:
  - Don't include candidate name, surname or any personal information in summary
  - Don't include company names or university names from resume.
  - Mention general industry of the person based on work experience.
  - Summary info about education fields.
  - Summary info about work experience fields.
  - Don't include soft skills that are listed in resume. Generate info about soft skills based on work experience.
  - Don't start summary with word 'experienced'
  - Mention actual level of candidate in summary, is this candidate beginner, intermediate, experienced
  - Don't include information about languages
  Here is the resume <resume>{resume}</resume>

  Put your response in <response></response> tags."""

  response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {'role': 'system', 'content': 'You are an AI recruiter'},
                {'role': 'user',
                 'content': prompt}
            ]
        )
  res = response.choices[0].message.content

  return res


client = OpenAI()

def get_resume_summary(file_path):
  resume = read_file(file_path)
  resume = preprocess_resume(resume)
  resume = GoogleTranslator(source='auto', target='en').translate(resume)
  summary = get_openai_response(resume)
  start_idx = summary.index('<response>') + len('<response>')
  end_idx = summary.index('</response>')
  resume_summary = summary[start_idx:end_idx]
  return resume_summary

file_path = sys.argv[1]

get_resume_summary(file_path)