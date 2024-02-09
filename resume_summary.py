from google.colab import files
from transformers import AutoTokenizer, AutoModel
import torch
import torch.optim as optim
import torch.nn as nn
import tika
import os
import torch.nn.functional as F
import openai
from openai import OpenAI
from tika import parser
from deep_translator import GoogleTranslator
import sys

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
  - Return response in English

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

def get_resume_summary(file_path):
  resume = read_file(file_path)
  resume = preprocess_resume(resume)
  resume = GoogleTranslator(source='auto', target='en').translate(resume)
  summary = get_openai_response(resume)
  start_idx = summary.index('<response>') + len('<response>')
  end_idx = summary.index('</response>')
  resume_summary = summary[start_idx:end_idx]
  return resume_summary

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_openai_response_for_jd(job_description):
  prompt = f"""Your goal is write a job description summary, max 50 words.
  Here are some important rules:
  - Don't include company name or any personal information in summary.
  - Mention general industry of the company based on description.
  - Summary info about main requirements.
  - Don't include soft skills that are listed in job description. Generate info about soft skills based on whole text.
  - Return response in English

  Here is the job description <job_description>{job_description}</job_description >

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

base_path = sys.argv[1]
api_key = sys.argv[2]

openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = api_key
client = OpenAI()

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-albert-small-v2')

model = AutoModel.from_pretrained('sentence-transformers/paraphrase-albert-small-v2')
classifier_head = nn.Linear(model.config.hidden_size, 2)

optimizer = optim.Adam([
    {'params': model.parameters()},
    {'params': classifier_head.parameters()}
], lr=2e-5)

checkpoint = torch.load(base_path + 'vacancy_resume_classification_epoch_10_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
classifier_head.load_state_dict(checkpoint['classifier_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()

# JOB DESCRIPTION SUMMARY
descr = read_file(base_path + 'jd_to_test/jd.pdf')
try:
	descr = GoogleTranslator(source='auto', target='en').translate(descr)
except:
	pass
jd_summary = get_openai_response_for_jd(descr)
jd_summary = get_openai_response(jd_summary)
start_idx = jd_summary.index('<response>') + len('<response>')
end_idx = jd_summary.index('</response>')
jd_summary = jd_summary[start_idx:end_idx]

# RESUME SUMMARY AND EVALUATION
path_to_resumes = base_path + 'resumes_to_test/'
for resume_name in os.listdir(path_to_resumes):
    resume = read_file(path_to_resumes + resume_name)
    resume = preprocess_resume(resume)
    resume = GoogleTranslator(source='auto', target='en').translate(resume)
    summary = get_openai_response(resume)
    start_idx = summary.index('<response>') + len('<response>')
    end_idx = summary.index('</response>')
    summary = summary[start_idx:end_idx]

    sentences = [jd_summary, summary]

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    tensor1 = sentence_embeddings[0]
    tensor2 = sentence_embeddings[1]

    cos_sim = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)
    print("Resume is: ", resume_name)
    score = cos_sim.item()*2
    if score > 1:
        score = 1
    print("Score is:", score*100)