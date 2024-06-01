from fastapi import FastAPI, File, UploadFile, Form
from ollama_client import create_quiz_generator, evaluator, generate_chunk
from pydantic import BaseModel
from pathlib import Path, os
import shutil

import json
import time
import uvicorn
import os

from dotenv import load_dotenv

load_dotenv()


#server1 "https://librairy.linkeddata.es/ollama" #remote
#server2 = "http://127.0.0.1:11434" #localPaul
#server3 "https://splendid-warthog-helpful.ngrok-free.app #cesvima

server = os.environ.get('SESSION_API_URL')

app = FastAPI()

class DataInput(BaseModel):
    text: str
    level: int = 2
    answer: str = None
    question: str = None
    openQuestion: bool = False
    
class number_questions(BaseModel):
    user_questions: int


@app.post("/generate_quiz/")
async def genera_cuestionario(input: DataInput):
    level = "easy" if input.level == 1 else "medium" if input.level == 2 else "hard"
    
    start_time = time.time()
    print("making one " + level + " questionaire..")
    
    output = create_quiz_generator(server, input.text, level, input.openQuestion)

    end_time = time.time()
    duration = end_time - start_time  
    print(f"Request served in {duration} seconds")
    return  json.dumps(output, indent=2, ensure_ascii=False)


@app.post("/evaluate_answer/")
async def answer_evaluation(input: DataInput):
    start_time = time.time()
    print("evaluating the answer..")
    
    output = evaluator(server, input.text, input.question, input.answer)

    end_time = time.time()
    duration = end_time - start_time  
    print(f"Request served in {duration} seconds")
    return  json.dumps(output, indent=2, ensure_ascii=False)

UPLOAD_DIRECTORY = Path("uploads")
#UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    
@app.post("/semantic_chunking/")
async def semantic_chunking(file: UploadFile = File(...), user_questions: int = Form(4)):
    start_time = time.time()
    print("sppliting the text..")
    #file_location = UPLOAD_DIRECTORY / file.filename
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    with open(file_location, "wb") as buffer: #save the pdf file
        shutil.copyfileobj(file.file, buffer)
    
    chunk = generate_chunk(os.path.abspath(file_location), user_questions)

    end_time = time.time()
    duration = end_time - start_time  
    print(f"Request served in {duration} seconds")
    return  json.dumps(chunk, indent=2, ensure_ascii=False)

