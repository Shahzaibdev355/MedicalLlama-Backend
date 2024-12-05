import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import gdown

# FastAPI app initialization
app = FastAPI()

# Google Drive file URLs
pytorch_model_url = "https://drive.google.com/uc?export=download&id=1zCasi0vGr8lTnqZqmxlsqK4nb2HagNpG"  # Replace YOUR_FILE_ID
model_path = "./model_file"

# Ensure the model folder exists locally
if not os.path.exists(model_path):
    os.makedirs(model_path)

# Download pytorch_model.bin if not already available
bin_file_path = os.path.join(model_path, "pytorch_model.bin")
if not os.path.exists(bin_file_path):
    print("Downloading pytorch_model.bin from Google Drive...")
    gdown.download(pytorch_model_url, bin_file_path, quiet=False)

# Use Hugging Face base model for config and tokenizer files
base_model_id = "BioMistral/BioMistral-7B"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load base model with your local weight
print("Loading the base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    state_dict=torch.load(bin_file_path, map_location="cpu"),  # Load custom weights
    device_map="cpu",  # Explicitly load to CPU
)

# Load the fine-tuned PEFT model
print("Loading the fine-tuned PEFT model...")
peft_model_id = "ShahzaibDev/biomistral-medqa-finetune"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Define request format
class QuestionRequest(BaseModel):
    question: str
    question_type: str = None  # Optional for structured questions

# Health check endpoint
@app.get("/")
async def health_check():
    return JSONResponse(content={"status": "Backend is live!"})

# API endpoint for asking questions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    # Generate prompt for structured or simple question
    if request.question_type:
        prompt = f"""From the MedQuad MedicalQA Dataset: Given the following medical question and question type, provide an accurate answer:
### Question type:
{request.question_type}
### Question:
{request.question}
### Answer:
"""
    else:
        prompt = request.question

    # Tokenize input and ensure it's on CPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Generate response
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse(content={"answer": answer})
