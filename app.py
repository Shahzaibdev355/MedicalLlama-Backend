import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gdown
import os
from pydantic import BaseModel

# FastAPI app initialization
app = FastAPI()

# Define configurations for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Model and tokenizer IDs
base_model_id = "BioMistral/BioMistral-7B"
peft_model_id = "ShahzaibDev/biomistral-medqa-finetune"
model_path = "./BioMistral-7B"

# Download base model weights from Google Drive if not available locally
drive_url = "https://drive.google.com/uc?id=1zCasi0vGr8lTnqZqmxlsqK4nb2HagNpG"
if not os.path.exists(model_path):
    print("Downloading model weights from Google Drive...")
    gdown.download(drive_url, f"{model_path}.zip", quiet=False)
    os.system(f"unzip {model_path}.zip -d ./")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load base model with 4-bit precision
print("Loading the base model with 4-bit precision...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    quantization_config=bnb_config
)

# Load the fine-tuned PEFT model
print("Loading the PEFT model...")
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
    if request.question_type:  # Structured question
        eval_prompt = f"""From the MedQuad MedicalQA Dataset: Given the following medical question and question type, provide an accurate answer:

### Question type:
{request.question_type}

### Question:
{request.question}

### Answer:
"""
        inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    else:  # Simple question
        inputs = tokenizer(request.question, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse(content={"answer": answer})

