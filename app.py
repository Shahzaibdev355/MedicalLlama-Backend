import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os

# Initialize FastAPI app
app = FastAPI()

# Define configurations for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Define model and tokenizer IDs
base_model_id = "BioMistral/BioMistral-7B"
peft_model_id = "ShahzaibDev/biomistral-medqa-finetune"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Define the offload folder to store weights temporarily if needed
offload_folder = "./offload_folder"
os.makedirs(offload_folder, exist_ok=True)  # Ensure the folder exists

# Load base model with 4-bit precision
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",  # Automatically uses available devices (GPU, CPU)
    offload_folder=offload_folder,  # Specify where to offload weights if necessary
    torch_dtype=torch.float16  # Use FP16 precision to reduce memory usage
)

# Load the fine-tuned PEFT model
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Function to ask a structured question with a question type
def ask_structured_question(question, question_type, max_new_tokens=300):
    eval_prompt = f"""From the MedQuad MedicalQA Dataset: Given the following medical question and question type, provide an accurate answer:

### Question type:
{question_type}

### Question:
{question}

### Answer:
"""
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**model_input, max_new_tokens=max_new_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Pydantic model to receive questions
class QuestionRequest(BaseModel):
    question: str
    question_type: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        response = ask_structured_question(request.question, request.question_type)
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Test route to check if backend is working
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

# Run the app using uvicorn (in case you need to run it locally for testing)
# uvicorn.run(app, host="0.0.0.0", port=8000)
