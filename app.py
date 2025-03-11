import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load Model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to Generate Text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio Web UI
def gradio_interface(prompt):
    return generate_text(prompt)

demo = gr.Interface(fn=gradio_interface, inputs="text", outputs="text")
demo.launch()
