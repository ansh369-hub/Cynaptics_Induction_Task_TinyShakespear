import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig



def generate_response(instruction, input_text=None, model_path="./gpt2-large-alpaca-lora"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = PeftModel.from_pretrained(base_model, "./gpt2-large-alpaca-lora")
    model.to(device)
    model.eval() 

   
    if input_text:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )


    inputs = tokenizer(prompt, return_tensors="pt").to(device)


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,         
            temperature=0.7,            
            do_sample=True,             
            pad_token_id=tokenizer.eos_token_id, 
            top_p=0.9,                  
            repetition_penalty=1.2      
        )


    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    response_start = generated_text.find("### Response:\n") + len("### Response:\n")
    final_response = generated_text[response_start:].strip()
    
    return final_response

if __name__ == "__main__":
    print("Loading model for inference...\n")
    
    # Test 1: Instruction without Input
    inst_1 = """Give me three elaborate reasons not to do engineering.Point wise like:
    
    1.

    2.

    3.
    
    """
    print(f"Instruction: {inst_1}")
    print(f"Assistant: {generate_response(inst_1)}\n")
    print("-" * 50)



    # Test 2: Instruction with Input
    inst_2 = "Generate a complete story of 100 words."
    inp_2 = """
      The man on way back to home in his car on a deserted road at night saw a strange figure in the dark
    """
    print(f"\nInstruction: {inst_2}")
    print(f"Input: {inp_2}")
    print(f"Assistant: {generate_response(inst_2, input_text=inp_2)}\n")


    