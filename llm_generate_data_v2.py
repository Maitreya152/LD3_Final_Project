import pandas as pd
import torch
from huggingface_hub import login
from vllm import LLM, SamplingParams

# Authenticate
login(token="")

# 1. Configuration
INPUT_CSV = "SHC_Test.csv"
OUTPUT_CSV = "generated_articles_v2.csv"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# 2. Setup vLLM
def main():
    print("Initializing vLLM engine...")
    # Loading in 16-bit precision. Requires ~6-7GB VRAM for a 3B model.
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=2, # Uncomment this if you want to split across 2 GPUs
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        enforce_eager=True
    )

    tokenizer = llm.get_tokenizer()

    # Define generation parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # 3. Load Data and Prepare Prompts
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print("Formatting prompts...")
    prompts = []
    valid_indices = []

    for index, row in df.iterrows():
        headline = row.get('Text', '')
        category = row.get('Label', '')
        
        # Skip truly empty rows to prevent prompt errors
        if pd.isna(headline) or str(headline).strip() == "":
            continue
            
        # Use the tokenizer from the LLM instance
        # We need to pass the tokenizer if we want to format prompt inside main
        # or just use the global one if it's available. 
        # But tokenizer is initialized inside main now.
        
        # Defining format_prompt inside main or passing tokenizer to it.
        # Let's keep format_prompt outside and pass tokenizer.
        prompts.append(format_prompt(headline, category, tokenizer))
        valid_indices.append(index)

    # 4. Batch Generation with vLLM
    print(f"Generating {len(prompts)} articles in batch...")
    # vLLM takes the entire list of strings and processes them automatically
    outputs = llm.generate(prompts, sampling_params)

    # 5. Process Outputs and Clean Text
    # Initialize a list of empties so we map the results back to the correct CSV rows
    generated_articles = ["SKIPPED_OR_ERROR"] * len(df) 

    for i, output in enumerate(outputs):
        original_index = valid_indices[i]
        headline = str(df.at[original_index, 'Text']).strip()
        
        # Extract the generated text from the vLLM RequestOutput object
        generated_text = output.outputs[0].text.strip()
        
        # Post-processing cleanup logic
        if headline in generated_text:
            generated_text = generated_text.replace(headline, "").strip()
            
        if generated_text.startswith(":") or generated_text.startswith("-"):
            generated_text = generated_text[1:].strip()
            
        generated_articles[original_index] = generated_text

    # 6. Save the Results
    df['Generated_Article'] = generated_articles
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Generation complete! Results saved to {OUTPUT_CSV}")

def format_prompt(headline, category, tokenizer):
    headline = str(headline).strip()
    category = str(category).strip()

    messages = [
        {
            "role": "system", 
            "content": "You are a professional Marathi news reporter. Write a detailed, two paragraph long Marathi news article based on the category and headline. CRITICAL RULES: 1. DO NOT print, repeat, or translate the headline. 2. Start immediately with the first paragraph of the news report. 3. Output ONLY Marathi text."
        },
        {
            "role": "user", 
            "content": f"Category: {category}\nHeadline Context: {headline}\n\nWrite the article now. Skip the headline."
        }
    ]
    
    # Pre-format the string using the chat template so vLLM just has to infer
    return tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )

if __name__ == "__main__":
    main()
