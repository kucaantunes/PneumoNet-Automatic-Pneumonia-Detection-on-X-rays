from transformers import GPT2LMHeadModel, GPT2Tokenizer, BartForConditionalGeneration, BartTokenizer

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load BART model and tokenizer
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

def generate_medical_report(prediction):
    if prediction[0][0] > 0.5:
        diagnosis = "Pneumonia detected"
    else:
        diagnosis = "No pneumonia detected"

    prompt = f"Patient diagnosis: {diagnosis}. Please provide a detailed medical report with technical and medical information, including treatment suggestions."

    # Generate text using GPT-2
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    gpt2_outputs = gpt2_model.generate(inputs, max_new_tokens=300, num_return_sequences=1, no_repeat_ngram_size=2)
    gpt2_report = gpt2_tokenizer.decode(gpt2_outputs[0], skip_special_tokens=True)

    # Generate text using BART
    inputs = bart_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    bart_outputs = bart_model.generate(inputs, max_new_tokens=300, num_return_sequences=1, no_repeat_ngram_size=2)
    bart_report = bart_tokenizer.decode(bart_outputs[0], skip_special_tokens=True)

    report = f"{gpt2_report}\n\n{bart_report}"
    return report
