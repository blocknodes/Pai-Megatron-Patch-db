from transformers import AutoTokenizer                                                                               
import transformers
import torch
import sys

model = "meta-llama/Llama-2-7b-chat-hf"
model = "../step180000_token707B"
#model = "test/my_hf_model"
#model = "my_hf_model"
#model = "../compare_1000step/base/"
#model = "../compare_1000step/updated/"
#model = "../compare_1000step/updated500/"
model=sys.argv[1]
inputs=sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

sequences = pipeline(
    #'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    #'老抽和生抽的区别是什么?\n',
    inputs+'\n',
    #do_sample=True,
    do_sample=False,
    #top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
