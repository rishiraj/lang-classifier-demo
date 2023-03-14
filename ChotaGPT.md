## Training/Finetuning Process

Clone the base Microsoft GODEL model
```
!git clone https://github.com/microsoft/GODEL.git
%cd GODEL
```
Requirement file has error, keep only nltk==3.7 in requirement.txt and change torch version to 1.13.1 and absl-py version to 1.0.0
```
# !pip install -q nltk==3.7 jedi==0.10 torch==1.13.1 absl-py==1.0.0
!pip install -q -r requirements.txt
!export PYTHONPATH="`pwd`"
```
Create GODEL compatible dataset of Dialog System Technology Challenge for handling chit chat conversations
```
%cd examples/dstc9
!bash create_data.sh
```
Training python file has error, change the following in train.py
```
parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/GODEL-v1_1-base-seq2seq",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
```
Start training process
```
%cd
%cd /content/GODEL/GODEL
!python train.py --dataset_name ../examples/dstc9/dstc9_dataset.py   \
	--output_dir ../examples/dstc9/ckpt   \
	--per_device_train_batch_size=16  \
	--per_device_eval_batch_size=16  \
	--max_target_length 128  \
	--max_length 512  \
	--num_train_epochs 50  \
	--save_steps 10000  \
	--num_beams 5  \
	--exp_name wow-test \
	--preprocessing_num_workers 24 \
	--save_every_checkpoint 
```

## Python Usage

Install the transformers library
```
!pip install -q transformers
```
Download and install the trained model & tokenizer
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("rishiraj/ChotaGPT-mini", use_auth_token="hf_go_get_your_own_token")
model = AutoModelForSeq2SeqLM.from_pretrained("rishiraj/ChotaGPT-mini", use_auth_token="hf_go_get_your_own_token")
```
Setup a context from where answers are to be generated
```
context = "Cricket is a popular sport that is played between two teams of eleven players each. It originated in England in the 16th century and has since spread to other parts of the world, especially countries like India, Australia, and South Africa. The game is played on a circular or oval-shaped field, with a rectangular strip in the middle called the pitch. The objective of the game is to score more runs than the opposing team while also getting all of their players out. There are several rules of cricket that players must abide by. The first and most important rule is that each team gets a turn to bat and a turn to bowl. The team that is batting tries to score runs by hitting the ball that is bowled by the opposing team. The team that is bowling tries to get the batsmen out by hitting the stumps with the ball, catching the ball that is hit by the batsman, or getting the batsman out in other ways. There are several ways that a batsman can be out in cricket. The most common way is for the bowler to hit the stumps with the ball, also known as a bowled dismissal. The second most common way is for the fielding team to catch the ball that is hit by the batsman, also known as a caught dismissal. Other ways to get a batsman out include hitting the ball and then hitting the stumps before the batsman can get back to their crease, hitting the batsman's leg in front of the stumps, or hitting the batsman's hand while they are attempting to hit the ball. In addition to the basic rules of cricket, there are also several other rules that players must follow. For example, players are not allowed to tamper with the ball in any way, and they must also wear protective gear such as helmets and pads. Additionally, there are rules about how the pitch should be maintained and how the ball should be bowled. Overall, cricket is a complex and nuanced sport with many rules and regulations."
```
Define the function to call the model
```
def chat(message):
    instruction = f'Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.'
    knowledge = '[KNOWLEDGE] ' + context
    dialog = ' EOS '.join([message])
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output)
    return output
```
Call the function with your message
```
chat("what is the objective of the game cricket?")
```

## API Usage

Create a FastAPI main.py app with two endpoints and a json for storing data
```
import json
from fastapi import FastAPI
app = FastAPI()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("rishiraj/ChotaGPT", use_auth_token="hf_go_get_your_own_token")
model = AutoModelForSeq2SeqLM.from_pretrained("rishiraj/ChotaGPT", use_auth_token="hf_go_get_your_own_token")

try:
    with open("bots.json", "r") as f:
        bots = json.load(f)
except:
    bots = {}

@app.get("/{bot_id}/context/{message}")
async def set_context(bot_id: str, message: str):
    global bots
    bots[bot_id] = message
    with open("bots.json", "w") as f:
        json.dump(bots, f)
    return {"response": "Context updated successfully for " + bot_id}


@app.get("/{bot_id}/chat/{message}")
async def chat(bot_id: str, message: str):
    global tokenizer, model, bots
    context = bots[bot_id]
    instruction = f'Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.'
    knowledge = '[KNOWLEDGE] ' + context
    dialog = ' EOS '.join([message])
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").to("cuda:0").input_ids
    outputs = model.to("cuda:0").generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": output}
```
Run the API
```
uvicorn main:app --reload
```

## Performance

Large / high accuracy model on GPU: 0.7 sec
Large / high accuracy model on CPU: 13 sec
Restructured mini model without optimization on CPU: 3.78 sec
Restructured mini model with optimization on CPU: 1.7 sec
