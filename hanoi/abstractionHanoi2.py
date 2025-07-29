import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pickle


# Set up directories
def set_up_dir(path):
   os.makedirs(path, exist_ok=True)


# Define custom stopping criterion
class StopOnAnyTokenSequence(StoppingCriteria):
   def __init__(self, stop_sequences_ids):
       self.stop_sequences_ids = stop_sequences_ids


   def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
       for seq in self.stop_sequences_ids:
           if input_ids.shape[1] >= len(seq):
               if torch.equal(input_ids[0, -len(seq):], torch.tensor(seq, device=input_ids.device)):
                   return True
       return False


# Dimensionality reduction
def get_projection(embeddings, method, labels, ax, c=None, plot=False):
   if method == "pca":
       reducer = PCA(n_components=2)
   elif method == "tsne":
       reducer = TSNE(n_components=2, perplexity=20)
   elif method == "umap":
       reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
   else:
       raise ValueError("Unsupported method.")
   reduced = reducer.fit_transform(embeddings)
   if plot:
       for i, label in enumerate(labels):
           x, y = reduced[i]
           ax.scatter(x, y, color=c[i], alpha=0.7, s=60)
           ax.text(x + np.random.normal(0, 0.01), y, label, fontsize=8)
   return reduced


# Setup
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
   model_name,
   trust_remote_code=True,
   config=model_config,
   device_map='auto',
   torch_dtype=torch.float16
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_module = model.model.embed_tokens


# Prompt
sentence_list = [("A = [1, 2, 3], B = [], C = []", "Yes")]


request_template = [
   'Given three lists, A, B, and C, where only the largest element in each list can be moved to another list, and an element can only be added to a list if it is larger than all other elements in the list, determine if all elements of list A can be moved to list C in 5 moves.\nOnly answer "Yes" or "No".\nThe lists are: {}\nAnswer: '
]


answer_tokens = ['yes', 'no', 'Yes', 'No']
stop_token_ids = [tokenizer(t, add_special_tokens=False).input_ids for t in answer_tokens]
yes_ids = [tokenizer('yes', add_special_tokens=False).input_ids]
no_ids = [tokenizer('no', add_special_tokens=False).input_ids]


res_dir = 'algeval/list_reasoning'
set_up_dir(res_dir)


for idx, (sentence, target_answer) in enumerate(sentence_list):
   request = request_template[0].format(sentence)


   messages = [{"role": "user", "content": request}]
   tokenized_chat = tokenizer.apply_chat_template(
       messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
   ).to(model.device)


   stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_token_ids)])
   prompt_length = tokenized_chat.shape[1]


   torch.manual_seed(1)
   output_until_answer = model.generate(
       tokenized_chat,
       max_new_tokens=5,
       eos_token_id=tokenizer.eos_token_id,
       pad_token_id=tokenizer.eos_token_id,
       do_sample=False,
       stopping_criteria=stopping_criteria
   )


   full_decoded = tokenizer.decode(output_until_answer[0], skip_special_tokens=True)
   answer_only = tokenizer.decode(output_until_answer[0][prompt_length:], skip_special_tokens=True)
   print("Model response:", full_decoded)


   tokenized_chat_until_answer = output_until_answer[:, :-1]
   embeddings_out = embedding_module(tokenized_chat_until_answer)
   embeddings_ = embeddings_out.detach().requires_grad_(True)


   torch.manual_seed(1)
   output_xai = model.generate.__wrapped__(
       model,
       inputs_embeds=embeddings_,
       max_new_tokens=1,
       eos_token_id=tokenizer.eos_token_id,
       pad_token_id=tokenizer.eos_token_id,
       do_sample=False,
       stopping_criteria=stopping_criteria,
       return_dict_in_generate=True,
       output_scores=True,
       output_logits=True
   )


   next_token_id = output_xai.logits[-1].argmax()
   token_explained = tokenizer.decode([next_token_id])
   print("Explained token:", token_explained)


   # Get hidden states for visualization
   output = model(inputs_embeds=embeddings_, output_hidden_states=True)
   hidden_states = output.hidden_states
   last_layer_states = hidden_states[-1][0].detach().cpu().numpy()


   # Get embedding vectors for 'yes' and 'no'
   yes_vec = model.lm_head.weight[tokenizer('yes', add_special_tokens=False).input_ids[0]].detach().cpu().numpy()
   no_vec = model.lm_head.weight[tokenizer('no', add_special_tokens=False).input_ids[0]].detach().cpu().numpy()


   # Combine all embeddings for projection
   all_vecs = [yes_vec, no_vec] + list(last_layer_states)
   all_labels = ['yes', 'no'] + [f'token_{i}' for i in range(last_layer_states.shape[0])]
   all_colors = ['green', 'red'] + ['gray'] * last_layer_states.shape[0]


   # Visualization
   fig, ax = plt.subplots(figsize=(8, 6))
   reduced_embeddings = get_projection(
       np.array(all_vecs), method='tsne', labels=all_labels, ax=ax, c=all_colors, plot=True
   )
   ax.set_title("Embedding space: reasoning over list transformation")
   fig.tight_layout()
   plt.savefig(os.path.join(res_dir, f'projection_{idx}.pdf'), dpi=300)
   plt.show()


   # Save hidden states
   pickle.dump(hidden_states, open(os.path.join(res_dir, f'{idx}_hidden_states.p'), 'wb'))