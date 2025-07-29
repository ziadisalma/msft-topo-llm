import pandas as pd
import numpy as np
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
from os.path import join, exists
from os import makedirs
import torch.nn as nn
from transformers.generation import StoppingCriteria, StoppingCriteriaList
import itertools
import os
import itertools
import matplotlib.cm as cm

import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import string

# Define the custom stopping criteria
class StopOnAnyTokenSequence(StoppingCriteria):
    def __init__(self, stop_sequences_ids):
        self.stop_sequences_ids = stop_sequences_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in self.stop_sequences_ids:
            if input_ids.shape[1] >= len(seq):
                if torch.equal(input_ids[0, -len(seq):], torch.tensor(seq, device=input_ids.device)):
                    print('Stop generation', seq)
                    return True
        return False


def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_projection(embeddings, method, labels, ax, c = None, perplexity=50, fontsize=20, s=100,  n_neighbors=15, min_dist=0.1, random_state=42, plot=False): 
    
    # Perform dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)

    elif method == "umap":

        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, metric='euclidean', init="random")
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    if plot:
        for i, label in enumerate(labels):
            x, y = reduced_embeddings[i]
            #c = color_dict[label]
            ax.scatter(x, y, s=s, alpha=0.7, color= c[i])
            ax.text(x + np.random.normal(0,0.01), y, label, fontsize=fontsize, alpha=0.9)    

    return reduced_embeddings


def get_node_dict(tokenized_chat, node_names):
    
    node_dict = {k: [] for k in node_names}
    for k,v in enumerate(tokenized_chat.detach().squeeze().tolist()):
        token= tokenizer.convert_ids_to_tokens([v])[0]
        if token in node_dict:
            node_dict[token].append(k)

    node_dict_names = {}
    for k,v in node_dict.items():
        for i,v_ in enumerate(v):            
            name = k +' '+ str(i)
            node_dict_names[v_] = k +' '+ str(i)
                
    return node_dict, node_dict_names

def get_token_pair(token_explained):
    if token_explained == 'yes':
        contrast_token = 'no'
    elif  token_explained == 'no':
        contrast_token = 'yes'
    if token_explained == 'Yes':
        contrast_token = 'No'
    elif  token_explained == 'No':
        contrast_token = 'Yes'
    else:
        raise
    return contrast_token

def simplified_forward(layers, norm, rotary_emb, lm_head, inputs_embeds, attention_mask, l_lrp=(None, None)):

    # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    hidden_states = inputs_embeds
    position_embeddings = rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = {}
    all_self_attns = () 
    next_decoder_cache = None

    causal_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)

    # causal_mask = attention_mask  model.model._update_causal_mask(
    #    attention_mask, inputs_embeds, None, None, False
    # )

    answer_probs = {'yes': [] , 'no': []}
    answer_logs = {'yes': [] , 'no': []}

    for l, decoder_layer in enumerate(layers):
        
        all_hidden_states[l] = hidden_states

        if l_lrp[0] is not None:
            if l_lrp[0] == l:
                hidden_states_ = hidden_states.detach().cpu().numpy().squeeze()

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]

    hidden_states = norm(hidden_states)
    
    all_hidden_states[l+1] = hidden_states

    output_scores = lm_head(hidden_states)

    pred_intermediate = output_scores[:, -1, :][0].argmax()
    probs = nn.functional.softmax(output_scores[:,-1,:], dim=-1)
    p = probs[:,int(pred_intermediate)].squeeze().detach().cpu().numpy()

    # Collect last layer probabilities
    if l_lrp[0] == 31:
        str_ = ''
        for tok_, id_ in [('yes', yes_ids), ('no', no_ids)]:

            log1 = output_scores[:,-1,:][:,int(id_[0])].squeeze().detach().cpu().numpy()
            log2 = output_scores[:,-1,:][:,int(id_[1])].squeeze().detach().cpu().numpy()
            
            p1 = probs[:,int(id_[0])].squeeze().detach().cpu().numpy()
            p2 = probs[:,int(id_[1])].squeeze().detach().cpu().numpy()
            str_ +=  tok_ + ' {:0.3f} {:0.3f} \t'.format(p1, p2)


            answer_probs[tok_].append(np.max([p1, p2]))
            answer_logs[tok_].append(np.max([log1, log2]))


        print('final', l, int(pred_intermediate), tokenizer.convert_ids_to_tokens([int(pred_intermediate)]), p)
        print(str_)
    

    return all_hidden_states, output_scores, hidden_states_,  answer_logs, answer_probs

def decode(x):
    txt = tokenizer.decode(x.squeeze(), skip_special_tokens=False)
    return txt


def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB array normalized to [0, 1]."""
    return np.array(mcolors.to_rgb(hex_color))



def get_color_transparent(rgba_color, transparency=100):
    hex_color = "#{:02x}{:02x}{:02x}{:02x}".format(int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255), transparency)
    return hex_color


import re

def extract_isolated_capitals(input_string):
    # Use regex to find single capital letters surrounded by non-word characters or spaces
    return re.findall(r'\b[A-Z]\b', input_string)


from transformers import AutoTokenizer, AutoModelForCausalLM

model_name =  "meta-llama/Meta-Llama-3.1-8B-Instruct"

model_config = AutoConfig.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    quantization_config=None,
    device_map='auto',
    torch_dtype=torch.float16)

_ = model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_module = model.model.embed_tokens

# answer_tokens = ['yes', 'no', 'Yes', 'No', 'Ġno', 'Ġyes'] 
# Define stop tokens (only include single-token ones)
answer_tokens = ['Yes', 'No', 'yes', 'no']
stop_token_ids = []
for tok in answer_tokens:
    ids = tokenizer(tok, add_special_tokens=False).input_ids
    if len(ids) == 1:
        stop_token_ids.append(ids)

# Set up custom stopping criteria
stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_token_ids)])

# Build the prompt
prompt_text = (
    "Given three lists, A, B, and C, where only the largest element in each list can be moved to another list, "
    "and an element can only be added to a list if it is larger than all other elements in the list, "
    "determine if all elements of list A can be moved to list C in 7 moves.\n"
    "Only answer with \"Yes\" or \"No\".\n"
    "The lists are: A = [1, 2, 3], B = [], C = []\n"
    "Answer (choose one: Yes or No):"
)

messages = [{"role": "user", "content": prompt_text}]
tokenized_chat = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

attention_mask = torch.ones_like(tokenized_chat)

# Generate output with stopping
SEED = 1  # or any integer seed you want
torch.manual_seed(SEED)
output_until_answer = model.generate(
    tokenized_chat,
    max_new_tokens=5,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False,
    stopping_criteria=stopping_criteria,
    attention_mask=attention_mask
)

prompt_length = tokenized_chat.shape[1]
response_ids = output_until_answer[0][prompt_length:]
decoded_response = tokenizer.decode(response_ids, skip_special_tokens=True)

print("🧾 Raw model response:", decoded_response)

# Get the predicted token (should match stop token)
if response_ids.shape[0] > 0:
    next_token = response_ids[0].item()
    token_explained = tokenizer.decode([next_token]).strip()
    print("🔍 Token explained (cleaned):", token_explained)

    expected_tokens = [tokenizer.decode(t[0]) for t in stop_token_ids]
    print("✅ Expected stop tokens:", expected_tokens)

    if token_explained not in expected_tokens:
        print(f"⚠ Token \"{token_explained}\" not in expected stop token list.")
        compute_xai = False
        token_explained = "N/A"  # You can choose to skip XAI or return here
else:
    print("⚠ No token was generated.")

# stop_token_ids = [tokenizer(token, add_special_tokens=False).input_ids for token in answer_tokens]

stop_token_ids = []
for token in ['Yes', 'No', 'yes', 'no']:
    ids = tokenizer(token, add_special_tokens=False).input_ids
    if len(ids) == 1:
        stop_token_ids.append(ids)

yes_ids = [9891, 9642]
no_ids = [2201, 2822]

stop_token_ids_decoded = [tokenizer.decode(token) for token in stop_token_ids]
stop_token_ids_decoded_split = [tokenizer.convert_ids_to_tokens(token) for token in stop_token_ids]


green_color = np.array([0, 68, 27])/255.
red_color = np.array([103, 0, 13])/255.

special_string = ''

"""
sentence_list = [
                ("From the hotel lobby, there are various rooms denoted by letter names. The lobby connects to O, which connects to W and Q. The lobby also connects to G, which connects to V and M.", "W", ["W", "O", "Q"], "Yes"),
]
"""

start_room = 'the lobby'

question = "if someone can get to {} from {}?"

"""
request_template =    ['Given a description of a set of rooms, determine ' + question + ' \n'
            'Only answer "Yes" or "No". \n'
            'The description is: {} \n'
              'Answer: ']
"""

request_template = ['Given a description of a set of rooms, determine ' + question + '\n'
            'Answer with only "Yes" or "No".\n'
            'The description is: {} \n'
            'Answer (Yes or No):']

"""
sentence_list = [
    (
        "Given three lists, A, B, and C, where only the largest element in each list can be moved to another list, and an element can only be added to a list if it is larger than all other elements in the list, determine if all elements of list A can be moved to list C in 7 moves?",
        "",  # No "target_room"
        [],  # No "target_branch_nodes"
        "Yes"  # Expected answer (for evaluation or checking)
    ),
]
"""

sentence_list = [
    ("A = [1, 2, 3], B = [], C = []", "Yes"),  # (list description, correct answer)
]

transparency=95
proj_method = 'tsne'
SEED = 1


# res_dir = 'algeval/test_{}'.format(start_room)
res_dir = 'algeval/test_list_move'
set_up_dir(res_dir)

prompts = []

"""
for k, (sentence, target_room, target_branch_nodes, target_answer) in enumerate(sentence_list):

    color_dict = {'lobby': cm.Blues(0.8)} 
    
    # request = request_template[0].format(target_room, start_room, sentence)
    
    request = request_template[0].format(sentence)
"""

for k, (sentence, target_answer) in enumerate(sentence_list):
    request = (
        "Given three lists, A, B, and C, where only the largest element in each list can be moved to another list, "
        "and an element can only be added to a list if it is larger than all other elements in the list, "
        "determine if all elements of list A can be moved to list C in 7 moves?\n"
        "Only answer \"Yes\" or \"No\".\n"
        "The lists are: {}\n"
        "Answer:".format(sentence)
    )

    
    node_names_raw = extract_isolated_capitals(sentence)

    # node_names = ['Ġlobby'] + ['Ġ'+n_ for n_ in node_names_raw]
    
    color_dict = {}
    _ = color_dict.update({n_: green_color for n_ in node_names_raw})
    
    # goal_nodes = ['lobby',  target_room ]
    
    node_names_raw = extract_isolated_capitals(sentence)

    # To avoid failure due to empty node list:
    if node_names_raw:
        node_names = ['Ġlobby'] + ['Ġ'+n_ for n_ in node_names_raw]
        _ = color_dict.update({n_: green_color for n_ in node_names_raw})
    else:
        node_names = []


    messages = [
        {
            "role": "user",
            "content": request,
        },
    ]

    prompts.append('{}) {}'.format(k+1, request))
    
    """
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt").to(model.device)
    """
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = torch.ones_like(tokenized_chat)  # ✅ Fixes the error


    node_dict, node_dict_names = get_node_dict(tokenized_chat, node_names)

    focus_nodes = list(itertools.chain(*node_dict.values()))

    focus_names = [node_dict_names[n_] for n_ in focus_nodes]
        
    prompt_length = tokenized_chat.shape[1]
    
    # Create a stopping criteria list    
    stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_token_ids)])
    
    # Regenerate for explainability but with a stopping criterion, requires seed to be set again
    torch.manual_seed(SEED)
    
    # Generate with a stopping criteria and from input_ids (to find prompt length until answer)
    output_until_answer = model.generate(tokenized_chat,
                                         max_new_tokens=5,
                                         eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.eos_token_id,
                                         repetition_penalty= 1, #config.repetition_penalty,
                                         do_sample=False,
                                         num_return_sequences=1,
                                         stopping_criteria=stopping_criteria,
                                         use_cache=False
                                        )
    
    print(decode(output_until_answer[0]))
    
    responses = tokenizer.decode(output_until_answer[0][prompt_length:], skip_special_tokens=True)
    
    tokenized_chat_until_answer = output_until_answer[:, :-1]
    
    embeddings_out = embedding_module(tokenized_chat_until_answer)
    embeddings_ = embeddings_out.detach().requires_grad_(True)
    
    torch.manual_seed(SEED)
    """
    output_xai = model.generate.__wrapped__(model,
                                            inputs_embeds=embeddings_,
                                            max_new_tokens=1,  # just generate one next token
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.eos_token_id,
                                            repetition_penalty= 1., #config.repetition_penalty,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            stopping_criteria=stopping_criteria,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            output_logits=True,
                                            use_cache=False
                                            )
    """
    # Update this call to fix attention_mask warning

    attention_mask = torch.ones(embeddings_.shape[:-1], dtype=torch.long).to(embeddings_.device)
    output_xai = model.generate.__wrapped__(model,
                                            inputs_embeds=embeddings_,
                                            attention_mask=attention_mask,  # ✅ Fixed here
                                            max_new_tokens=1,
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.eos_token_id,
                                            repetition_penalty=1.,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            stopping_criteria=stopping_criteria,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            output_logits=True,
                                            use_cache=False)
    
    responses = tokenizer.decode(output_until_answer[0].squeeze(), skip_special_tokens=True)

    # Take the logits of the last generated token (which should be the answer token)
    xai_scores = output_xai.scores[-1]
    xai_logits = output_xai.logits[-1]
    
    xai_generated_ids = output_xai.sequences[0]
    xai_output_ids = torch.cat([tokenized_chat_until_answer, xai_generated_ids[None, :]], dim=-1)
    
    
    xai_output_ids_ = xai_output_ids[0].tolist()
    output_words = tokenizer.convert_ids_to_tokens(xai_output_ids_, skip_special_tokens=False)
    output_words = [token.replace('Ġ', '') for token in output_words]
        
    assert xai_scores.argmax() == xai_logits.argmax()
    
    """
    next_token = xai_logits.argmax()
    # token_explained = tokenizer.decode([next_token]).strip()
    token_explained = tokenizer.decode([next_token]).strip().strip(string.punctuation)
    
    orig_logit = xai_logits[:, next_token]
    
    if token_explained not in list(itertools.chain(*stop_token_ids_decoded_split)):
        print('not in list', token_explained)
        # example['relevance_{}'.format(config.xai)] = 'N/T'
        compute_xai = False
        raise
    
    if token_explained not in list(itertools.chain(*stop_token_ids_decoded_split)):
        print('⚠ Token "{}" not in expected stop token list: {}'.format(token_explained, stop_token_ids_decoded_split))
        continue  # Skip this example and move to the next
    """

    # Decode and strip punctuation
    next_token = xai_logits.argmax()
    token_explained = tokenizer.decode([next_token]).strip().strip(string.punctuation)

    # Log generated output and token
    print("Decoded output:", tokenizer.decode(xai_output_ids[0], skip_special_tokens=False))
    print("Token explained (cleaned):", token_explained)

    if token_explained not in list(itertools.chain(*stop_token_ids_decoded_split)):
        print('⚠ Token "{}" not in expected stop token list: {}'.format(token_explained, stop_token_ids_decoded_split))
        continue  # Skip this example instead of raising

    responses_xai = tokenizer.decode(xai_output_ids[prompt_length:], skip_special_tokens=False)
    selected_logit = xai_logits[:, next_token]       
    output_xai = model(inputs_embeds=embeddings_, output_hidden_states = True)
    hidden_states = output_xai.hidden_states

    ######### also do it for each layer  #########

    all_states = {}
    attention_mask = torch.ones_like(tokenized_chat_until_answer)
    
    start_ = 0

    all_hidden_focus_nodes = {l:{} for l in range(start_, 32)}
    
    for l in range(start_, 32):
            
        outputs, o2, hidden_states_, answer_logs, answer_probs = simplified_forward(model.model.layers, model.model.norm, model.model.rotary_emb, 
                                         model.lm_head, embeddings_, attention_mask, l_lrp=(l, next_token))

        all_states[l] =  hidden_states_

    assert l+1==32
    
    all_states[l+1] =  outputs[l+1][:,-1,:].detach().cpu().numpy().squeeze()

    contrast_token = get_token_pair(token_explained)

    contrast_id = torch.tensor(tokenizer(contrast_token, add_special_tokens=False).input_ids).cuda()
    class_id = torch.tensor(tokenizer(token_explained, add_special_tokens=False).input_ids).cuda()

    contrast_vec0 = model.lm_head.weight[contrast_id].detach().cpu().numpy().squeeze()
    class_vec0 = model.lm_head.weight[class_id].detach().cpu().numpy().squeeze()
    

    H_all = [contrast_vec0, class_vec0]
    L_all = [ contrast_token.lower(), token_explained.lower() ]
    C_all = ['pink', 'green']


    keys_ = list(range(start_, 32))

    if 31 not in keys_:
        keys_ += [31]

    all_last = []
    for k_ in  keys_:

        h = all_states[k_]

        # Focus nodes only
        h_rep = h[focus_nodes]
        
        L = [node_dict_names[n_].replace('Ġ','') for n_ in focus_nodes]
        c_transparency =  int(((k_+4)/(32+3))*100)


        assert len(L) == len(h_rep)
        for node_, hh in zip(L, h_rep):
            all_hidden_focus_nodes[k_][node_] = hh
        
        C = []

        # Determine the color for the repsective set of nodes/branch
        color_dict_all = {}
        for l_ in L:
            node_ = l_.split(' ')[0]

            ##### GOAL NODES #### (Lobby or other)
            if node_ in goal_nodes:
                if int(l_.split(' ')[1])==0:
                    c_ = get_color_transparent(cm.Purples(0.8), c_transparency)
                    C.append(c_)
                else:
                    if 'lobby' in l_:
                        c_ = get_color_transparent(cm.Blues(0.8), c_transparency)
                    else:
                        c_ = get_color_transparent(green_color, c_transparency)
                    C.append(c_)

            ##### Non-goal NODES #### (Lobby or other)
            else:
                if node_ in target_branch_nodes:
                    c_ = get_color_transparent(green_color, c_transparency )
                    C.append(c_)
                else:
                    c_ = get_color_transparent(red_color, c_transparency )
                    C.append(c_)

            color_dict_all[l_] = c_

        
        all_last.append(all_states[32])
        
    
        H_all.extend(h_rep)
        L_all.extend([l_ + ' ' + str(k_) for l_ in L])
        C_all.extend(C)

        assert len(H_all) == len(C_all)


    # adding final hidden state rep (used for determining the answer)
    H_all += [all_states[32]]
    L_all += ['eos']
    C_all += ['cyan']


    all_hidden_focus_nodes['eos'] = all_states[32]
    all_hidden_focus_nodes['yes'] = class_vec0
    all_hidden_focus_nodes['no'] = contrast_vec0
    
    ########################################################################
    
    score_class = (model.lm_head.weight[class_id]*outputs[l+1][:,-1,:]).sum().detach().cpu().numpy()
    score_contrast = (model.lm_head.weight[contrast_id]*outputs[l+1][:,-1,:]).sum().detach().cpu().numpy()


    print(class_id, score_class, contrast_id, score_contrast)
    print('*****')


    f, ax = plt.subplots(1,1, figsize=(7,5))

    reduced_embeddings = get_projection(np.array(H_all), method=proj_method, labels=L_all, ax=ax, c=C_all,
                                perplexity=20, fontsize=5, s=70,
                                n_neighbors=32,
                                min_dist=.6, plot=True)

    for i, label in enumerate(L_all):
        x, y = reduced_embeddings[i]
        ax.scatter(x, y, s=70, color= C_all[i])

        label_ = label.strip().split(' ')

        fontsize = 10

        if len(label_)>1:
            layer_idx = label_[2]

            if int(layer_idx)==31:
                label = label_[0]  #+ ' ' + label_[2]
                fontsize = 16

            elif int(layer_idx)%4 == 0:
                label =  label_[2]
            else:
                label = ''

        else:
            label = label_[0]
            fontsize = 12
           
        ax.text(x + np.random.normal(0,0.03), y + np.random.normal(0,0.03), label, fontsize=fontsize, alpha=1.)
    

    seq = {}    
    for n_ in node_dict_names.values():
        seq[n_] = []
        for i, label in enumerate(L_all):
            if n_.replace('Ġ','') in label:
                seq[n_].append( reduced_embeddings[i,:])
    
        n_name = n_.replace('Ġ','').split(' ')[0]
        try:
            c_ = color_dict_all[n_.replace('Ġ','')]   #cm.Purples(1.0) if n_name in goal_nodes else color_dict[n_name]
        except:
            import pdb;pdb.set_trace()
        
        ax.plot(np.array(seq[n_])[:,0], np.array(seq[n_])[:,1], alpha=0.5, color=c_) 
    

    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlabel(proj_method + ' 1', fontsize=14)
    ax.set_ylabel(proj_method + ' 2',  fontsize=14)

    str_ = ''
    for kk,vv in answer_probs.items():
        str_ += kk + ': {:0.3f} '.format(vv[-1])
    
    ax.set_title('{}->{} | '.format(start_room, target_room) + str_,) 

    f.tight_layout()
    f.savefig(os.path.join(res_dir, str(k) +'_' +special_string +'_' + proj_method + '.pdf'), dpi=300, transparent=True)
    
    plt.show()

    pickle.dump(all_hidden_focus_nodes, open(os.path.join(res_dir, str(k) + '.p'), 'wb'))

with open(os.path.join(res_dir,  "prompts.txt"), "w") as file:
    file.write("\n".join(prompts))