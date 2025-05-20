# Minimal GRPO training loop

import copy
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import re

train_data = [
    {
        "prompt": "<|start_header_id|>system<|end_header_id|>\n\nThis is a conversation between a user and an assistant. The user poses questions. The assistant thinks and then answers the question. The assistant answers with its reasoning within <think></think> and the answer within <answer></answer>.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of Brazil?.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer": "Brasília",
     },
]

# Training params
temperature = 1.0
top_k = 500
I_iterations = 1
M_steps = 50
G = 2
mu_iterations = 1
clip_eps = 5
beta = 0.05
max_tokens = 100
lr = 1e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
curr_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.decode(128004) # '<|finetune_right_pad_id|>'
tokenizer.pad_token_id = 128004 # '<|finetune_right_pad_id|>'

# Setup optimizer
optimizer = torch.optim.AdamW(curr_model.parameters(), lr=lr)

# reward_function
def reward(completion: str, q: dict[str, str]) -> float:
    r = 0
    pattern = r'(?:<think>)([^<]*?)(?:</think>)' # reward for <think> tag
    matches = re.findall(pattern, completion)
    if matches:
       r+=1
    pattern = r'(?:<answer>)([^<]*?)(?:</answer>)' # reward for <answer> tag
    matches = re.findall(pattern, completion)
    if matches:
       r+=1
       if q['answer'] in matches: r += 1 # reward for including answer in <answer> tag
    if q['answer'] in completion: r += 1
    return float(r)

# Group Normalized Advantage
def group_normalised_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    rewards : shape (B, G)   - any real numbers (e.g. 0/1 accuracy)
    returns : shape (B, G)   - normalised advantages A_i
    """
    mean = rewards.mean(dim=-1, keepdim=True) # (B, 1)
    std = rewards.std(dim=-1, keepdim=True).clamp_min(1e-6) # (B, 1)
    return (rewards - mean) / std  # (B, G)

def per_token_logps(model, sequence_ids, mask, logits_to_keep):
    """
    Compute the per-token log probabilities for a batch of sequences using the model's logits.

    Args:
        model: The language model to use for computing logits.
        sequence_ids (torch.Tensor): Tensor of token ids for each sequence, shape (B, T).
        mask (torch.Tensor): Attention mask for the input sequences, shape (B, T).
        logits_to_keep (int): Number of logits to keep from the model output.

    Returns:
        torch.Tensor: Per-token log probabilities, shape (B, T-1).
    """
    model.to(device)
    logits = model(
                input_ids=sequence_ids, attention_mask=mask, logits_to_keep=logits_to_keep+1
            ).logits
    
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    # Divide logits by sampling temperature.
    logits = logits / temperature

    # Get relevant logits
    logits = logits[:,-logits_to_keep:]
    sequence_ids = sequence_ids[:,-logits_to_keep:].unsqueeze(-1)

    # Gather logits from relevant tokens
    selected_logits = torch.gather(logits, dim=-1, index=sequence_ids).squeeze(-1) # (G, T)

    # Compute logits normalizing factor
    logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits]) # (G, T)

    # Compute log probabilities per token
    per_token_logps = selected_logits - logsumexp_values # (G, T) # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps

for i in range(I_iterations):
  ref_model = copy.deepcopy(curr_model)
  # eval mode and freeze ref_model params
  ref_model.eval()
  for param in ref_model.parameters():
    param.requires_grad = False

  for m in range(M_steps):
    batch = [random.choice(train_data)]

    old_policy_model = copy.deepcopy(curr_model)
    # eval mode and freeze old_policy_model params
    old_policy_model.eval()
    for param in old_policy_model.parameters():
      param.requires_grad = False

    curr_model = curr_model.to(device)

    # generate G outputs using the old_policy_model
    for q in batch:
      inputs = tokenizer([q["prompt"] for _ in range(G)], return_tensors="pt")
      inputs = inputs.to(device)
      input_ids = inputs.input_ids
      input_mask = inputs.attention_mask
      input_len = inputs.input_ids.shape[-1]

      old_policy_model.to(device)
      outputs = old_policy_model.generate(**inputs,
                                          temperature=temperature,
                                          top_k=top_k,
                                          output_logits=True, 
                                          return_dict_in_generate=True, 
                                          do_sample=True, 
                                          max_new_tokens=max_tokens, 
                                          pad_token_id=tokenizer.pad_token_type_id)
      
      output_ids = outputs.sequences[:,input_len:] # (G, T)
      output_mask = outputs.sequences[:,input_len:] != tokenizer.pad_token_id # (G, T)
      output_logits = torch.stack(outputs.logits, dim=1) # (G, T, V)
      logits_to_keep = output_ids.size(1)

      mask = torch.cat([input_mask, output_mask], dim=1)

      completions = [
          tokenizer.decode(o, skip_special_tokens=True)
          for o in outputs.sequences[:,input_len:]
      ]
      
      rewards = torch.tensor([reward(c, q) for c in completions]).unsqueeze(0).to(device)
      advantages = group_normalised_advantage(rewards)
      # print("Rewards", rewards.shape, rewards)
      # print("Advantages", advantages.shape, advantages)

      for mu in range(mu_iterations):
        # zero grads in every new iteration
        optimizer.zero_grad()

        # compute current model per_token_logprobs
        new_per_token_logps = per_token_logps(curr_model, outputs.sequences, mask, logits_to_keep)
        
        # compute old model and ref model per_token_logprobs
        with torch.no_grad():
          old_per_token_logps = per_token_logps(old_policy_model, outputs.sequences, mask, logits_to_keep).detach()
          

        # compute policy ratio
        ratio = torch.exp(new_per_token_logps - old_per_token_logps)
        # compute clipped policy ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

        weighted_policy_ratio_1 = advantages.unsqueeze(-1)*ratio # (B, G, T)
        weighted_policy_ratio_2 = advantages.unsqueeze(-1)*clipped_ratio # (B, G, T)

        # Get the minimal value
        per_token_loss = -torch.min(weighted_policy_ratio_1, weighted_policy_ratio_2)

        # compute KL divergence if beta != 0.0
        if beta != 0.0:
          with torch.no_grad():
            ref_per_token_logps = per_token_logps(ref_model, outputs.sequences, mask, logits_to_keep).detach()

          per_token_kl = (
                  torch.exp(ref_per_token_logps - new_per_token_logps) - (ref_per_token_logps - new_per_token_logps) - 1
              )
          
          per_token_loss = per_token_loss + beta * per_token_kl

        # compute loss, maximize the loss
        loss = ((per_token_loss * output_mask).sum(-1) / output_mask.sum(-1).clamp(min=1.0)).mean()
        loss.backward()

        # update curr_model params
        optimizer.step()

        print(f'Iteration {m}: loss {loss.item():.5f}, KL divergence: {per_token_kl.mean().item():.5f}')
        print(f'---'*30)
        print(f'Sample completions\n')
        for idx, (c, r) in enumerate(zip(completions, rewards[0])):
          print(f'{idx+1}) Reward: {r.item():.1f}. Completion: {c}')
        print(f'---'*30)
        print('\n')