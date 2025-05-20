# mini-GRPO
This is a minimal Python implementation of the Group Relative Policy Optimization (GRPO). It is a personal exercise of reimplementing the algorithm from scratch and learn in the process. There are plenty of resources online explaining the intuitions that finally lead to GRPO, so this repository is dedicated mostly to the actual implementation of the algorithm. I'll leave the resources that I found most useful at the end. 

## Intuition

There are a couple of key ideas that, together, makes GRPO so successful as an unsupervised way of incentivizing reasoning capabilities in Large Language Models. 

First, it is an algorithm for reinforcement learning with verifiable rewards. That means that, for each task, we need to have an objective way to state that the model answered correctly or not. That works for math and coding, for example. 

Second, as in other reinforcement learning algorithms, the model needs to explore different solutions and receive feedback in order to learn. In this case, the language model generates several completions to the same question/task. Then, each of the completions receives a score based on the verifiable rewards mentioned above. In this aspect, more diversity means more ability to explore the possibilities and find good solutions.

Third, someone needs to design well the rewards. That depends completely on the task you are willing to accomplish. If your objective is not for the model to reason, but to always generate poems in the format of a sonet, you just need to build a rule-based function that scores the model response, and give higher scores to those the approximate to your final goal. This is not exactly a simple task, since you can incitivize model behavvior that hinders its performance. For example, in the sonbet example, the model may repeat several times the same word in the form of a sonet. It gets the reward, but is definitely not a poem. And this is an important moment for good engineering, iterative improvement, and certainly creativity depending on the task in hand. 

Fourth, it is important to not deteriorate the model performance. One way of accomplishing this is by limiting the amount of divergence of the training model from the base model. 

Considering all this, let's take a look at the mathematical definition of GRPO.


## Mathematical description

The GRPO loss function, as implemented in the DeepSeek R1 paper, has two main components:
- *policy gradient loss*: incentivizes the model to produce responses with higher rewards. **Seeks greater rewards.**
- *KL divergence loss*: measures the divergence between the current model and a reference model. Serves as an strategy to avoid greater big changes with respect to the original model. **Limits divergence.**
- *entropy loss*: applied with a negative coefficient, incentivizes higher per-token entropy to encourage exploration and generate more diverse reasoning paths. **Adds diversity.**

The first two aspects were introduced in the [DeepSeek R1 paper](https://arxiv.org/abs/2501.12948) and in the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300). The entropy loss factor is present in another paper entitled [Reinforcement Learning for Reasoning in Large Language Models with One Training Example](https://arxiv.org/abs/2504.20571), by Wang et al.

The GRPO can be described as the following formula:
```math
\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{ \substack{q \sim P(Q)  \\ \{o_i\}^G_{i=1} \sim \pi_{\theta_{old}} (O|q)}} [ \mathcal{L}^\backprime_{PG-GRPO}(\cdot | \theta) + \beta \mathcal{L}^\backprime_{KL}(\cdot, \theta, \theta_{ref}) + \alpha \mathcal{L}^\backprime_{Entropy}(\cdot, \theta) ]
```

- $E$ is the expected value. The expected value of a random variable with a finite number of outcomes is a weighted average of all possible outcomes.
- $\theta$ represents the trainable parameters of the current model or the vector of the network weights. $\theta_{ref}$ represents the trainable parameters of the reference model.
- $\pi_\theta$  represents a model with parameters $\theta$ .
- $q \sim P(Q)$ means to randomly pick a question q from the dataset P(Q).
- $\{o_i\}^G_{i=1} \sim \pi_{old}(O|q)$ means that a set of outputs of length G is drawn from the old policy’s distribution over outputs O given the question q.
- “$\cdot$” is the abbreviation of the sampled prompt-responses for the question q: $\{q, \{o_i\}_{i=1}^G\}$
- $\mathcal{L}^\backprime_{PG-GRPO}(\cdot | \theta)$ is the **policy gradient loss.**
- $\beta \mathcal{L}^\backprime_{KL}(\cdot, \theta, \theta_{ref})$ is the **KL divergence loss** scaled by a hyperparameter $\beta$.
- $\alpha \mathcal{L}^\backprime_{Entropy}(\cdot, \theta)$ is the **entropy loss** scaled by a hyperparameter  $\alpha$.
- $\alpha \mathcal{L}^\backprime_{Entropy}(\cdot, \theta)$ is the entropy loss scaled by a hyperparameter  $\alpha$.


### 1. Policy gradient loss

Defined by:

```math
\mathcal{L}^\backprime_{PG-GRPO}(q, \{o_i\}_{i=1}^G, \theta) = - \frac{1}{G} \sum^G_{i=1} \left(min \left(\frac{\pi_\theta (o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A_i}, clip \left(\frac{\pi_\theta (o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A_i}\right) \right)
```

- $\theta_{ref}$ is the reference model.
- $\frac{\pi_\theta (o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$ measures how much the probability distribution of the model changed with respect to the old response. Each part of this fraction $\pi_\theta (o_i|q)$ and $\pi_{\theta_{old}}(o_i|q)$ represents the probability of each model output that specific sequence of tokens.
- $\varepsilon$  is a hyperparameter of the clipping threshold.
- $\hat{A_i}$ is the group-normalized advantage. This carries information about the quality of the response. A bigger number means greater quality and vice-versa. The hat simply means that the advantage is normalized.

The group-normalized advantage is defined by:

```math
\hat{A_i} = \frac{
    r_i - mean(\{ r_1, r_2, ..., r_G \})
}{
    std(\{r_1, r_2, ..., r_G\})
} \hspace{1cm} i \in [G]
```

```python
def group_normalised_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    rewards : shape (B, G)   – any real numbers (e.g. 0/1 accuracy)
    returns : shape (B, G)   – normalised advantages A_i
    """
    mean = rewards.mean(dim=-1, keepdim=True) # (B, 1)
    std = rewards.std(dim=-1, keepdim=True).clamp_min(1e-6) # (B, 1)
    return (rewards - mean) / std  # (B, G)
```

- $r_i$ is the reward for the generation index $i$.
- This function is a standard normalization of the reward among all generations to the same question. The values are centred by subtracting the mean of the rewards and scaled by dividing by the rewards standard-deviation.

### 2. KL Divergence Loss (Kullback-Leibler Divergence) <a name="section2"></a>

**Entropy** is defined by:

```math
H(P) = -\sum_i p_i log(p_i)
```

```python
def entropy_from_logits(logits: torch.Tensor, mask: torch.Tensor = None):
    """
    logits : shape (B, T, V)
    mask : shape (B, T) with 1 for real tokens, 0 for padding

    Returns scalar mean entropy over *valid/unmasked* positions.
    """
    probs = F.softmax(logits, dim=-1).detach()
    logp = F.log_softmax(logits, dim=-1).detach()
    token_entropy = -(probs * logp).sum(dim=-1) # shape (B, T)

    if mask is None:
        return token_entropy.mean()
    else:
        return (token_entropy * mask).sum() / mask.sum()
```

Entropy is a metric related to the probability of getting that specific probability distribution P. The higher the entropy, the more homogeneous and random the probability distribution is. 

Entropy is a measure of uncertainty that depends only on the true distribution p.
Zero entropy means no surprise (the outcome is certain); higher entropy means more unpredictability.

Cross-entropy is a similar definition but compares two different probability distributions. Defined by:

```math
H(P|Q) = -\sum_i p_i log(q_i)
```

```python
def cross_entropy_from_logits(
    ref_logits: torch.Tensor,
    new_logits: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
  ) -> torch.Tensor:
  """
    Cross‑entropy H(p,q) = - Σ_x  p(x) log q(x)
    where p  = softmax(ref_logits)   (kept fixed)
          q  = softmax(new_logits)   (trainable)

    Shapes
    ref_logits : (B, T, V)
    new_logits : (B, T, V)
    mask       : (B, T)  1 = real token, 0 = padding  (optional)

    Returns
    Scalar tensor – mean cross‑entropy over valid tokens.
    """

  # 1) convert reference logits to probabilities **without grads**
  ref_probs = F.softmax(ref_logits, dim=-1).detach()

  # 2) log‑probs of the *new* policy (grad flows through these)
  log_new_probs = F.log_softmax(new_logits, dim=-1)

  # 3) token‑level cross‑entropy:  H(p,q)  (B,T)
  token_entropy = -(ref_probs * log_new_probs).sum(dim=-1) # shape (B, T)

  # 4) masking & reduction
  if mask is None:
      return token_entropy.mean()
  else:
      # make sure mask is float32 so broadcasting works
      mask = mask.float()
      return (token_entropy * mask).sum() / mask.sum()
```

This is the cross entropy between two probability distributions: P and Q, where P is the reference and Q is the one new we are comparing to the first. It is a weighted mean of the probability of obtained a sequence. 

***Entropy tells you the irreducible uncertainty in a source; cross‑entropy tells you the extra cost you incur when your model of that source is wrong.***

**KL divergence** is then defined by:

```math
\mathbb{D}_{KL}(P||Q) = H(P|Q) - H(P)
```

```python
# KL divergence
def kl_divergence(
    ref_logits: torch.Tensor,
    new_logits: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
  ) -> torch.Tensor:
  """
    Mean token‑level KL divergence  KL(P‖Q)  for a batch.

    Parameters
    ----------
    ref_logits : (B, T, V) frozen reference logits  (P)
    new_logits : (B, T, V) trainable actor logits   (Q)
    mask       : (B, T) 1 = valid token, 0 = pad (optional)
    eps        : small value to avoid div/0 when all tokens are masked

    Returns
    -------
    Scalar tensor — average KL over unmasked tokens.

    Notes
    -----
    * `KL = cross_entropy(P,Q) − entropy(P)`.
    * Gradients flow only through `new_logits`; `ref_logits` is detached.
  """
  return cross_entropy_from_logits(ref_logits, new_logits, mask, eps) - entropy_from_logits(ref_logits, mask)
```

KL divergence is the change in cross-entropy. KL is zero when both distributions are the same. **It is a way to compare distributions.**

**Note:** || reminds us that KL divergence is not symmetric. $KL(P||Q)$ is different from $KL(Q||P)$.

The DeepSeek papers use an approximated version of KL Divergence defined in this [blog post by John Schulman](http://joschu.net/blog/kl-approx.html):

```math
\mathcal{L}^\backprime_{KL}(q, \{o_i\}^G_{i=1}, \theta, \theta_{ref}) = \mathbb{D}_{KL}(\pi_\theta||\pi_{\pi_{ref}})
```

```math
\mathbb{D}_{KL}(\pi_\theta||\pi_{\pi_{ref}}) = \frac{\pi_{\theta{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)} - log\frac{\pi_{\theta{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1
```

```python
import torch
import torch.nn.functional as F
from typing import Optional, Literal

def approximate_kl_divergence(
    ref_logits: torch.Tensor,
    new_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    Schulman first‑order KL surrogate.

    Args
    ----
    ref_logits : (..., K) tensor
        Logits from the reference policy (no gradients wanted).
    new_logits : (..., K) tensor
        Logits from the new policy (gradients flow through this).
    mask       : (...,)  tensor, optional
        Boolean/float mask for padding positions (1 = keep, 0 = ignore).
    eps        : float
        Clamp floor to avoid log/0 and div/0.
    reduction  : {"none","mean","sum"}
        How to reduce across the last dimension (actions) and any masks.

    Returns
    -------
    Tensor
        KL estimate (shape depends on `reduction`).
        Scalar tensor — average approximate KL over unmasked tokens - 'mean' reduction.
    """
    # Probabilities
    ref_probs = F.softmax(ref_logits, dim=-1).detach().clamp_min(eps)
    new_probs = F.softmax(new_logits, dim=-1).clamp_min(eps)

    # Element‑wise surrogate
    ratio = ref_probs / new_probs
    kl_per_token = ratio - torch.log(ratio) - 1  # shape (..., K)

    # Optional masking (broadcast mask to last dim)
    if mask is not None:
        kl_per_token = kl_per_token * mask.unsqueeze(-1).type_as(kl_per_token)

    if reduction == "sum":
        return kl_per_token.sum()
    if reduction == "mean":
        return kl_per_token.mean()
    return kl_per_token  # "none"
```

### 3. Entropy loss
The third term is simply the entropy in the generated responses, and can be computed as above. It is important to not forget about masking padding tokens, since they do not carry meaningful information for the training. 

```math
\mathcal{L}^\backprime_{Entropy}(\cdot, \theta) = \frac{\sum_{b, s} M_{b,s}\cdot H_{b,s}(X)}{\sum_{b,s}M_{b,s}}
```

- $X$ is the generated logits
- $M_{b,s}$ is the mask. $b$ and $s$ index the batch and the sequence position

The python implementation is the same as in the `entropy_from_logits` function in [section above](#2-kl-divergence-loss-kullback-leibler-divergence).

## Minimal Python implementation

The module `grpo.py` contains a full working script using dummy data and the Llama 3.2 1B model. You should be able to run it locally or in Google colab. 

## References
- https://arxiv.org/abs/2501.12948
- https://arxiv.org/abs/2402.03300
- https://arxiv.org/abs/2504.20571
- https://huggingface.co/learn/llm-course/chapter12/3a#advanced-understanding-of-group-relative-policy-optimization-grpo-in-deepseekmath
- https://huggingface.co/blog/NormalUhr/grpo