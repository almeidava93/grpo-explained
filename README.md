# mini-GRPO
A minimal and detailed Python implementation of the Group Relative Policy Optimization (GRPO).

The GRPO loss function has three main components:
- *policy gradient loss*: incentivizes the model to produce responses with higher rewards.
- *KL divergence loss*: measures the divergence between the current model and a reference model. Serves as an strategy to avoid greater big changes with respect to the original model. 
- *entropy loss*: incentivizes higher entropy, which means more exploration and diversity in model generations.

## Mathematical description
The GRPO loss is described as the following formula:
$$
\mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{ \substack{q \sim P(Q)  \\ \{o_i\}^G_{i=1} \sim \pi_{\theta_{old}} (O|q)}} [\mathcal{L}^\backprime_{PG-GRPO}(\cdot | \theta) + \beta \mathcal{L}^\backprime_{KL}(\cdot, \theta, \theta_{ref}) + \alpha \mathcal{L}^\backprime_{Entropy}(\cdot, \theta)]
$$

- $E$ is the expected value. The expected value of a random variable with a finite number of outcomes is a weighted average of all possible outcomes.
- $\theta$ represents the trainable parameters of the current model or the vector of the network weights. $\theta_{ref}$ represents the trainable parameters of the reference model.
- $\pi_\theta$  represents a model with parameters $\theta$ .
- $q \sim P(Q)$ means to randomly pick a question q from the dataset P(Q).
- $\{o_i\}^G_{i=1} \sim \pi_{old}(O|q)$ means that a set of outputs of length G is drawn from the old policy’s distribution over outputs O given the question q.
- “$\cdot$” is the abbreviation of the sampled prompt-responses for the question q: $\{q, \{o_i\}_{i=1}^G\}$
- $\mathcal{L}^\backprime_{PG-GRPO}(\cdot | \theta)$ is the **policy gradient loss.**
- $\beta \mathcal{L}^\backprime_{KL}(\cdot, \theta, \theta_{ref})$ is the **KL divergence loss** scaled by a hyperparameter $\beta$.
- $\alpha \mathcal{L}^\backprime_{Entropy}(\cdot, \theta)$ is the **entropy loss** scaled by a hyperparameter  $\alpha$.