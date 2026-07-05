# Direct-Preference-Optimization-Objective-Function---ANLP-HW3
## Commands

```bash
python -m pytest tests/test_dpo_objective.py
python run_experiment.py --data data/toy_preferences.jsonl --beta 0.5
```

## Test result

```text
collected 8 items
tests/test_dpo_objective.py ........
8 passed in 0.40s
```

## DPO objective

The DPO logit compares how strongly the current policy prefers the chosen answer over the rejected one, relative to the reference policy. The loss is `-log(sigmoid(z))`, implemented with the stable form `np.logaddexp(0, -z)`. Positive logits are good, negative logits show that the chosen answer is not preferred strongly enough.

## Toy experiment

```text
Loaded 12 preference pairs
beta: 0.5
mean DPO loss: 0.6371
preference accuracy from DPO logits: 0.583
```

Some examples:

```text
ex01: logit= 0.350, loss= 0.533
ex03: logit=-0.050, loss= 0.718
ex11: logit=-0.250, loss= 0.826
```

## Interpretation

All tests passed, so the implementation works as expected. The toy experiment shows mixed performance: several logits are positive, but some are still negative, and the preference accuracy is only 0.583. This means the policy matches the preferred response in some cases, but there is still noticeable room for improvement.

## Answers to the Questions
**What does a positive DPO logit mean?**  
A positive DPO logit means that the current policy prefers the chosen response over the rejected response more strongly than the reference policy does. In other words, the model is moving in the desired direction for that preference pair.

**Why does DPO compare the current policy against a reference policy?**  
The reference policy acts as a baseline. DPO does not just reward the model for preferring the chosen answer, but for preferring it more than the reference model does. This helps keep training stable and prevents the policy from drifting too far.

**What does the beta parameter control?**  
The `beta` parameter controls how strongly the difference between the current policy and the reference policy affects the loss. A larger `beta` makes the objective more sensitive to preference differences, while a smaller `beta` makes the updates softer.

**Pick one example with a low loss and one example with a high loss. Explain the difference.**  
A low-loss example is `ex04`, with logit `0.400` and loss `0.513`. This means the policy already prefers the chosen response clearly more than the rejected one.  
A high-loss example is `ex11`, with logit `-0.250` and loss `0.826`. This means the policy does not prefer the chosen response enough, and may even favor the rejected one relative to the reference policy.  
So the main difference is that low-loss examples already match the preference signal well, while high-loss examples still disagree with it.
