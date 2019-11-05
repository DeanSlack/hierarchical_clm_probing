# Weekly Meeting Notes

## 5th November 2019

- Started implementation of experiment code which runs all contextualizers across all tasks (essentially a framework for the project)
- This will include the GLUE tasks, but aims to cover a wide range of the NLP pipeline

### To-Do List

- Begin paper writeup and draft of sections, should be at stage now to know (roughly) how each chapter will be drafted.
- For next week, have more tasks covered with transformer model
- Ideally, we need another model variant working (either a layer mixing model still based on the transformer architecture, or a new architecture). This will allow for a good comparison of results and analysis
- Cover the analysis techniques and metrics to use for the models (e.g. comparing against static layer selection model)
  - Read the paper Noura sent on Slack r.e. SHAP values and probing metrics

## 30th October 2019

- Transformer baseline model implemented, initial performance increase of attention-based layer selection model ~1.6%, needs further testing on more contextualizers and more tasks

### To-Do List

- Cross validate results and collect standard deviations
- Visualization of attention maps for sanity, heatmap and changes over epochs: deadline for next week

## 23rd October 2019

- Initial model performing well w.r.t. single layer contribution per token, not so well when considering a mixing of multiple layers based on attention weightings; needs investigating
- Research potential datasets for hierarchical feature analysis along the same vein as work done for AAAI student paper

### To-Do List

- Visualization of attention maps for sanity, heatmap and changes over epochs
- Run transformer contextual baseline
- Neuron extraction for qualitative assessment, and groupings of functionality per layer

## 15th October 2019

- Push GIT changes. add meeting notes
- Get model to work, confirm if attention mechanism is performing as intended
- Visualization of attention maps for sanity, heatmap and changes over epochs
- If model is rubbish, need to think about adding some heuristics e.g. POS
- Non-linear contextual baseline  for control
- Neuron extraction for qualitative assessment, and groupings of functionality per layer
- Think of a paper title for ACL