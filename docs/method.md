# Method and score interpretation

propaxis builds conceptual directions from written propositions. It estimates all document-scaling parameters on an explicit reference dataset, saves them, and applies them unchanged when scoring new documents.

## Axis construction

For component c, let P_c and C_c contain its positive and contrast propositions. Encode every proposition with the same model and preprocessing used for passages. Normalize each anchor vector to unit length. Form the two centroids and their normalized difference:

```text
positive_mean_c = mean(unit(encode(p)) for p in P_c)
contrast_mean_c = mean(unit(encode(p)) for p in C_c)
axis_c = unit(positive_mean_c - contrast_mean_c)
```

A zero-length difference is rejected. Each pole contributes one average, so different pole sizes are permitted. Permuting propositions within a pole does not change the axis. The method does not fit weights to the propositions.

The standard E5 backend uses attention-mask mean pooling with `query: ` prefixed to both anchors and passages, and a default 256-token input limit. It returns raw pooled embeddings. Anchor normalization occurs in axis construction; passage normalization follows centering.

## Passages and centering

Documents are split at punctuation followed by whitespace. Sentences longer than `max_words` are chunked, preferring clause boundaries. Sliding windows contain `window` segments with stride one. Short documents produce one window.

The reference mean μ is the mean of all reference passage embeddings. Longer documents contribute more passages to this estimate. For each passage w:

```text
x_w = unit(encode(w) - μ)
r_c(w) = dot(x_w, axis_c)
z_c(w) = (r_c(w) - reference_projection_mean_c) / reference_projection_sd_c
```

Projection means and standard deviations use all reference passages, with population standard deviations (`ddof=0`). An axis with effectively zero projection variation is rejected.

## Activation and component combination

```text
softplus(a) = log(1 + exp(a))
activation_c(w) = softplus(z_c(w) / temperature)^2 / reference_activation_median_c
```

The default temperature is .25. The implementation uses `logaddexp` for numerical stability and a small positive lower bound for numerical underflow. The reference activation median is computed before dividing by that median.

For a multi-component construct, combine activations using the configured rule:

- Geometric mean: the nth root of the product of n component activations.
- Arithmetic mean: the average component activation.
- Minimum: the smallest component activation.

The geometric mean favors balanced evidence for a fixed sum, but remains compensatory: sqrt(100 × .01) equals sqrt(1 × 1). It is not a logical conjunction or the probability that all claims hold.

## Document aggregation

For the composite score, select passages by their combined activation and average the top k. If fewer than k passages exist, average all available passages. With `top_k=None`, average every passage.

Each component also selects and averages its own highest-activation passages. Thus a component-only measurement can locate relevant evidence even when another component is absent. Ties preserve passage order.

Without NLI, the reported `score` is the document's aggregate activation standardized against the mean and standard deviation of reference-document aggregates. `<component>_score` uses the corresponding component's reference scale. `axis` and `<component>_axis` retain the unstandardized aggregate activations.

## Optional entailment scoring

For each selected passage, the NLI backend scores all written hypotheses as premise–hypothesis pairs. The default is `facebook/bart-large-mnli`. The entailment probability uses the softmax over contradiction, neutral, and entailment logits. The neutral class is retained.

For each component, the window's NLI value is the maximum entailment probability over that component's hypotheses. Components are combined using the specified rule. Document NLI scores average the passages selected by the axis. The implementation evaluates the union of passages needed for component and composite outputs.

The final reported score is:

```text
score(document) = 0.5 * standardized_axis(document)
                + 0.5 * standardized_nli(document)
```

Both standardizations use reference-document statistics. The combination itself is not subsequently standardized to unit variance. A low NLI score does not veto a high axis score. Maximum entailment across hypotheses is not a calibrated probability of a logical disjunction.

If a document-level reference standard deviation is smaller than 1e-12, the implementation uses 1e-12 to keep computation defined. Such a narrow reference distribution can produce very large scores for new data and should be revised. Component projection distributions with effectively zero variance are rejected earlier.

## Diagnostics

For each component, the package reports centroid distance, anchor projections, and the smallest cosine between the original axis and axes reconstructed after removing one anchor. Removal is skipped when it would empty a pole; a resulting zero-length axis receives a stability value of zero. Between-axis cosines indicate geometric overlap between components.

These are descriptive construction checks. They do not demonstrate reliability, independence of constructs, or validity against human coding.

## Classification

`calibrate()` selects a threshold from the distinct calibration scores, including a cutoff above the maximum score. Predictions use `score >= threshold`. It maximizes balanced accuracy or F1. Ties favor the higher threshold. Labels must be binary, and both classes must be present.

Evaluation computes ROC area with tied scores grouped and average precision as the noninterpolated sum of precision times recall increments. It does not calculate cluster-adjusted confidence intervals, perform cross-validation, or calibrate probabilities. Those analyses remain the researcher's responsibility in v0.1.

## Relationship to the research implementation

The centroid-difference axes, reference centering, activation, passage aggregation, and equal-weight NLI fusion derive from the Populism Axis project. This general library makes several explicit changes:

- Any number of named components is supported.
- Reference data are explicit and statistics remain fixed during scoring.
- Component-only outputs use their own selected passages and aggregate activations. Some research scripts instead used standardized projections for component outputs.
- Default input is a one-sentence window with top-one aggregation and E5-small. The populism preset uses three-sentence windows and top-eight aggregation; users choose E5-large separately.
- Reference parameters are estimated afresh; the research project's fitted parameters and datasets are not bundled.
- The package does not apply the research repository's speech-specific metadata cleaning.

Consequently, this release implements a related general workflow, not a certified numerical reproduction of every research benchmark. Validate the complete saved instrument used in your application.
