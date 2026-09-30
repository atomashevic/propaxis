# Constructing a text measure

A useful axis begins with a clear measurement question. Write down what a high score means, what the relevant alternative is, and which texts the measure will be used on.

## Define the target

Distinguish a topic from a claim about that topic. “Public transportation” is a topic. “The city should invest more in public transportation” states a policy position. A classifier for topic presence and a measure of policy support need different propositions and validation examples.

Start with one component. Add another only when the definition requires a distinct dimension. For each component, record a brief description and the theoretical or substantive basis for its propositions.

## Write positive and contrast propositions

The positive pole expresses the target. The contrast pole describes the alternative against which it should be measured. Neither pole is automatically good, bad, correct, or incorrect.

| Useful practice | Why it matters |
| --- | --- |
| Write complete assertions | They specify a relationship rather than only naming a topic. |
| Use several formulations | An axis should not depend entirely on one phrase. |
| Keep the topic comparable | Otherwise the direction can measure topic differences. |
| Compare grammatical form, length, and emotional tone | These features can enter the axis alongside the construct. |
| Include plausible alternatives | Random unrelated sentences create a poorly defined contrast. |
| Review each proposition separately | Repeated wording can unintentionally weight an idea more heavily. |

Equal numbers on the two poles are not required: each pole contributes its average embedding. The ordering within each pole does not affect the axis. There is no fitted pairing of positive and contrast sentences.

Direct negation is one way to state an alternative, but embedding models may place a claim and its negation close together. Test actual behavior rather than assuming a grammatical negation creates a good contrast. For example, compare “We should invest more in transit” against both “We should not invest more in transit” and “We should reduce transit funding.” Choose formulations based on the construct and development evidence, not the final test result.

## Choose the unit and aggregation

Use the unit that matches your research question and available labels. A sentence can contain a claim, while a longer passage can supply context, resolve a pronoun, or include a rebuttal.

- `window=1`: individual sentences or word-capped sentence segments.
- `window=3`: overlapping groups of three segments, with stride one.
- `top_k=1`: strongest passage in each document.
- `top_k=8`: average of the eight strongest passages, or all if fewer are available.
- `top_k=None`: average across every passage.

Top-k scores describe the selected passages. They do not estimate how much of the document belongs to a class. Longer documents provide more opportunities for a strong passage, and overlapping passages can repeat the same sentence. Check length effects and inspect the selected text.

## Choose the reference dataset

Reference texts set the numerical scale. They should represent the application population, including weak, strong, and unrelated expression of the construct where relevant. For short-message applications, use short messages rather than maxima from long speeches.

Fit once, save the instrument, and score new documents using the saved parameters. Avoid recalculating the scale for each batch. If your reference population changes, fit a new instrument and retain its distinct identifier.

The library requires at least three reference documents and rejects axes with no projection variation. These are computational checks. There is no universal sufficient reference-sample size. Use a substantially larger sample for research and examine how scores change across plausible reference samples.

## Inspect before classifying

Review the anchor projections, cosine between axes, and leave-one-out stability in `instrument.diagnostics`. Read examples from both ends and the middle of the score distribution. Look for sensitivity to names, topics, sentiment, document length, and genre.

Create a challenge set that distinguishes the target from related concepts. For a claim-based measure, include assertions, quotations, explicit denials, attribution to others, and ordinary discussion of the same subject. Human coding should be independent of model scores.

## Add labels only when the task requires them

Continuous scores can be useful without a binary threshold. If you need labels, estimate a threshold on a dedicated calibration sample. Hold a separate test sample for final evaluation. Repeated authors, speakers, or source documents should remain together when partitioning data.

Choosing propositions, an encoder, aggregation settings, or thresholds using labels is part of model development, even though encoder weights are unchanged. Report those decisions and the data used for each. The package does not automatically create or audit data partitions.

## Revise and version

Use `dataclasses.replace` to create a new specification while preserving the old one:

```python
from dataclasses import replace

revised_component = replace(
    construct.components[0],
    positive=construct.components[0].positive + ("An additional expression of the claim.",),
)
revised_construct = replace(
    construct,
    components=(revised_component,) + construct.components[1:],
)
revised_instrument = revised_construct.fit(reference_texts, encoder=encoder)
```

The new instrument has a different identifier. Its scales and classification thresholds must be estimated for the new specification. A saved instrument is not edited in place.
