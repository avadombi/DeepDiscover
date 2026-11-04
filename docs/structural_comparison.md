# Comparing structural hypotheses

Both three-process two-DU configurations in [model topologies](topologies.md) can be supplied as candidates. The package provides an explicit workflow to compare alternative structures under identical training and validation data. Each candidate is a Python function that creates a new DeepDiscover model. This makes the number of DUs, processes and routes visible in each experiment.

```python
from deepdiscover import Candidate, compare_structures, equivalent_candidates

candidates = [
    Candidate('one_state', build_one_state, {'multiple_states': False}),
    Candidate('two_states', build_two_states, {'multiple_states': True}),
]
results = compare_structures(
    candidates, train, validation, seeds=(42, 200, 22), epochs=150,
)
selected = equivalent_candidates(results, reference='two_states', loss_tolerance=0.001)
```

`build_one_state` and `build_two_states` are user-defined functions returning fresh `DeepDiscover` instances. The example names indicate declared structural properties, with their absence and presence explicitly represented.

Validation equivalence depends on the model envelope, objective, tolerance, training budget and optimization quality. Identifying a property as supported requires testing adequately fitted alternatives in which the property is absent. Numerical mass conservation is checked separately; long-term storage behaviour also needs hydrological examination. One selected topology is not evidence of a unique physical architecture.

The function does not access a test period. An independent test should be used only after the final candidate families and comparison rules are fixed. The package currently supplies direct candidate comparison; it does not silently search and delete graph components.
