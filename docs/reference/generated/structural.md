# `deepdiscover.structural`

Public classes and functions of this module.

## `Candidate`

## `compare_structures`

```python
compare_structures(candidates: list[Candidate], train: HydroBatch, validation: HydroBatch, *, seeds=(42, 200, 22), epochs=100, batch_size=8, learning_rate=0.001, loss='mse', weights=None, patience=15, normalize=True, mass_tolerance=0.001)
```

## `equivalent_candidates`

```python
equivalent_candidates(results, reference, loss_tolerance)
```

## `save_comparison`

```python
save_comparison(results, path)
```
