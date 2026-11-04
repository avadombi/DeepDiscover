# Getting started

The smallest model has one storage and one process routed to the outlet.

```python
import numpy as np
from deepdiscover import CRCNet, DiscoverUnit, MultiDUModel, DeepDiscover

p = np.array([2, 1, 0, 4, 3], np.float32)
pet = np.ones_like(p)
water = p[None, :, None]             # (batch, time, DU)
drivers = np.stack([p, pet], -1)[None]

net = CRCNet([1, -1, 1], [5, 3, 100])
du = DiscoverUnit([net], context_indices=[(0, 1)])
model = DeepDiscover(MultiDUModel([du], [["outlet"]]))
result = model.simulate(water, drivers)

print(result["outlet"].shape)        # (1, 5, 1)
print(result["storage"].shape)       # (1, 5, 1)
print(result["mass_residual"].numpy().max())
```

The `drivers` tensor contains information available to constitutive networks. Its columns do not automatically enter the water balance. Every conserved water input must be included in `water` or defined as a boundary source.

For a complete training workflow, open the `01_streamflow.ipynb` notebook.
