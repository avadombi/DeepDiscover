"""Two three-process networks with distinct internal structure."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from deepdiscover import CRCNet, DeepDiscover, DiscoverUnit, MultiDUModel


def build_candidate(kind):
    def net(n):
        return CRCNet([1] * n, [10.0, 100.0] if n == 2 else [10.0, 100.0, 10.0],
                      hidden_units=4)
    if kind == "direct":
        du0 = DiscoverUnit([net(2), net(2)], context_indices=[(0,), (0,)])
        du1 = DiscoverUnit([net(3)], context_indices=[(0,)], upstream_count=1)
        routes = [["outlet", 1], ["outlet"]]
    else:
        du0 = DiscoverUnit([net(2)], context_indices=[(0,)])
        du1 = DiscoverUnit([net(3), net(3)], context_indices=[(0,), (0,)], upstream_count=1)
        routes = [[1], ["outlet", "outlet"]]
    return DeepDiscover(MultiDUModel([du0, du1], routes))


@pytest.mark.parametrize("kind", ["direct", "downstream"])
def test_three_process_structures_conserve_water(kind):
    dd = build_candidate(kind)
    p = np.array([0., 2., 1., 0., 4., 0., 1., 3.], dtype=np.float32)
    water = np.stack([p, np.zeros_like(p)], axis=-1)[None]
    drivers = p[None, :, None]
    result = dd.simulate(water, drivers)
    assert result["outlet"].shape == (1, len(p), 1)
    assert result["storage"].shape == (1, len(p), 2)
    assert result["process_fluxes"].shape == (1, len(p), 3)
    assert dd.model.n_routes == 3
    assert np.max(np.abs(result["mass_residual"].numpy())) < 2e-5


def test_cross_topology_checkpoint_rejected(tmp_path):
    direct = build_candidate("direct")
    downstream = build_candidate("downstream")
    p = np.array([1., 2., 1., 3.], np.float32)
    water = np.stack([p, np.zeros_like(p)], -1)[None]
    drivers = p[None, :, None]
    direct.predict(water, drivers)
    downstream.predict(water, drivers)
    direct.save_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="architecture|topology|parameter"):
        downstream.load_checkpoint(tmp_path)
