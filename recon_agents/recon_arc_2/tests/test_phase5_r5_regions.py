import sys

sys.path.insert(0, '/workspace/recon-platform')

import pytest

from recon_agents.recon_arc_2.hypothesis import HypothesisManager
from recon_engine.node import ReCoNState


@pytest.fixture
def hm():
    return HypothesisManager()


def test_regions_as_alternatives_prefer_higher_alpha_valid(hm):
    # Treat two regions as separate action hypotheses (synthetic action indices)
    region_low = 200
    region_high = 201

    # Create region hypotheses
    hm.create_action_hypothesis(region_low, 0.5, None)
    hm.create_action_hypothesis(region_high, 0.5, None)

    # Set alpha_valid priors favoring region_high; alpha_value equal to avoid ordering bias
    hm.set_alpha_valid({region_low: 0.1, region_high: 0.9})
    hm.set_alpha_value({region_low: 0.0, region_high: 0.0})

    # Build alternatives parent for these regions
    alt = hm.create_alternatives_hypothesis([region_low, region_high])

    # Make terminals succeed
    hm.set_terminal_measurement(region_low, True)
    hm.set_terminal_measurement(region_high, True)

    # Request alternatives parent
    hm.request_hypothesis_test(alt)

    # Early preference: higher prior should progress while lower remains inhibited/inactive at least once
    observed_preference = False
    for _ in range(12):
        hm.propagate_step()
        hs = hm.action_hypotheses[region_high].state
        ls = hm.action_hypotheses[region_low].state
        if hs in (ReCoNState.ACTIVE, ReCoNState.WAITING, ReCoNState.TRUE, ReCoNState.CONFIRMED) \
           and ls in (ReCoNState.INACTIVE, ReCoNState.SUPPRESSED):
            observed_preference = True
            break

    assert observed_preference is True

    # Let execution proceed to completion for the parent
    for _ in range(12):
        hm.propagate_step()
        parent_state = hm.graph.get_node(alt).state
        if parent_state in (ReCoNState.TRUE, ReCoNState.CONFIRMED):
            break

    assert hm.graph.get_node(alt).state in (ReCoNState.TRUE, ReCoNState.CONFIRMED)
