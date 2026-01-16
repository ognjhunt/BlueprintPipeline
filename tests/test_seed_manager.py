import importlib
import importlib.util
import random

import numpy as np

from tools.config.seed_manager import configure_pipeline_seed, set_global_seed


def test_seed_manager_reproducible_outputs() -> None:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = None
    torch = None
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        torch_state = torch.get_rng_state()

    try:
        set_global_seed(123)
        first = (random.random(), np.random.rand(3).tolist())

        set_global_seed(123)
        second = (random.random(), np.random.rand(3).tolist())

        assert first == second

        if torch is not None:
            set_global_seed(456)
            first_tensor = torch.rand(3)
            set_global_seed(456)
            second_tensor = torch.rand(3)
            assert torch.equal(first_tensor, second_tensor)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)


def test_configure_pipeline_seed_from_env(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_SEED", "321")
    seed = configure_pipeline_seed()

    assert seed == 321
