import pytest

torch = pytest.importorskip("torch")

from tools.isaac_lab_tasks.reward_functions import RewardFunctionGenerator, RewardTemplateRegistry


class DummyActionManager:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action


class DummyEnv:
    def __init__(self, action: torch.Tensor, device: torch.device | str) -> None:
        self.action_manager = DummyActionManager(action)
        self.num_envs = action.shape[0]
        self.device = device


def test_reward_action_jerk_penalty_computes_expected_penalty():
    generator = RewardFunctionGenerator()
    reward_fn = generator.compile("action_jerk_penalty")
    action = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    env = DummyEnv(action=action, device=torch.device("cpu"))
    env._prev_actions = torch.zeros_like(action)

    reward = reward_fn(env, penalty_scale=0.01)

    expected = torch.tensor([-0.02, -0.005])
    assert torch.allclose(reward, expected)


def test_default_reward_stub_returns_zero_tensor():
    generator = RewardFunctionGenerator()
    reward_fn = generator.compile("custom_reward")
    env = DummyEnv(action=torch.zeros((3, 2)), device=torch.device("cpu"))

    reward = reward_fn(env)

    assert reward.shape == (3,)
    assert torch.allclose(reward, torch.zeros(3))


@pytest.mark.parametrize(
    "source, function_name, error_match",
    [
        (
            """
def reward_bad(env):
    import os
    return torch.zeros(env.num_envs, device=env.device)
""",
            "reward_bad",
            "Disallowed AST node: Import",
        ),
        (
            """
def reward_bad(env):
    return env.__globals__
""",
            "reward_bad",
            "Disallowed attribute access: __globals__",
        ),
        (
            """
def reward_bad(env):
    return eval("1")
""",
            "reward_bad",
            "Disallowed name: eval",
        ),
        (
            """
def reward_bad(env):
    exec("print('nope')")
    return torch.zeros(env.num_envs, device=env.device)
""",
            "reward_bad",
            "Disallowed name: exec",
        ),
    ],
)
def test_reward_source_validation_rejects_unsafe_nodes(source, function_name, error_match):
    with pytest.raises(ValueError, match=error_match):
        RewardTemplateRegistry._compile_source(source, function_name)
