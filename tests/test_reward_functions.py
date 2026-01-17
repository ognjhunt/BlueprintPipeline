import pytest

torch = pytest.importorskip("torch")

from tools.isaac_lab_tasks.reward_functions import RewardFunctionGenerator


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
