import pytest

from hmlf.environments import DummyHybrid, SequenceWrapper
from hmlf.environments.stage_controller import OneStepPerStageController, StateDependentStageController

STAGE_CALLS = 0


@pytest.fixture
def dummy_stage_function():
    return lambda x: True


@pytest.fixture
def dummy_false_stage_function():
    return lambda x: False


@pytest.fixture
def dummy_reward_function():
    return lambda x: 0.1


@pytest.fixture
def dummy_reward_function_2():
    return lambda x: 2.0


def stage_after_two_calls(obs):
    global STAGE_CALLS
    STAGE_CALLS += 1
    return (STAGE_CALLS % 2) == 0


@pytest.fixture
def dummy_env():
    continuous_dimensions = [1, 2, 1]
    return DummyHybrid(continuous_dimensions)


def test_init(dummy_stage_function, dummy_reward_function):
    n_functions = 2
    controller = OneStepPerStageController()
    assert not controller.can_calculate_reward()
    controller = StateDependentStageController([dummy_stage_function] * n_functions)
    assert not controller.can_calculate_reward()

    controller = OneStepPerStageController([dummy_reward_function] * n_functions)
    assert controller.can_calculate_reward()
    controller = StateDependentStageController([dummy_stage_function] * n_functions, [dummy_reward_function] * n_functions)
    assert controller.can_calculate_reward()

    with pytest.raises(AssertionError):
        StateDependentStageController(1)
    with pytest.raises(AssertionError):
        StateDependentStageController([])
    with pytest.raises(AssertionError):
        StateDependentStageController([dummy_stage_function], 0.1)
    with pytest.raises(AssertionError):
        OneStepPerStageController([])


def test_run(dummy_false_stage_function, dummy_env):
    stage_functions = [dummy_false_stage_function] * dummy_env.n_parameter_spaces
    sequence = [0, 0, 1, 2, 0, 2, 1]

    wrapped_env = SequenceWrapper(dummy_env, sequence, StateDependentStageController(stage_functions))

    for _ in range(2):
        obs = wrapped_env.reset()
        for action in sequence[:1]:
            assert obs[0] == action
            obs, r, done, info = wrapped_env.step(wrapped_env.action_space.sample())


def test_run_same_stage(dummy_false_stage_function, dummy_env):
    stage_functions = [dummy_false_stage_function] * dummy_env.n_parameter_spaces
    sequence = [0, 0, 1, 2, 0, 2, 1]

    wrapped_env = SequenceWrapper(dummy_env, sequence, StateDependentStageController(stage_functions))

    for _ in range(2):
        obs = wrapped_env.reset()
        for _ in range(10):
            assert obs[0] == sequence[0]
            obs, r, done, info = wrapped_env.step(wrapped_env.action_space.sample())


def test_run_two_calls_stage(dummy_env, dummy_reward_function, dummy_reward_function_2):
    stage_functions = [stage_after_two_calls] * dummy_env.n_parameter_spaces
    reward_functions = []
    sequence = [0, 0, 1, 2, 0, 2, 1]

    unique_sorted_in_order = sorted(list(set(sequence)))
    for action in unique_sorted_in_order:
        if action <= 1:
            reward_functions.append(dummy_reward_function)
        else:
            reward_functions.append(dummy_reward_function_2)

    controller = StateDependentStageController(stage_functions, reward_functions)
    wrapped_env = SequenceWrapper(dummy_env, sequence, controller)

    for _ in range(2):
        obs = wrapped_env.reset()
        for i in range(len(sequence) * 2):
            action = sequence[i // 2]
            assert obs[0] == action
            obs, r, done, info = wrapped_env.step(wrapped_env.action_space.sample())

            if action <= 1:
                assert r == 0.1
            else:
                assert r == 2.0
