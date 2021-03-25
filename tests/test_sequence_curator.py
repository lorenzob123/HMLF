import pytest

from hmlf.environments.sequence_curator import SequenceCurator


def test_init_wrong_arguments():
    with pytest.raises(AssertionError):
        SequenceCurator(0.1)
    with pytest.raises(AssertionError):
        SequenceCurator("hallo")
    with pytest.raises(AssertionError):
        SequenceCurator([])


@pytest.mark.parametrize(
    "sequence",
    [
        [1, 0, 2, 3],
        [1],
        [1, 1, 1, 1, 0, 2, 3],
    ],
)
def test_init(sequence):
    os_curator = SequenceCurator(sequence)
    assert os_curator.current_action_index == 0
    assert os_curator.sequence == sequence

    for i, action in enumerate(sequence):
        if i < len(sequence) - 1:
            assert os_curator.has_next()
            assert os_curator.current_action_index == i
            assert os_curator.get_current() == action
            os_curator.next()
        else:
            assert not os_curator.has_next()
            with pytest.raises(IndexError):
                os_curator.next()
