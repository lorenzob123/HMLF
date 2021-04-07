from typing import List


class SequenceCurator:
    """
    A class that manages the sequence of actions. It is used in the `SequenceWrapper`

    :param: The sequence of the skills.
    """

    def __init__(self, sequence: List[int]) -> None:
        self.sequence = sequence
        self.current_action_index = 0

        self._validate_arguments()

    def _validate_arguments(self) -> None:
        assert isinstance(self.sequence, list)
        assert len(self.sequence) > 0

    def reset(self):
        """
        Resets the sequence curator to the initial state.
        """
        self.current_action_index = 0

    def get_current(self) -> int:
        """
        Returns the current part of the sequence.

        :return: The sequence at the current position.
        """
        return self.sequence[self.current_action_index]

    def next(self) -> int:
        """
        Increments the current position.

        :return: The new position.
        """
        if self.has_next():
            self.current_action_index += 1
            return self.get_current()
        else:
            raise IndexError("SequenceCurator reached end of current sequence.")

    def has_next(self) -> bool:
        """
        Whether the sequence has a next element.

        :return: Whether the sequence has a next element.
        """
        return self.current_action_index < (len(self.sequence) - 1)
