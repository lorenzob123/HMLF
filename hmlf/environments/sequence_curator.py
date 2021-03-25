from typing import List


class SequenceCurator:
    def __init__(self, sequence: List[int]) -> None:
        self.sequence = sequence
        self.current_action_index = 0

        self._validate_arguments()

    def _validate_arguments(self) -> None:
        assert isinstance(self.sequence, list)
        assert len(self.sequence) > 0

    def reset(self):
        self.current_action_index = 0

    def get_current(self) -> int:
        return self.sequence[self.current_action_index]

    def next(self) -> int:
        if self.has_next():
            self.current_action_index += 1
            return self.get_current()
        else:
            raise IndexError("SequenceCurator reached end of current sequence.")

    def has_next(self) -> bool:
        return self.current_action_index < (len(self.sequence) - 1)
