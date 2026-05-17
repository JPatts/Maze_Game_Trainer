# python/recording/game_recorder.py
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

class GameRecorder:
    """Records moves and RL experiences, mirroring the JS version."""

    def __init__(self, config: Optional[Dict] = None):
        self.game_id = str(uuid.uuid4())
        self.step_counter = 0
        self.moves: List[Dict] = []
        self.experiences: List[Dict] = []
        self.start_time = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.outcome: Optional[str] = None
        self.config = config or {
            "boardWidth": 15,
            "boardHeight": 15,
            # ... add your default config here
        }

    def record_move(self, entity: str, action: str, from_row: int, from_col: int,
                    to_row: int, to_col: int, timestamp: float) -> None:
        self.step_counter += 1
        move = {
            "move_id": self.step_counter,
            "game_id": self.game_id,
            "step_number": self.step_counter,
            "entity": entity,
            "action": action,
            "from_row": from_row,
            "from_col": from_col,
            "to_row": to_row,
            "to_col": to_col,
            "timestamp": timestamp,
        }
        self.moves.append(move)
        if entity == "zombie":
            self._record_zombie_experience(action, from_row, from_col, to_row, to_col)

    def _record_zombie_experience(self, action: str, from_row: int, from_col: int,
                                  to_row: int, to_col: int) -> None:
        """Creates an RL experience tuple for Q‑learning."""
        state = [from_row, from_col]
        next_state = [to_row, to_col]
        exp = {
            "exp_id": len(self.experiences) + 1,
            "game_id": self.game_id,
            "step_number": self.step_counter,
            "state": state,
            "action": self._action_to_index(action),
            "reward": None,  # Filled later
            "next_state": next_state,
            "done": False,
            "from_row": from_row,
            "from_col": from_col,
            "to_row": to_row,
            "to_col": to_col,
        }
        self.experiences.append(exp)

    def record_key_collection(self, key_index: int, row: int, col: int) -> None:
        self.step_counter += 1
        self.moves.append({
            "move_id": self.step_counter,
            "game_id": self.game_id,
            "step_number": self.step_counter,
            "entity": "event",
            "action": "KEY_COLLECTED",
            "key_index": key_index,
            "row": row,
            "col": col,
        })

    def record_game_over(self, outcome: str) -> None:
        self.outcome = outcome
        self.end_time = datetime.now(timezone.utc)

    def finalize_experiences(self) -> None:
        """Marks the last zombie experience as terminal."""
        if self.experiences:
            self.experiences[-1]["done"] = True

    def export_session(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "outcome": self.outcome,
            "total_steps": self.step_counter,
            "moves": self.moves,
            "experiences": self.experiences,
        }

    @staticmethod
    def _action_to_index(action: str) -> int:
        mapping = {"up": 0, "right": 1, "down": 2, "left": 3}
        return mapping.get(action.lower(), -1)