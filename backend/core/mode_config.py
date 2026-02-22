"""
Mode Configuration — Sentinel-E Cognitive Engine 3.X

Master mode control object that governs all Omega Kernel executions.

MODE = { STANDARD, RESEARCH }
RESEARCH.sub_modes = { DEBATE, GLASS, EVIDENCE, STRESS }
KILL is NOT a separate mode — it is a diagnostic state inside GLASS.

All runs must be governed by this config. No hidden branching logic.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import uuid
import logging

logger = logging.getLogger("ModeConfig")


# ============================================================
# MODE ENUMS
# ============================================================

class Mode(str, Enum):
    STANDARD = "standard"
    RESEARCH = "research"


class SubMode(str, Enum):
    DEBATE = "debate"
    GLASS = "glass"
    EVIDENCE = "evidence"
    STRESS = "stress"


# ============================================================
# ROLE ASSIGNMENT
# ============================================================

class DebateRole(str, Enum):
    FOR = "for"
    AGAINST = "against"
    JUDGE = "judge"
    NEUTRAL = "neutral"


# ============================================================
# MODE CONFIG — Master Control Object
# ============================================================

@dataclass
class ModeConfig:
    """
    Master mode control object. All Omega Kernel runs are governed by this.
    
    mode: STANDARD | RESEARCH
    sub_mode: DEBATE | GLASS | EVIDENCE | STRESS (only used when mode=RESEARCH)
    rounds: Number of debate/stress rounds
    role_map: Model-to-role assignments for debate
    adaptive_learning: Whether to apply knowledge learner weights
    kill_override: Diagnostic state inside GLASS only
    """
    # Core fields
    text: str
    mode: Mode = Mode.STANDARD
    sub_mode: SubMode = SubMode.DEBATE
    rounds: int = 3
    role_map: Dict[str, DebateRole] = field(default_factory=dict)
    adaptive_learning: bool = True
    kill_override: bool = False
    
    # Session fields
    chat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: List[Dict[str, str]] = field(default_factory=list)
    enable_shadow: bool = False

    def __post_init__(self):
        # Normalize mode from legacy strings
        if isinstance(self.mode, str):
            mode_map = {
                "standard": Mode.STANDARD,
                "conversational": Mode.STANDARD,
                "forensic": Mode.STANDARD,
                "experimental": Mode.RESEARCH,
                "research": Mode.RESEARCH,
                "kill": Mode.RESEARCH,  # Kill → RESEARCH/GLASS with kill_override
            }
            raw = self.mode.lower() if isinstance(self.mode, str) else "standard"
            if raw == "kill":
                self.kill_override = True
                self.sub_mode = SubMode.GLASS
            self.mode = mode_map.get(raw, Mode.STANDARD)

        # Normalize sub_mode
        if isinstance(self.sub_mode, str):
            sub_map = {
                "debate": SubMode.DEBATE,
                "glass": SubMode.GLASS,
                "evidence": SubMode.EVIDENCE,
                "stress": SubMode.STRESS,
            }
            self.sub_mode = sub_map.get(self.sub_mode.lower(), SubMode.DEBATE)

        # kill_override only valid inside GLASS
        if self.kill_override and self.sub_mode != SubMode.GLASS:
            logger.warning("kill_override only valid in GLASS sub-mode. Forcing sub_mode=GLASS.")
            self.sub_mode = SubMode.GLASS

        # Clamp rounds
        self.rounds = max(1, min(self.rounds, 10))

        # Default role map if empty in debate mode
        if self.sub_mode == SubMode.DEBATE and not self.role_map:
            self.role_map = {
                "qwen": DebateRole.NEUTRAL,
                "groq": DebateRole.NEUTRAL,
                "mistral": DebateRole.NEUTRAL,
            }

    @property
    def is_research(self) -> bool:
        return self.mode == Mode.RESEARCH

    @property
    def is_standard(self) -> bool:
        return self.mode == Mode.STANDARD

    @property
    def is_debate(self) -> bool:
        return self.is_research and self.sub_mode == SubMode.DEBATE

    @property
    def is_glass(self) -> bool:
        return self.is_research and self.sub_mode == SubMode.GLASS

    @property
    def is_evidence(self) -> bool:
        return self.is_research and self.sub_mode == SubMode.EVIDENCE

    @property
    def is_stress(self) -> bool:
        return self.is_research and self.sub_mode == SubMode.STRESS

    @property
    def is_kill(self) -> bool:
        return self.is_glass and self.kill_override

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "sub_mode": self.sub_mode.value if self.is_research else None,
            "rounds": self.rounds,
            "role_map": {k: v.value for k, v in self.role_map.items()},
            "adaptive_learning": self.adaptive_learning,
            "kill_override": self.kill_override,
            "chat_id": self.chat_id,
        }

    @classmethod
    def from_legacy(cls, text: str, mode: str, sub_mode: str = "debate",
                    rounds: int = 3, chat_id: str = None,
                    history: list = None, enable_shadow: bool = False,
                    kill_switch: bool = False, role_map: dict = None) -> "ModeConfig":
        """Create ModeConfig from legacy API parameters."""
        config = cls(
            text=text,
            mode=mode,
            sub_mode=sub_mode,
            rounds=rounds,
            chat_id=chat_id or str(uuid.uuid4()),
            history=history or [],
            enable_shadow=enable_shadow,
            kill_override=kill_switch,
            role_map=role_map or {},
        )
        return config
