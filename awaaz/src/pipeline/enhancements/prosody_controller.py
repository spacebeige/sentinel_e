"""
Prosody Controller Module
Turns [P:N] markers into ProsodyPlans.
"""
import re
from dataclasses import dataclass

BASE_PAUSE_MS = {"short": 200, "medium": 400, "long": 700}
FORMALITY_MULTIPLIER = {"SIMPLE": 1.3, "STANDARD": 1.0, "FORMAL": 0.85}
EMERGENCY_MAX_PAUSE_MS = 250
PAUSE_MARKER_RE = re.compile(r'\[P:(\d+)\]')

@dataclass
class ProsodyUnit:
    text: str
    pause_ms: int
    is_emphasis: bool = False

@dataclass
class ProsodyPlan:
    units: list[ProsodyUnit]
    total_pause_ms: int
    estimated_duration_s: float

    @property
    def has_pauses(self) -> bool:
        return self.total_pause_ms > 0

class ProsodyController:
    def strip_markers(self, text: str) -> str:
        return PAUSE_MARKER_RE.sub('', text)

    def inject_natural_pauses(self, text: str) -> str:
        return text

    def build_plan(self, text: str, formality_label: str, accent_region: str, is_emergency: bool) -> ProsodyPlan:
        parts = PAUSE_MARKER_RE.split(text)
        units = []
        total_pause = 0
        
        mult = FORMALITY_MULTIPLIER.get(formality_label, 1.0)
        if "rural" in accent_region.lower():
            mult *= 1.2
            
        for i in range(0, len(parts), 2):
            chunk = parts[i].strip()
            if not chunk: continue
            
            pause_ms = 0
            if i + 1 < len(parts):
                pause_ms = int(float(parts[i+1]) * mult)
                if is_emergency:
                    pause_ms = min(pause_ms, EMERGENCY_MAX_PAUSE_MS)
                
            units.append(ProsodyUnit(text=chunk, pause_ms=pause_ms, is_emphasis=False))
            total_pause += pause_ms
            
        est_dur = (len(text) * 0.1) + (total_pause / 1000)
        return ProsodyPlan(units=units, total_pause_ms=total_pause, estimated_duration_s=est_dur)

_global_controller = None

def get_controller() -> ProsodyController:
    global _global_controller
    if not _global_controller:
        _global_controller = ProsodyController()
    return _global_controller

def apply_prosody(text: str, formality_label: str = "STANDARD",
                  accent_region: str = "neutral", is_emergency: bool = False, **kwargs) -> ProsodyPlan:
    # Compatibility aliases expected by older call-sites
    formality_label = kwargs.get("formality", formality_label)
    accent_region = kwargs.get("region", accent_region)
    is_emergency = kwargs.get("emergency", is_emergency)
    c = get_controller()
    return c.build_plan(text, formality_label, accent_region, is_emergency)