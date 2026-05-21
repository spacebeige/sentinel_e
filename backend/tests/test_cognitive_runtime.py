import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.orchestration_run import CognitivePhase, RunLifecycle, create_orchestration_run
from core.runtime_event_bus import RunEventBus


def test_route_phase_sets_routing_lifecycle():
    run = create_orchestration_run(chat_id="chat-1", user_id="user-1", query_preview="route me")

    run.transition_to(CognitivePhase.ROUTE, {"path": "ensemble"})

    assert run.lifecycle_state == RunLifecycle.ROUTING
    assert run.cognitive_phase == CognitivePhase.ROUTE


def test_event_bus_close_emits_terminal_event():
    async def _collect_terminal_event():
        bus = RunEventBus("run-1")
        bus.close()
        event = await asyncio.wait_for(bus._queue.get(), timeout=1.0)
        assert event["event_type"] == "orchestration_completed"
        assert event["run_id"] == "run-1"

    asyncio.run(_collect_terminal_event())
