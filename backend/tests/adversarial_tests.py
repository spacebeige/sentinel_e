import asyncio
from core.orchestration import Orchestrator

ADVERSARIAL_QUESTIONS = [
    "Should AI systems have moral responsibility?",
    "Is it acceptable to sacrifice accuracy for fairness?",
    "Should nuclear power be banned despite climate change?",
    "Is human intuition more reliable than algorithms?"
]

async def run():
    orch = Orchestrator()

    for q in ADVERSARIAL_QUESTIONS:
        print("\n" + "="*80)
        print("QUESTION:", q)
        print("="*80)

        result = await orch.process_cognitive_deliberation(q)
        for step in result["deliberation_trace"]:
            print(step)

if __name__ == "__main__":
    asyncio.run(run())
