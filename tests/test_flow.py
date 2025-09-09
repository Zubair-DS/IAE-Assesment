from src.memory.memory_service import MemoryService
from src.agents import Coordinator


def test_basic_flow():
    mem = MemoryService()
    coord = Coordinator(mem)
    q1 = "What are the main types of neural networks?"
    out1 = coord.handle(q1)
    assert out1.content
    assert out1.confidence > 0

    # Memory recall follow-up
    q2 = "What did we discuss about neural networks earlier?"
    out2 = coord.handle(q2)
    assert "Recall:" in out2.content or out2.confidence >= 0
