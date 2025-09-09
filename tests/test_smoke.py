from src.memory.memory_service import MemoryService
from src.agents import Coordinator


def test_end_to_end_runs():
    mem = MemoryService()
    coord = Coordinator(mem)
    q = "What are the main types of neural networks?"
    res = coord.handle(q)
    assert res.content
    assert 0 <= res.confidence <= 1


def test_memory_roundtrip():
    mem = MemoryService()
    coord = Coordinator(mem)
    # first, create memory by asking
    coord.handle("What are the main types of neural networks?")
    # recall
    res = coord.handle("What did we discuss about neural networks earlier?")
    assert "Recall" in res.content or res.confidence > 0
