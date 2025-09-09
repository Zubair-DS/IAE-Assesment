from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass
from .memory.memory_service import MemoryService
from .llm.azure_openai import AzureLLM


def trace_print(prefix: str, payload: Dict[str, Any]) -> None:
    print(f"[{prefix}] {payload}")


@dataclass
class AgentResult:
    content: str
    confidence: float
    meta: Dict[str, Any]


class ResearchAgent:
    def __init__(self, memory: MemoryService):
        self.memory = memory
        # Seed knowledge base will be provided externally

    def research(self, query: str) -> AgentResult:
        # first check memory to avoid redundant work
        hits = self.memory.search_knowledge(query, top_k=5, mode="hybrid")
        if hits:
            content = "\n".join(f"- {h['topic']}: {h['content']} (conf={h['confidence']:.2f})" for h in hits)
            return AgentResult(content=content, confidence=min(0.9, max(h["confidence"] for h in hits)), meta={"used_memory": True, "hits": hits})

        # simulate lookup by returning a templated result
        simulated = f"Simulated findings related to: {query}."
        kn_id = self.memory.add_knowledge(topic=query, content=simulated, source="seeded_lookup", agent="research", confidence=0.6, tags=["research", "auto"])
        return AgentResult(content=simulated, confidence=0.6, meta={"used_memory": False, "stored_as": kn_id})


class AnalysisAgent:
    def __init__(self, memory: MemoryService):
        self.memory = memory

    def analyze(self, data_points: List[str], goal: str | None = None) -> AgentResult:
        # simplistic scoring: length and diversity
        coverage = len(set(" ".join(data_points).lower().split()))
        reasoning = f"Analyzed {len(data_points)} items with approx {coverage} unique tokens."
        if goal:
            reasoning += f" Goal: {goal}"
        summary = "; ".join(dp.strip() for dp in data_points if dp.strip())
        content = f"Summary: {summary}\nReasoning: {reasoning}"
        conf = min(0.95, 0.5 + min(0.4, coverage / 400))
        return AgentResult(content=content, confidence=conf, meta={"coverage": coverage})


class MemoryAgent:
    def __init__(self, memory: MemoryService):
        self.memory = memory

    def remember(self, topic: str, content: str, source_agent: str, confidence: float, tags: List[str] | None = None) -> str:
        return self.memory.add_knowledge(topic=topic, content=content, source="memory_agent", agent=source_agent, confidence=confidence, tags=tags or [])

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.memory.search_knowledge(query, top_k=top_k, mode="hybrid")


class Coordinator:
    def __init__(self, memory: MemoryService):
        self.memory = memory
        self.research_agent = ResearchAgent(memory)
        self.analysis_agent = AnalysisAgent(memory)
        self.memory_agent = MemoryAgent(memory)
        self.azure_llm = AzureLLM()

    def classify(self, question: str) -> Dict[str, Any]:
        ql = question.lower()
        steps: List[str] = []
        # Try Azure OpenAI if configured
        if self.azure_llm.enabled:
            try:
                steps = self.azure_llm.classify_plan(question)
                if steps:
                    return {"plan": steps}
            except Exception:
                # graceful fallback to rules when errors occur
                steps = []
        if any(k in ql for k in ["research", "find", "look up", "papers", "information", "what are", "list"]):
            steps.append("research")
        if any(k in ql for k in ["analyze", "compare", "efficiency", "trade-off", "recommend", "which is better", "summarize"]):
            steps.append("analysis")
        if any(k in ql for k in ["what did we", "earlier", "previously", "remember", "recall"]):
            steps = ["memory"] + steps
        if not steps:
            # default: research then analysis
            steps = ["research", "analysis"]
        return {"plan": steps}

    def handle(self, question: str) -> AgentResult:
        self.memory.add_message("user", question)
        plan = self.classify(question)["plan"]
        trace_print("manager.plan", {"question": question, "plan": plan})

        results: List[str] = []
        confs: List[float] = []

        try:
            if plan and plan[0] == "memory":
                # recall first
                recall = self.memory_agent.recall(question)
                if recall:
                    recall_text = "\n".join(f"- {r['topic']}: {r['content']}" for r in recall)
                    results.append(f"Recall:\n{recall_text}")
                    confs.append(0.8)
                plan = [p for p in plan if p != "memory"] or ["research"]

            research_out = None
            if "research" in plan:
                r = self.research_agent.research(question)
                trace_print("research.out", {"content": r.content[:200], "confidence": r.confidence})
                self.memory.add_agent_state("research", question, {"confidence": r.confidence})
                results.append(r.content)
                confs.append(r.confidence)
                research_out = r

            if "analysis" in plan:
                inputs = results.copy()
                a = self.analysis_agent.analyze(inputs, goal=question)
                trace_print("analysis.out", {"content": a.content[:200], "confidence": a.confidence})
                self.memory.add_agent_state("analysis", question, {"confidence": a.confidence})
                results.append(a.content)
                confs.append(a.confidence)

            final = "\n\n".join(results)
            final_conf = sum(confs) / len(confs) if confs else 0.5
            # persist summary to memory
            self.memory_agent.remember(topic=question, content=final, source_agent="manager", confidence=final_conf, tags=["summary"])
            self.memory.add_message("manager", final, {"confidence": final_conf})
            return AgentResult(content=final, confidence=final_conf, meta={"plan": plan})

        except Exception as e:
            fallback = f"Encountered an error; providing best-effort summary. Error: {e}"
            self.memory.add_message("manager", fallback, {"error": True})
            return AgentResult(content=fallback, confidence=0.3, meta={"error": str(e)})
