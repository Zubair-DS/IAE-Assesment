from __future__ import annotations

from typing import List, Dict, Any
from .schemas import Message, KnowledgeRecord, AgentState, now_ts, gen_id, to_dict
from .vector_store import InMemoryVectorStore


class MemoryService:
    def __init__(self):
        self.conversation: List[Dict[str, Any]] = []
        self.knowledge_meta: Dict[str, Dict[str, Any]] = {}
        self.agent_states: List[Dict[str, Any]] = []
        self.vstore = InMemoryVectorStore()

    # Conversation Memory
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] | None = None) -> str:
        msg = Message(id=gen_id("msg"), role=role, content=content, timestamp=now_ts(), metadata=metadata or {})
        self.conversation.append(to_dict(msg))
        return msg.id

    def get_conversation(self) -> List[Dict[str, Any]]:
        return list(self.conversation)

    # Knowledge Base
    def add_knowledge(self, topic: str, content: str, source: str, agent: str, confidence: float, tags: List[str] | None = None) -> str:
        rec = KnowledgeRecord(
            id=gen_id("kn"),
            topic=topic,
            content=content,
            source=source,
            agent=agent,
            timestamp=now_ts(),
            confidence=confidence,
            tags=tags or [],
        )
        meta = to_dict(rec)
        self.knowledge_meta[rec.id] = meta
        self.vstore.add(rec.id, {"type": "knowledge", "topic": topic, "text_index": f"{topic} {' '.join(rec.tags)} {content}"}, f"{topic}\n{content}\n{' '.join(rec.tags)}")
        return rec.id

    def search_knowledge(self, query: str, top_k: int = 5, mode: str = "hybrid") -> List[Dict[str, Any]]:
        results = []
        if mode in ("hybrid", "vector"):
            results.extend(self.vstore.search(query, top_k=top_k))
        if mode in ("hybrid", "keyword"):
            results.extend(self.vstore.keyword_search(query, top_k=top_k))
        # dedupe by id keep best score
        best: Dict[str, Dict[str, Any]] = {}
        for iid, payload, score in results:
            meta = self.knowledge_meta.get(iid)
            if not meta:
                continue
            cur = best.get(iid)
            if not cur or score > cur["_score"]:
                best[iid] = {**meta, "_score": score}
        # sort by score desc
        ordered = sorted(best.values(), key=lambda x: x["_score"], reverse=True)
        return ordered[:top_k]

    # Agent State Memory
    def add_agent_state(self, agent: str, task: str, details: Dict[str, Any]) -> str:
        st = AgentState(id=gen_id("st"), agent=agent, task=task, details=details, timestamp=now_ts())
        self.agent_states.append(to_dict(st))
        return st.id

    def get_agent_states(self, agent: str | None = None) -> List[Dict[str, Any]]:
        if not agent:
            return list(self.agent_states)
        return [s for s in self.agent_states if s["agent"] == agent]
