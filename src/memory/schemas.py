from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid


def now_ts() -> str:
    # Use timezone-aware UTC timestamps
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Message:
    id: str
    role: str  # user|manager|agent
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeRecord:
    id: str
    topic: str
    content: str
    source: str  # e.g., "seed", "research_agent"
    agent: str
    timestamp: str
    confidence: float
    tags: List[str] = field(default_factory=list)


@dataclass
class AgentState:
    id: str
    agent: str
    task: str
    details: Dict[str, Any]
    timestamp: str


def to_dict(obj) -> Dict[str, Any]:
    return asdict(obj)
