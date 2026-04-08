from collections import defaultdict
from typing import List, Dict
from app.core.config import settings


class SessionManager:
    """In-memory chat session store. Replace with Redis for production."""

    def __init__(self):
        self._sessions: Dict[str, List[dict]] = defaultdict(list)

    def get_history(self, session_id: str) -> List[dict]:
        return self._sessions[session_id][-settings.MAX_HISTORY:]

    def add_message(self, session_id: str, role: str, content: str):
        self._sessions[session_id].append({"role": role, "content": content})

    def clear_session(self, session_id: str):
        self._sessions[session_id] = []

    def get_all_sessions(self) -> List[str]:
        return list(self._sessions.keys())


session_manager = SessionManager()
