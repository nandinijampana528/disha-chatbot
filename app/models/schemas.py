from pydantic import BaseModel, Field
from typing import Optional, List


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    session_id: str = Field(..., description="Unique session/user ID")
    include_sources: bool = Field(False, description="Return source documents")


class SourceDoc(BaseModel):
    source: str
    page: Optional[str] = ""
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: Optional[List[SourceDoc]] = None


class ClearSessionRequest(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str
    model: str
    collection: str
    document_count: int
