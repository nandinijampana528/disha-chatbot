import asyncio
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.core.rag import build_chain, get_sources, get_vectorstore
from app.core.session import session_manager
from app.core.config import settings
from app.models.schemas import (
    QueryRequest, QueryResponse, ClearSessionRequest,
    HealthResponse, SourceDoc
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Standard query endpoint with chat history."""
    try:
        history = session_manager.get_history(request.session_id)
        chain = build_chain(history)

        response = await asyncio.to_thread(chain.invoke, request.query)

        # Save to history
        session_manager.add_message(request.session_id, "human", request.query)
        session_manager.add_message(request.session_id, "ai", response)

        sources = None
        if request.include_sources:
            raw = get_sources(request.query)
            sources = [SourceDoc(**s) for s in raw]

        return QueryResponse(
            answer=response,
            session_id=request.session_id,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming query endpoint — tokens stream as they're generated."""
    try:
        history = session_manager.get_history(request.session_id)
        chain = build_chain(history)

        full_response = []

        def token_generator():
            for chunk in chain.stream(request.query):
                full_response.append(chunk)
                yield chunk
            # Save complete response to history after streaming
            session_manager.add_message(request.session_id, "human", request.query)
            session_manager.add_message(request.session_id, "ai", "".join(full_response))

        return StreamingResponse(token_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/clear")
async def clear_session(request: ClearSessionRequest):
    """Clear chat history for a session."""
    session_manager.clear_session(request.session_id)
    return {"message": f"Session {request.session_id} cleared."}


@router.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Get chat history for a session."""
    history = session_manager.get_history(session_id)
    return {"session_id": session_id, "history": history}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check model + vector store status."""
    try:
        vectordb = get_vectorstore()
        count = vectordb._collection.count()
        return HealthResponse(
            status="healthy",
            model=settings.CHAT_MODEL,
            collection=settings.CHROMA_COLLECTION,
            document_count=count,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
