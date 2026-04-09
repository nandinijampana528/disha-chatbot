import logging
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are {bot_name}, the AI assistant for {company_name}.

STRICT RULES:
- Answer ONLY the specific question asked. Do not dump all document content.
- Use ONLY the most relevant part of the context.
- If the answer is not clearly in the context, say: "I don't have that information."
- Be concise. 2-3 sentences maximum unless a detailed answer is truly needed.
- Never list unrelated information from other documents.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer concisely and only what was asked:"""


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    logger.info("Initializing ChromaDB vector store...")
    embedding = OllamaEmbeddings(
        model=settings.EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )
    vectordb = Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embedding,
        collection_name=settings.CHROMA_COLLECTION,
    )
    logger.info(f"Vector store loaded: {vectordb._collection.count()} documents")
    return vectordb


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    logger.info(f"Loading LLM: {settings.CHAT_MODEL}")
    return ChatOllama(
        model=settings.CHAT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.2,
    )


def format_docs(docs) -> str:
    if not docs:
        return "No relevant context found."
    return "\n\n---\n\n".join([
        f"Source: {d.metadata.get('source', 'Unknown')}\n{d.page_content}"
        for d in docs
    ])


def format_chat_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history[-settings.MAX_HISTORY:]:
        role = "User" if msg["role"] == "human" else settings.BOT_NAME
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_chain(history: list = None):
    vectordb = get_vectorstore()
    llm = get_llm()

    retriever = vectordb.as_retriever(
        search_type=settings.RETRIEVER_TYPE,
        search_kwargs={"k": settings.RETRIEVER_K},
    )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: format_chat_history(history or [])),
            "bot_name": RunnableLambda(lambda _: settings.BOT_NAME),
            "company_name": RunnableLambda(lambda _: settings.COMPANY_NAME),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_sources(query: str) -> list[dict]:
    """Return source documents used for a query."""
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(
        search_type=settings.RETRIEVER_TYPE,
        search_kwargs={"k": settings.RETRIEVER_K},
    )
    docs = retriever.invoke(query)
    return [
        {
            "source": d.metadata.get("source", "Unknown"),
            "page": d.metadata.get("page", ""),
            "snippet": d.page_content[:200] + "...",
        }
        for d in docs
    ]
