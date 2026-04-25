"""
main.py
Highwatch AI — RAG over Google Drive
FastAPI application entrypoint.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from api.routes import router

# ------------------------------------------------------------------ #
#  Logging setup                                                      #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("highwatch")


# ------------------------------------------------------------------ #
#  App lifecycle                                                      #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Highwatch AI RAG system starting up")
    logger.info(f"   Embedding model : {settings.embedding_model}")
    logger.info(f"   LLM             : {settings.groq_model}")
    logger.info(f"   FAISS index     : {settings.faiss_index_path}")
    yield
    logger.info("Shutting down Highwatch AI")


# ------------------------------------------------------------------ #
#  App                                                                #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="Highwatch AI — RAG over Google Drive",
    description=(
        "A production-grade Retrieval-Augmented Generation system that connects to "
        "Google Drive, indexes your documents, and lets you ask natural language questions "
        "answered by Groq LLM — grounded entirely in your documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Highwatch AI RAG",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "auth":      "GET  /auth/login  →  POST /auth/callback",
            "sync":      "POST /sync-drive",
            "query":     "POST /ask",
            "status":    "GET  /status",
            "documents": "GET  /documents",
        },
    }


# ------------------------------------------------------------------ #
#  Entrypoint                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=600,
    )