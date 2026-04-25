from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from api.pipeline import RAGPipeline

router = APIRouter()
pipeline = RAGPipeline()

class SyncRequest(BaseModel):
    folder_id: Optional[str] = Field(None, description="Specific Google Drive folder ID to sync (defaults to all)")
    incremental: bool = Field(False, description="Only sync files modified since last sync")
    force_reindex: bool = Field(False, description="Re-index files that are already stored")

class SyncResponse(BaseModel):
    status: str
    files_discovered: int
    files_processed: int
    files_skipped: int
    files_failed: int
    chunks_added: int
    duration_seconds: float
    errors: list[str]

class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Your question about the documents")
    top_k: int = Field(5, ge=1, le=20, description="Number of document chunks to retrieve")
    filter_files: Optional[list[str]] = Field(None, description="Limit search to specific file names")

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    model: str
    query: str

@router.get("/auth/login", tags=["Auth"], summary="Get Google OAuth URL")
async def auth_login():
    """Returns the Google OAuth2 URL. Visit it in your browser to authorize."""
    try:
        url = pipeline.drive.get_oauth_url()
        return {
            "auth_url": url,
            "instructions": "Visit this URL in your browser, authorize, then copy the 'code' param from the redirect URL and POST to /auth/callback"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auth/callback", tags=["Auth"], summary="Complete OAuth flow (Google redirects here)")
async def auth_callback(
    code: str = Query(...),
    state: str = Query(None),
    scope: str = Query(None),
    iss: str = Query(None),
):
    from fastapi.responses import HTMLResponse
    try:
        result = pipeline.drive.handle_oauth_callback(code)
        email = result.get("email", "your account")
        html = f"""<html><body style="font-family:sans-serif;max-width:480px;margin:80px auto;text-align:center;">
          <h2 style="color:#1a7f4b;">&#10003; Authenticated!</h2>
          <p>Google Drive connected as <strong>{email}</strong></p>
          <p style="color:#555;">You can close this tab and return to the terminal.</p>
          <p style="margin-top:2rem;">Next step: call <code>POST /sync-drive</code> to index your documents.</p>
        </body></html>"""
        return HTMLResponse(content=html)
    except Exception as e:
        html = f"""<html><body style="font-family:sans-serif;max-width:480px;margin:80px auto;text-align:center;">
          <h2 style="color:#c0392b;">&#10007; Auth Failed</h2>
          <p>{str(e)}</p>
          <p>Visit <a href="/auth/login">/auth/login</a> to try again.</p>
        </body></html>"""
        return HTMLResponse(content=html, status_code=400)

@router.post("/sync-drive", response_model=SyncResponse, tags=["Sync"], summary="Sync Google Drive documents")
async def sync_drive(body: SyncRequest = SyncRequest()):
    try:
        result = await pipeline.sync_drive(
            folder_id=body.folder_id,
            incremental=body.incremental,
            force_reindex=body.force_reindex,
        )
        return SyncResponse(
            status="success",
            files_discovered=result.files_discovered,
            files_processed=result.files_processed,
            files_skipped=result.files_skipped,
            files_failed=result.files_failed,
            chunks_added=result.chunks_added,
            duration_seconds=result.duration_seconds,
            errors=result.errors,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {e}")


@router.post("/ask", response_model=AskResponse, tags=["Query"], summary="Ask a question over your documents")
async def ask(body: AskRequest): 
    try:
        response = await pipeline.ask(
            query=body.query,
            top_k=body.top_k,
            filter_file_names=body.filter_files,
        )
        return AskResponse(
            answer=response.answer,
            sources=response.sources,
            chunks_used=response.chunks_used,
            model=response.model,
            query=response.query,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
@router.get("/status", tags=["Utility"], summary="System status")
async def status():
    return pipeline.get_status()


@router.get("/documents", tags=["Utility"], summary="List indexed documents")
async def list_documents():
    docs = pipeline.store.list_documents()
    return {"total": len(docs), "documents": docs}


@router.delete("/documents/{doc_id}", tags=["Utility"], summary="Remove a document from the index")
async def delete_document(doc_id: str):
    if not pipeline.store.document_exists(doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in index")
    pipeline.store.remove_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@router.get("/debug/config", tags=["Utility"], summary="Check loaded config values")
async def debug_config():
    from config import settings

    def mask(val: str, placeholder: str) -> str:
        if not val or val == placeholder:
            return f"⚠️ NOT SET (value: '{val}')"
        return val[:20] + "..." if len(val) > 20 else "SET ✓"

    return {
        "google_client_id":     mask(settings.google_client_id, "your_google_client_id"),
        "google_client_secret": mask(settings.google_client_secret, "your_google_client_secret"),
        "google_redirect_uri":  settings.google_redirect_uri,
        "google_auth_mode":     settings.google_auth_mode,
        "groq_api_key":         mask(settings.groq_api_key, "your_groq_api_key_here"),
        "groq_model":           settings.groq_model,
        "embedding_model":      settings.embedding_model,
        "tip":                  "Run `python main.py` from the folder that contains your .env file",
    }