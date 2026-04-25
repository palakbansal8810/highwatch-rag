import io
import os
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TOKEN_FILE = "./storage/gdrive_token.json"

SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.google-apps.document": ".docx",
    "text/plain": ".txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    size: Optional[int] = None
    content: Optional[bytes] = None
    extension: str = ""

    def __post_init__(self):
        self.extension = SUPPORTED_MIME_TYPES.get(self.mime_type, "")


class GoogleDriveConnector:
    def __init__(self):
        self.service = None
        self._creds = None

    def get_oauth_url(self) -> str:
        flow = self._build_flow()
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return auth_url

    def handle_oauth_callback(self, code: str) -> dict:
        flow = self._build_flow()
        flow.fetch_token(code=code)
        creds = flow.credentials
        self._save_token(creds)
        self._creds = creds
        self.service = build("drive", "v3", credentials=creds)
        return {"status": "authenticated", "email": self._get_user_email()}

    def authenticate(self) -> bool:
        try:
            if settings.google_auth_mode == "service_account":
                self._creds = service_account.Credentials.from_service_account_file(
                    settings.google_service_account_file, scopes=SCOPES
                )
            else:
                self._creds = self._load_or_refresh_token()

            if not self._creds:
                return False

            self.service = build("drive", "v3", credentials=self._creds)
            logger.info("Google Drive authenticated successfully")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def is_authenticated(self) -> bool:
        return self.service is not None and self._creds is not None

    def list_files(
        self,
        folder_id: Optional[str] = None,
        modified_after: Optional[str] = None,
    ) -> list[DriveFile]:
        if not self.is_authenticated():
            raise RuntimeError("Not authenticated with Google Drive")

        mime_filter = " or ".join(
            f"mimeType='{m}'" for m in SUPPORTED_MIME_TYPES
        )
        query = f"({mime_filter}) and trashed=false"

        if folder_id:
            query += f" and '{folder_id}' in parents"

        if modified_after:
            query += f" and modifiedTime > '{modified_after}'"

        files: list[DriveFile] = []
        page_token = None

        while True:
            resp = (
                self.service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                    pageToken=page_token,
                    pageSize=100,
                )
                .execute()
            )

            for f in resp.get("files", []):
                files.append(
                    DriveFile(
                        file_id=f["id"],
                        name=f["name"],
                        mime_type=f["mimeType"],
                        modified_time=f.get("modifiedTime", ""),
                        size=int(f.get("size", 0)) if f.get("size") else None,
                    )
                )

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Found {len(files)} files in Google Drive")
        return files

    def download_file(self, drive_file: DriveFile) -> Optional[bytes]:
        try:
            if drive_file.mime_type in EXPORT_MIME_TYPES:
                # Google Docs → export as DOCX
                export_mime = EXPORT_MIME_TYPES[drive_file.mime_type]
                request = self.service.files().export_media(
                    fileId=drive_file.file_id, mimeType=export_mime
                )
                drive_file.mime_type = export_mime
                drive_file.extension = ".docx"
            else:
                request = self.service.files().get_media(fileId=drive_file.file_id)

            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            content = buf.getvalue()
            logger.debug(f"Downloaded {drive_file.name} ({len(content)} bytes)")
            return content

        except Exception as e:
            msg = str(e)
            if "403" in msg or "cannotDownloadFile" in msg or "forbidden" in msg.lower():
                logger.warning(f"Skipping restricted file (no download permission): {drive_file.name}")
            elif "404" in msg:
                logger.warning(f"Skipping file not found: {drive_file.name}")
            else:
                logger.error(f"Failed to download {drive_file.name}: {e}")
            return None

    def fetch_all_files(
        self,
        folder_id: Optional[str] = None,
        modified_after: Optional[str] = None,
    ) -> list[DriveFile]:
        """List and download all supported files."""
        files = self.list_files(folder_id=folder_id, modified_after=modified_after)
        for f in files:
            f.content = self.download_file(f)
        return [f for f in files if f.content is not None]

    def _build_flow(self) -> Flow:
        if not settings.google_client_id or settings.google_client_id == "your_google_client_id":
            raise ValueError(
                "GOOGLE_CLIENT_ID is not set. "
                "Please add it to your .env file:\n"
                "  GOOGLE_CLIENT_ID=<your-client-id>.apps.googleusercontent.com"
            )
        if not settings.google_client_secret or settings.google_client_secret == "your_google_client_secret":
            raise ValueError(
                "GOOGLE_CLIENT_SECRET is not set. "
                "Please add it to your .env file:\n"
                "  GOOGLE_CLIENT_SECRET=GOCSPX-..."
            )

        client_config = {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uris": [settings.google_redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }
        logger.debug(f"Building OAuth flow with client_id: {settings.google_client_id[:20]}...")
        return Flow.from_client_config(
            client_config, scopes=SCOPES, redirect_uri=settings.google_redirect_uri
        )

    def _load_or_refresh_token(self) -> Optional[Credentials]:
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._save_token(creds)
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}")
                return None

        return creds if (creds and creds.valid) else None

    def _save_token(self, creds: Credentials):
        Path(TOKEN_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    def _get_user_email(self) -> str:
        try:
            about = self.service.about().get(fields="user").execute()
            return about["user"]["emailAddress"]
        except Exception:
            return "unknown"