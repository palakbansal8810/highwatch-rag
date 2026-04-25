#!/usr/bin/env python3
"""
tests/sample_queries.py
Run sample queries against a running Highwatch RAG instance.
Usage: python tests/sample_queries.py
"""

import json
import time
import sys
import httpx

BASE_URL = "http://localhost:8000"

SAMPLE_QUERIES = [
    "What is Btech Syllabus?",
    "Tell me about palak's technical skills?",
    "What's the main architecture of TTS report"
]

client = httpx.Client(base_url=BASE_URL, timeout=120)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check_status():
    print_section("System Status")
    r = client.get("/status")
    status = r.json()
    print(json.dumps(status, indent=2))
    return status


def check_sync_status():
    r = client.get("/sync-status")
    return r.json()


def list_documents():
    print_section("Indexed Documents")
    r = client.get("/documents")
    data = r.json()
    print(f"Total documents: {data['total']}")
    for doc in data["documents"]:
        print(f"  • {doc['file_name']}  ({doc['chunk_count']} chunks)")
    return data


def ask_question(query: str, top_k: int = 5):
    print(f"\n{'─'*60}")
    print(f"Q: {query}")
    print('─'*60)

    r = client.post("/ask", json={"query": query, "top_k": top_k})

    if r.status_code != 200:
        print(f"ERROR {r.status_code}: {r.text}")
        return

    data = r.json()
    print(f"\nA: {data['answer']}")
    print(f"\n Sources ({len(data['sources'])}):")
    for src in data["sources"]:
        print(f"   • {src}")
    print(f"\n Chunks used: {data['chunks_used']}  |  Model: {data['model']}")


def poll_until_done():
    """Poll /sync-status until sync completes."""
    print("\nPolling for sync completion every 10s...")
    print("   (Watch server terminal for live file-by-file progress)\n")
    while True:
        time.sleep(10)
        state = check_sync_status()
        status = state.get("status")

        if status == "running":
            docs = state.get("documents_so_far", "?")
            chunks = state.get("chunks_so_far", "?")
            print(f" Running... {docs} docs | {chunks} chunks so far")
        elif status == "completed":
            print("\n Sync completed!")
            print(f"  Files processed : {state.get('files_processed', '?')}")
            print(f"  Files skipped   : {state.get('files_skipped', '?')}")
            print(f"  Files failed    : {state.get('files_failed', '?')}")
            print(f"  Chunks added    : {state.get('chunks_added', '?')}")
            print(f"  Duration        : {state.get('duration_seconds', '?'):.1f}s")
            if state.get("errors"):
                print(f"  Errors          : {len(state['errors'])} (restricted/unreadable files)")
            break
        elif status == "error":
            print(f"\n Sync failed: {state.get('error')}")
            sys.exit(1)
        elif status == "idle":
            # Sync already finished before we started polling (fast drives)
            print("  ✓ Sync already completed.")
            break
        else:
            print(f"  Status: {status}")


def sync_drive(incremental: bool = False):
    print_section("Starting Google Drive Sync")

    # Check if already running from a previous trigger
    current = check_sync_status()
    if current.get("status") == "running":
        print("Sync already in progress from a previous trigger, polling...")
        poll_until_done()
        return

    # Start sync
    r = client.post("/sync-drive", json={"incremental": incremental})
    resp = r.json()
    print(json.dumps(resp, indent=2))

    poll_until_done()


if __name__ == "__main__":
    status = check_status()

    if not status.get("authenticated"):
        print("\n  Not authenticated. Visit http://localhost:8000/auth/login")
        sys.exit(1)

    # Check if a sync is currently running (e.g. triggered from server terminal)
    sync_st = check_sync_status()

    if sync_st.get("status") == "running":
        print("\n Sync is already running in the background. Waiting for it...")
        poll_until_done()
    elif status.get("total_chunks", 0) == 0:
        print("\n No documents indexed yet. Starting sync...")
        sync_drive()
    else:
        print(f"\n Already have {status['total_chunks']} chunks from {status['total_documents']} docs. Skipping sync.")
        print("  Tip: POST /sync-drive with force_reindex=true to re-index everything.")

    # List documents
    list_documents()

    # Run sample queries
    print_section("Sample Queries")
    for query in SAMPLE_QUERIES:
        ask_question(query)

    print(f"\n\n Done! {len(SAMPLE_QUERIES)} queries executed.")