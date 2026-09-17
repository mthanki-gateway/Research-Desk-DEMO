"""The whole magic-link path, from brief to result.

    owner   writes a brief, gets a project with a frozen schema
    owner   creates a link
    GUEST   opens it with NO ACCOUNT, is interviewed
    GUEST   comes back on the same link -- same conversation, still there
    owner   sees the result
    owner   revokes it, and it stops working
    other   a different owner cannot see any of it

Everything goes through the real HTTP and WebSocket surface. The owner half
uses a dependency override for `current_user`; the GUEST half uses no token at
all, which is the only way to prove the anonymous path works rather than
quietly falling back to an authenticated one.

Run:  docker compose exec api python -m app.probe_invite
"""

from __future__ import annotations

import json
import time

from starlette.testclient import TestClient

from app.api import live as live_api
from app.auth import User, current_user
from app.main import app

BRIEF = """Qualify inbound leads. Find out their budget, when they want to go
live, and what tooling they use today."""

# owner_id is derived from id, so a non-anonymous User is scoped by its id.
OWNER = User(id="owner-1", email="owner@example.com", anonymous=False)
OTHER = User(id="owner-2", email="other@example.com", anonymous=False)

_as = {"user": OWNER}
app.dependency_overrides[current_user] = lambda: _as["user"]
# The guest socket path must NOT go through this. It is overridden only so the
# owner's own WebSocket calls work; guests pass `invite` and never reach it.
live_api._authenticate = lambda token: _none()


async def _none():
    return None


def greet(ws) -> str:
    ws.send_json({"type": "greet"})
    said: list[str] = []
    deadline = time.time() + 60
    while time.time() < deadline:
        message = ws.receive()
        if message.get("type") == "websocket.disconnect":
            break
        if message.get("bytes") is not None:
            continue
        if not message.get("text"):
            continue
        event = json.loads(message["text"])
        if event["type"] == "said":
            said.append(event["text"])
        elif event["type"] == "turn_end":
            break
        elif event["type"] == "error":
            return f"[ERROR {event['detail']}]"
    return "".join(said).strip()


def main() -> None:
    with TestClient(app) as http:
        print("=" * 70)
        print("OWNER: a brief becomes a project")
        print("=" * 70)
        project = http.post(
            "/howler/projects", json={"brief": BRIEF, "title": "Lead qualification"}
        ).json()
        print(f"  {len(project['fields'])} fields:", ", ".join(
            f["name"] for f in project["fields"]
        ))

        invite = http.post(
            f"/howler/projects/{project['id']}/invites",
            json={"label": "Priya Raman", "participant": "Head of Data, logistics."},
        ).json()
        token = invite["token"]
        print(f"  link: /howl/{token[:16]}…")

        print()
        print("=" * 70)
        print("GUEST: no account, no token — just the link")
        print("=" * 70)
        with http.websocket_connect(f"/live/ws?invite={token}") as ws:
            ready = ws.receive_json()
            first_session = ready.get("session_id", "")
            print(f"  connected: mode={ready.get('mode')} session={first_session[:8]}…")
            print(f"  HOWLER: {greet(ws)[:150]}")

        print()
        print("=" * 70)
        print("GUEST: comes back on the same link")
        print("=" * 70)
        # Checked through the service rather than a second socket: opening two
        # WebSockets inside one TestClient context cancels the first through
        # anyio's portal, which is a test-harness artefact and not a server
        # behaviour. What matters is that the link still points at the SAME
        # conversation, which is what a returning guest resolves to.
        rows = http.get(f"/howler/projects/{project['id']}/invites").json()
        attached = rows[0]["result"]["session_id"] if rows[0]["result"] else ""
        print(f"  link points at : {attached[:8]}…")
        print(f"  same as before : {attached == first_session}")

        print()
        print("=" * 70)
        print("OWNER: what came back")
        print("=" * 70)
        listed = http.get(f"/howler/projects/{project['id']}/invites").json()
        for row in listed:
            print(f"  {row['label']:16} status={row['status']:12} opens={row['opens']}")
            if row["result"]:
                print(f"    -> turns={row['result']['turns']} "
                      f"missing={row['result']['missing']}")

        print()
        print("=" * 70)
        print("ANOTHER OWNER cannot see it")
        print("=" * 70)
        _as["user"] = OTHER
        print(f"  their project list : {http.get('/howler/projects').json()}")
        pid = project["id"]
        print(f"  fetching by id     : {http.get(f'/howler/projects/{pid}').status_code}")
        _as["user"] = OWNER

        print()
        print("=" * 70)
        print("OWNER: revokes the link")
        print("=" * 70)
        code = http.delete(f"/howler/invites/{invite['id']}").status_code
        print(f"  revoke -> {code}")
        with http.websocket_connect(f"/live/ws?invite={token}") as ws:
            print(f"  guest now sees: {ws.receive_json().get('detail')}")

        print()
        print("=" * 70)
        print("A MADE-UP TOKEN")
        print("=" * 70)
        with http.websocket_connect("/live/ws?invite=not-a-real-token") as ws:
            print(f"  {ws.receive_json().get('detail')}")

        # The results survive revocation. Withdrawing a link and discarding
        # what was already learned are not the same decision.
        after = http.get(f"/howler/projects/{project['id']}/invites").json()
        print()
        print(f"  result kept after revoke: {after[0]['result'] is not None}")


if __name__ == "__main__":
    main()
