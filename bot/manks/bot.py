"""Manks bot: joins a Google Meet, records what it hears, uploads it in pieces.

Runs in its own container (see Dockerfile) and talks to the Research Desk API
with a shared secret. It holds no model keys: it never transcribes anything.

HOW IT LISTENS

A meeting's audio arrives as WebRTC tracks. An init script, injected before
the page's own code runs, wraps RTCPeerConnection so every remote audio track
is also routed into one WebAudio mix, and a MediaRecorder records that mix. No
virtual sound card, no screen capture: the page is asked for the audio it is
already playing. The recorder is restarted every SEGMENT seconds, so each
upload is a complete, independently playable webm file.

WHAT IS NOT VERIFIED

The selectors for Meet's join screen are written from how Meet looks today and
Google changes them without notice. When a step cannot find its button the bot
reports what it could see and saves a screenshot to /tmp, rather than waiting
for ever. Anonymous guests are also something a meeting's host can switch off,
in which case the call shows "sign in" and the bot reports exactly that.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import time

import httpx

try:  # the lightweight image has no browser, and does not need one
    from playwright.async_api import Page, async_playwright
except ImportError:  # pragma: no cover - exercised in the native image
    Page = object  # type: ignore[assignment,misc]
    async_playwright = None  # type: ignore[assignment]

API = os.environ.get("MANKS_API_URL", "http://api:8000").rstrip("/")
SECRET = os.environ.get("MANKS_BOT_SECRET", "")
SEGMENT = int(os.environ.get("MANKS_SEGMENT_SECONDS", "120"))
MAX_MINUTES = int(os.environ.get("MANKS_MAX_MINUTES", "180"))
ADMIT_WAIT_MINUTES = int(os.environ.get("MANKS_ADMIT_WAIT_MINUTES", "10"))
SILENCE_MINUTES = int(os.environ.get("MANKS_SILENCE_MINUTES", "8"))
# What this container can join. The browser image does Meet and BBB-in-a-browser;
# the native image does BBB without one, and many at once.
MODE = os.environ.get("MANKS_MODE", "browser")
MAX_BROWSER = int(os.environ.get("MANKS_MAX_BROWSER", "1"))
MAX_NATIVE = int(os.environ.get("MANKS_MAX_NATIVE", "25"))
# A bot account's saved login (Playwright storage state), for meetings that
# refuse anonymous guests. Optional.
STATE = os.environ.get("MANKS_GOOGLE_STATE", "")

HEADERS = {"X-Manks-Secret": SECRET}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------
# The recorder, injected into every page before its own scripts run.
# --------------------------------------------------------------------------

INIT_JS = r"""
(() => {
  if (window.__manks) return;
  const M = (window.__manks = {
    ctx: null, mix: null, dest: null, analyser: null,
    index: 0, t0: 0, running: false, done: false, tracks: 0,
    lastSound: Date.now(), level: 0, segSeconds: 120, rec: null, timer: null,
  });

  function ensure() {
    if (M.ctx) return M.ctx;
    M.ctx = new AudioContext();
    M.mix = M.ctx.createGain();
    M.dest = M.ctx.createMediaStreamDestination();
    M.analyser = M.ctx.createAnalyser();
    M.analyser.fftSize = 1024;
    M.mix.connect(M.dest);
    M.mix.connect(M.analyser);
    M.mic = M.ctx.createMediaStreamDestination();
    const buf = new Float32Array(M.analyser.fftSize);
    setInterval(() => {
      M.analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      M.level = Math.sqrt(sum / buf.length);
      if (M.level > 0.004) M.lastSound = Date.now();
    }, 500);
    return M.ctx;
  }

  function hook(pc) {
    pc.addEventListener('track', (e) => {
      if (e.track.kind !== 'audio') return;
      const ctx = ensure();
      const src = ctx.createMediaStreamSource(new MediaStream([e.track]));
      src.connect(M.mix);
      M.tracks++;
    });
  }

  function wrap(name) {
    const Orig = window[name];
    if (!Orig) return;
    const Wrapped = function (...args) { const pc = new Orig(...args); hook(pc); return pc; };
    Wrapped.prototype = Orig.prototype;
    Object.setPrototypeOf(Wrapped, Orig);
    window[name] = Wrapped;
  }
  wrap('RTCPeerConnection');
  wrap('webkitRTCPeerConnection');

  async function toBase64(blob) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin);
  }


  // ---- talking: a virtual microphone, and the meeting's audio as PCM ----------
  // Only when the bot was sent to talk. getUserMedia is answered with a stream
  // WE feed: silence until the API sends speech, then the speech.
  // (mediaDevices does not exist on about:blank or plain http, where this also runs)
  if (window.__manksTalk && navigator.mediaDevices) {
    const realGum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => {
      if (c && c.audio) {
        ensure();
        if (M.ctx.state === 'suspended') M.ctx.resume();
        const out = new MediaStream(M.mic.stream.getAudioTracks());
        if (c.video) { try { (await realGum({ video: c.video })).getVideoTracks().forEach(t => out.addTrack(t)); } catch (e) {} }
        return out;
      }
      return realGum(c);
    };

    M.next = 0; M.playing = new Set();
    M.speak = (b64) => {
      ensure();
      const bin = atob(b64), n = bin.length >> 1, f = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
        if (v & 0x8000) v -= 0x10000;
        f[i] = v / 32768;
      }
      const buf = M.ctx.createBuffer(1, n, 24000);
      buf.copyToChannel(f, 0);
      const src = M.ctx.createBufferSource();
      src.buffer = buf; src.connect(M.mic);
      const at = Math.max(M.ctx.currentTime + 0.03, M.next);
      src.start(at); M.next = at + buf.duration;
      M.playing.add(src); src.onended = () => M.playing.delete(src);
    };
    M.hush = () => {
      M.playing.forEach((s) => { try { s.stop(); } catch (e) {} });
      M.playing.clear(); M.next = 0;
    };
    M.tap = () => {
      ensure();
      const node = M.ctx.createScriptProcessor(4096, 1, 1);
      const ratio = M.ctx.sampleRate / 16000;
      const mute = M.ctx.createGain(); mute.gain.value = 0;
      node.onaudioprocess = (e) => {
        const x = e.inputBuffer.getChannelData(0), n = Math.floor(x.length / ratio);
        const out = new Int16Array(n);
        for (let i = 0; i < n; i++) {
          let a = 0, from = Math.floor(i * ratio), to = Math.floor((i + 1) * ratio);
          for (let j = from; j < to; j++) a += x[j];
          const v = Math.max(-1, Math.min(1, a / (to - from)));
          out[i] = v < 0 ? v * 32768 : v * 32767;
        }
        const bytes = new Uint8Array(out.buffer); let bin = '';
        for (let i = 0; i < bytes.length; i += 0x2000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x2000));
        window.__manksPcm(btoa(bin));
      };
      M.mix.connect(node); node.connect(mute); mute.connect(M.ctx.destination);
    };
  }

  M.start = (segSeconds) => {
    const ctx = ensure();
    if (ctx.state === 'suspended') ctx.resume();
    M.segSeconds = segSeconds;
    M.running = true;
    M.done = false;
    M.t0 = performance.now();
    const cycle = () => {
      if (!M.running) { M.done = true; return; }
      const rec = new MediaRecorder(M.dest.stream, {
        mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 48000,
      });
      const chunks = [];
      const index = M.index++;
      const start = (performance.now() - M.t0) / 1000;
      rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      rec.onstop = async () => {
        const seconds = (performance.now() - M.t0) / 1000 - start;
        const blob = new Blob(chunks, { type: 'audio/webm' });
        if (blob.size > 800) {
          try { await window.__manksSegment(index, start, seconds, await toBase64(blob)); }
          catch (err) { console.log('segment upload failed', String(err)); }
        }
        if (M.running) cycle(); else M.done = true;
      };
      rec.start(1000);
      M.rec = rec;
      M.timer = setTimeout(() => { if (rec.state !== 'inactive') rec.stop(); }, segSeconds * 1000);
    };
    cycle();
  };

  M.stop = () => {
    M.running = false;
    clearTimeout(M.timer);
    if (M.rec && M.rec.state !== 'inactive') M.rec.stop(); else M.done = true;
  };
})();
"""


# --------------------------------------------------------------------------
# Talking to the API
# --------------------------------------------------------------------------


class Api:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(base_url=API, headers=HEADERS, timeout=60)

    async def claim(self, platforms: list[str]) -> dict | None:
        r = await self.http.post("/manks/bot/claim", json={"platforms": platforms})
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    async def status(self, mid: str, status: str, detail: str = "") -> None:
        try:
            await self.http.post(f"/manks/bot/{mid}/status", json={"status": status, "detail": detail})
        except Exception as exc:  # noqa: BLE001 - reporting must not end the call
            log("status failed:", exc)

    async def leave_requested(self, mid: str) -> bool:
        try:
            r = await self.http.get(f"/manks/bot/{mid}/control")
            return bool(r.json().get("leave"))
        except Exception:  # noqa: BLE001
            return False

    async def segment(self, mid: str, index: int, start: float, seconds: float, data: bytes) -> None:
        for attempt in range(3):
            try:
                r = await self.http.post(
                    f"/manks/bot/{mid}/segment",
                    params={"index": index, "start": round(start, 2), "seconds": round(seconds, 2)},
                    content=data,
                    headers={"Content-Type": "audio/webm"},
                )
                r.raise_for_status()
                log(f"segment {index} uploaded ({len(data) // 1024} KB, {seconds:.0f}s)")
                return
            except Exception as exc:  # noqa: BLE001
                log(f"segment {index} upload attempt {attempt + 1} failed: {exc}")
                await asyncio.sleep(2 * (attempt + 1))


# --------------------------------------------------------------------------
# Getting into a Google Meet
# --------------------------------------------------------------------------

_LEAVE = re.compile(r"leave call", re.I)
_DENIED = re.compile(
    r"(denied your request|can.t join this (video )?call|no one responded|removed from the meeting|"
    r"you can.t join|not allowed to join|meeting code is invalid|check your meeting code)",
    re.I,
)
_WAITING = re.compile(r"(asking to be let in|someone will let you in|waiting for the host|let you in soon)", re.I)
_SIGN_IN = re.compile(r"(sign in to join|sign in with your google|you need to sign in)", re.I)
_ENDED = re.compile(r"(you left the meeting|the meeting has ended|you.ve been removed|call ended|return to home screen)", re.I)


async def body_text(page: Page) -> str:
    try:
        return (await page.inner_text("body", timeout=3000)) or ""
    except Exception:  # noqa: BLE001
        return ""


async def click_if(page: Page, role: str, name: re.Pattern, timeout: int = 1500) -> bool:
    try:
        loc = page.get_by_role(role, name=name).first
        await loc.click(timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        return False


async def in_call(page: Page) -> bool:
    try:
        return await page.get_by_role("button", name=_LEAVE).first.is_visible(timeout=1000)
    except Exception:  # noqa: BLE001
        return False


async def join_meet(page: Page, name: str, mid: str, api: Api) -> None:
    """From the link to inside the call. Raises RuntimeError with the reason."""
    await page.wait_for_load_state("domcontentloaded")
    await asyncio.sleep(3)

    # A prompt about the microphone and camera. The bot has neither.
    await click_if(page, "button", re.compile(r"continue without (microphone and camera|camera|microphone)", re.I))
    await asyncio.sleep(1)

    text = await body_text(page)
    # A guest join screen ALSO offers "Sign in with your Google account" as an
    # optional extra, so the words alone prove nothing. Only a page with no way
    # to ask to join is a page that demands a login.
    can_ask = (
        await page.get_by_role(
            "button", name=re.compile(r"^(ask to join|join now)$", re.I)
        ).count()
    ) > 0
    if _SIGN_IN.search(text) and not can_ask and not STATE:
        raise RuntimeError(
            "This meeting needs a signed-in Google account; guests are switched off. "
            "Give the bot an account (MANKS_GOOGLE_STATE) or allow guests."
        )
    if _DENIED.search(text):
        raise RuntimeError("Meet refused the link: " + _DENIED.search(text).group(0))

    # Name, for guests.
    try:
        field = page.get_by_placeholder(re.compile(r"your name", re.I)).first
        if await field.is_visible(timeout=2500):
            await field.fill(name)
    except Exception:  # noqa: BLE001
        pass

    # Silent and unseen: turn off whatever is on.
    await click_if(page, "button", re.compile(r"turn off microphone", re.I), 800)
    await click_if(page, "button", re.compile(r"turn off camera", re.I), 800)

    joined = await click_if(
        page, "button", re.compile(r"^(ask to join|join now|join anyway|join)$", re.I), 4000
    )
    if not joined:
        await page.screenshot(path=f"/tmp/manks-{mid}-join.png")
        seen = " ".join((await body_text(page)).split())[:200]
        raise RuntimeError(f"Could not find the join button. The page said: {seen!r}")

    await api.status(mid, "waiting", "Waiting to be let in")
    deadline = time.monotonic() + ADMIT_WAIT_MINUTES * 60
    while time.monotonic() < deadline:
        if await in_call(page):
            return
        text = await body_text(page)
        if _DENIED.search(text):
            said = _DENIED.search(text).group(0)
            if re.search(r"can.t join this (video )?call", said, re.I) and "your meeting is safe" in text.lower():
                # Meet's own refusal, shown within seconds and with no reason.
                # The host never saw a request. Either Google recognised an
                # automated browser, or this meeting does not accept guests.
                raise RuntimeError(
                    "Meet refused the bot before the host could see it (\"You can't join this video call\"). "
                    "Either Google detected an automated browser, or this meeting does not allow guests "
                    "(common for Workspace meetings). Try a meeting from a personal Google account, or "
                    "give the bot a signed-in account with MANKS_GOOGLE_STATE."
                )
            raise RuntimeError("The host did not let the bot in: " + said)
        if await api.leave_requested(mid):
            raise RuntimeError("Cancelled before the bot was let in.")
        await asyncio.sleep(2)
    raise RuntimeError(f"Nobody let the bot in within {ADMIT_WAIT_MINUTES} minutes.")


# --------------------------------------------------------------------------
# Getting into a BigBlueButton room
# --------------------------------------------------------------------------
#
# The join link carries everything (meeting, password, name, a checksum), so
# there is no name to type and nothing to be admitted to unless the room is
# moderated. What it does show is an audio dialog, and the only choice that is
# silent is "Listen only": the other one opens a microphone, and the bot has a
# fake one that BEEPS. If the server does not offer listen-only the bot refuses
# rather than put a tone into somebody's meeting.

_BBB_ENDED = re.compile(
    r"(meeting (has )?ended|the meeting is not running|ended the meeting|"
    r"you have been (kicked|removed|ejected)|you were (kicked|removed|ejected)|"
    r"session (has )?(expired|ended)|logged out|has been closed)",
    re.I,
)
_BBB_BAD_LINK = re.compile(
    r"(checksum|invalid|not found|meeting is not running|wrong password|expired)", re.I
)


async def _bbb_audio_modal(page: Page, talk: bool = False) -> bool:
    try:
        choice = r"^microphone$|listen only" if talk else r"listen only"
        return await page.get_by_role("button", name=re.compile(choice, re.I)).first.is_visible(timeout=800)
    except Exception:  # noqa: BLE001
        return False


async def _bbb_ready(page: Page) -> bool:
    """The room's own controls are on screen.

    Read off a real room: the sidebar toggle (`toggleUserList`) exists once the
    client has loaded and disappears when you are removed or the meeting ends.
    Many rooms never show an audio dialog at all and connect the listener
    straight into listen-only (`leaveListenOnly` is then on the toolbar), so
    "no dialog" cannot be taken to mean "not in yet".
    """
    try:
        return await page.locator('[data-test="toggleUserList"]').first.is_visible(timeout=800)
    except Exception:  # noqa: BLE001
        return False


async def in_bbb(page: Page) -> bool:
    """Inside the room: on the client, controls present, not ended."""
    if "html5client" not in page.url:
        return False
    if _BBB_ENDED.search(await body_text(page)):
        return False
    return await _bbb_ready(page)


async def _bbb_echo_test(page: Page) -> None:
    """The echo test asks "can you hear yourself?"; the bot answers yes."""
    for _ in range(10):
        try:
            await page.locator('[data-test="echoYesBtn"]').first.click(timeout=1500)
            return
        except Exception:  # noqa: BLE001
            if await click_if(page, "button", re.compile(r"^yes$", re.I), 800):
                return
        await asyncio.sleep(1)


async def _bbb_take_the_mic(page: Page, mid: str) -> None:
    """Make sure the bot is in the audio WITH a microphone, and unmuted.

    A room that connects every newcomer straight into listen-only never shows
    the dialog, so the bot leaves that audio and joins again by microphone.
    Best effort and reported: the room decides whether a guest may speak.
    """
    leave = page.locator('[data-test="leaveListenOnly"], [data-test="leaveAudio"]').first
    try:
        if await leave.is_visible(timeout=800):
            await leave.click(timeout=2000)
            await asyncio.sleep(1)
        join = page.locator('[data-test="joinAudio"]').first
        if await join.is_visible(timeout=2000):
            await join.click(timeout=2000)
            await click_if(page, "button", re.compile(r"^microphone$", re.I), 4000)
            await _bbb_echo_test(page)
            await asyncio.sleep(2)
        unmute = page.locator('[data-test="unmuteMicButton"]').first
        if await unmute.is_visible(timeout=1500):
            await unmute.click(timeout=2000)
        log("microphone ready")
    except Exception as exc:  # noqa: BLE001
        log("could not open the microphone:", str(exc)[:100])
        await page.screenshot(path=f"/tmp/manks-{mid}-mic.png")


async def join_bbb(page: Page, mid: str, api: Api, talk: bool = False) -> None:
    await page.wait_for_load_state("domcontentloaded")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        text = await body_text(page)
        if "html5client" not in page.url and _BBB_BAD_LINK.search(text):
            await page.screenshot(path=f"/tmp/manks-{mid}-join.png")
            raise RuntimeError(
                "The room refused the link: "
                + " ".join(text.split())[:160]
                + ". Join links stop working when the meeting ends or the link expires."
            )
        if _BBB_ENDED.search(text):
            raise RuntimeError("The meeting is not running: " + " ".join(text.split())[:160])

        # An audio dialog, on servers that show one: only the silent choice.
        if await _bbb_audio_modal(page, talk):
            if talk and await click_if(page, "button", re.compile(r"^microphone$", re.I), 1500):
                await api.status(mid, "waiting", "Connecting the microphone")
                await _bbb_echo_test(page)
            elif await click_if(page, "button", re.compile(r"listen only", re.I), 1500):
                await api.status(mid, "waiting", "Connecting the audio")
            await asyncio.sleep(2)
            continue
        if (
            not talk
            and await page.get_by_role("button", name=re.compile(r"^microphone$", re.I)).count()
            and not await _bbb_ready(page)
        ):
            await page.screenshot(path=f"/tmp/manks-{mid}-join.png")
            raise RuntimeError(
                "This room does not offer 'Listen only', and the bot will not open a microphone."
            )

        if await in_bbb(page):
            # Give the audio a moment to connect before recording starts.
            await asyncio.sleep(3)
            if talk:
                await _bbb_take_the_mic(page, mid)
            return
        if await api.leave_requested(mid):
            raise RuntimeError("Cancelled before the bot joined.")
        await asyncio.sleep(2)
    await page.screenshot(path=f"/tmp/manks-{mid}-join.png")
    seen = " ".join((await body_text(page)).split())[:200]
    raise RuntimeError(f"Could not get into the room. The page said: {seen!r}")


ANNOUNCE = os.environ.get("MANKS_ANNOUNCE", "1") not in ("0", "false", "")
ANNOUNCE_TEXT = os.environ.get(
    "MANKS_ANNOUNCE_TEXT",
    "Manks (notetaker) has joined. It is listening and recording the audio of this meeting "
    "to produce a transcript and notes.",
)


async def announce_bbb(page: Page) -> int:
    """Tell the people already here that a recorder has joined. Returns how many.

    A recording bot that arrives silently is a bad surprise, so it says so. The
    rooms seen so far have no public chat -- only private messages -- so the
    notice goes privately to each other participant (capped), moderators first.
    Best effort: a failure here never stops the recording.
    """
    if not ANNOUNCE:
        return 0
    sent = 0
    try:
        await page.locator('[data-test="toggleUserList"]').first.click(timeout=3000)
        await page.locator('[data-test="userListItem"]').first.wait_for(timeout=8000)
        items = page.locator('[data-test="userListItem"]')
        total = await items.count()
        order: list[int] = []
        for i in range(total):
            text = await items.nth(i).inner_text()
            if "(You)" in text:
                continue
            moderator = await items.nth(i).locator('[data-test="moderatorAvatar"]').count()
            order.insert(0, i) if moderator else order.append(i)
        for i in order[:5]:
            try:
                await items.nth(i).click(timeout=3000)
                await page.locator('[data-test="startPrivateChat"]').first.click(timeout=3000)
                box = page.locator("textarea#message-input, textarea").first
                await box.fill(ANNOUNCE_TEXT, timeout=3000)
                await box.press("Enter")
                sent += 1
                await asyncio.sleep(0.8)
                # Back to the list for the next person.
                await page.locator('[data-test="hidePrivateChat"], [data-test="closePrivateChat"]').first.click(timeout=1500)
                await asyncio.sleep(0.5)
                if not await items.first.is_visible(timeout=800):
                    await page.locator('[data-test="toggleUserList"]').first.click(timeout=2000)
            except Exception as exc:  # noqa: BLE001
                log("announce to one person failed:", str(exc)[:80])
                break
    except Exception as exc:  # noqa: BLE001
        log("announce failed:", str(exc)[:100])
    return sent


async def leave_bbb(page: Page) -> None:
    # The room's own Leave button, then its confirmation; failing that, just
    # drop the connection. Either way the bot leaves the participant list.
    try:
        await page.get_by_role("button", name=re.compile(r"^leave$", re.I)).first.click(timeout=2000)
        await asyncio.sleep(0.6)
        for pattern in (r"^(leave meeting|log ?out|ok|yes|confirm)$",):
            if await click_if(page, "button", re.compile(pattern, re.I), 1500):
                return
        await page.locator('[data-test="confirmEndMeeting"], [data-test="logout-button"]').first.click(timeout=1500)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        await page.goto("about:blank")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
# Talking: the page's audio to the API, the API's speech into the page
# --------------------------------------------------------------------------


async def talk_bridge(page: Page, mid: str):
    """Open the API's voice line for this meeting; returns the task running it."""
    import websockets  # only a talking bot needs it

    url = API.replace("https://", "wss://").replace("http://", "ws://") + f"/manks/bot/{mid}/talk"
    ws = await websockets.connect(url, additional_headers=HEADERS, max_size=None)
    first = json.loads(await ws.recv())
    if first.get("type") != "ready":
        await ws.close()
        raise RuntimeError(first.get("detail") or "The voice line refused the bot.")

    async def up(b64: str) -> None:
        try:
            await ws.send(base64.b64decode(b64))
        except Exception:  # noqa: BLE001 - the line closed; the call goes on
            pass

    await page.expose_function("__manksPcm", up)

    async def down() -> None:
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    await page.evaluate("(b) => window.__manks.speak(b)", base64.b64encode(message).decode())
                else:
                    event = json.loads(message)
                    if event.get("type") == "stop":
                        await page.evaluate("window.__manks.hush()")
                    elif event.get("type") == "say":
                        log("said:", str(event.get("text"))[:80])
        except Exception as exc:  # noqa: BLE001
            log("voice line ended:", str(exc)[:80])

    task = asyncio.create_task(down())

    class Line:
        async def hello(self) -> None:
            """In the room with a working microphone: the API may speak first."""
            await ws.send(json.dumps({"type": "joined"}))

        def cancel(self) -> None:
            task.cancel()
            asyncio.create_task(ws.close())

    return Line()


async def _page_value(page: Page, js: str, default):  # noqa: ANN001
    """Evaluate, or `default` if the page is navigating or gone.

    The end of a meeting is exactly when this happens: the room redirects to
    its logout page, and a failure to read a number off it is not a failure
    of the meeting.
    """
    try:
        return await page.evaluate(js)
    except Exception:  # noqa: BLE001
        return default


async def stay_in_the_room(page: Page) -> None:
    """Refuse the room's redirect away when the meeting ends.

    Ending a meeting for everyone sends each client to a logout URL (here a
    marketing site). The page that holds the recorder, and the part of the
    audio not yet uploaded, would go with it. Blocking top-level navigation
    away from the client keeps the room's "meeting ended" screen up, which is
    also how the bot recognises the end.
    """

    async def guard(route) -> None:  # noqa: ANN001
        req = route.request
        if req.is_navigation_request() and req.frame == page.main_frame and "html5client" not in req.url:
            await route.abort()
        else:
            await route.continue_()

    await page.route("**/*", guard)


# --------------------------------------------------------------------------
# One meeting, start to finish
# --------------------------------------------------------------------------


async def run_meeting(p, api: Api, job: dict) -> None:
    mid, url, name = job["id"], job["url"], job.get("name") or "Manks"
    platform = job.get("platform") or "meet"
    # A BigBlueButton link IS a credential (password and checksum are in it),
    # so only the host is ever written to the log.
    from urllib.parse import urlparse

    log("meeting", mid, platform, urlparse(url).netloc)

    browser = await p.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            # Let WebAudio start without a click, and grant (fake) devices so
            # Meet does not stall on a permission prompt it cannot answer.
            "--autoplay-policy=no-user-gesture-required",
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            "--disable-blink-features=AutomationControlled",
        ],
    )
    line = None
    try:
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            storage_state=STATE or None,
            permissions=["microphone", "camera"],
        )
        talk = bool(job.get("talk")) and platform == "bbb"
        if talk:
            await ctx.add_init_script("window.__manksTalk = true;")
        await ctx.add_init_script(INIT_JS)
        page = await ctx.new_page()

        async def on_segment(index: int, start: float, seconds: float, b64: str) -> None:
            await api.segment(mid, int(index), float(start), float(seconds), base64.b64decode(b64))

        await page.expose_function("__manksSegment", on_segment)

        await api.status(mid, "joining", "Opening the meeting")
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        if platform == "bbb":
            if talk:
                line = await talk_bridge(page, mid)
            await join_bbb(page, mid, api, talk)
            present = in_bbb
        else:
            await join_meet(page, name, mid, api)
            present = in_call

        log("in the call")
        if platform == "bbb":
            await stay_in_the_room(page)
        await api.status(mid, "in_meeting", "Listening and talking" if talk else "Listening")
        if talk and line:
            await line.hello()
        await page.evaluate(f"window.__manks.start({SEGMENT})")
        if talk:
            await page.evaluate("window.__manks.tap()")
        if platform == "bbb":
            told = await announce_bbb(page)
            log(f"told {told} participant(s) that it is recording")

        started = time.monotonic()
        absent = 0
        reason = "The meeting ended."
        while True:
            await asyncio.sleep(3)
            if await api.leave_requested(mid):
                reason = "Asked to leave."
                break
            if time.monotonic() - started > MAX_MINUTES * 60:
                reason = f"Left after the {MAX_MINUTES}-minute limit."
                break
            if await present(page):
                absent = 0
            else:
                absent += 1
                # "The meeting has ended" on screen is final; anything else
                # (a reconnect, a slow render) gets a couple more looks.
                if absent >= 3 or _BBB_ENDED.search(await body_text(page)):
                    break
            quiet = await _page_value(page, "(Date.now() - window.__manks.lastSound) / 1000", 0)
            if quiet > SILENCE_MINUTES * 60:
                reason = f"Left after {SILENCE_MINUTES} minutes of silence."
                break

        log("leaving:", reason)
        # Stop and upload the last part FIRST, while the room is still on
        # screen; leaving can navigate the page away and take it with it.
        await _page_value(page, "window.__manks && window.__manks.stop && window.__manks.stop()", None)
        for _ in range(60):
            if await _page_value(page, "!window.__manks || window.__manks.done", True):
                break
            await asyncio.sleep(0.5)
        try:
            if platform == "bbb":
                await leave_bbb(page)
            elif await in_call(page):
                await click_if(page, "button", _LEAVE, 3000)
        except Exception as exc:  # noqa: BLE001 - the room is already gone
            log("leave:", str(exc)[:80])
        await api.status(mid, "ended", reason)
    except Exception as exc:  # noqa: BLE001
        log("failed:", exc)
        # Whatever was already recorded is kept and analysed by the API.
        try:
            await page.evaluate("window.__manks && window.__manks.stop()")  # type: ignore[name-defined]
            await asyncio.sleep(2)
        except Exception:  # noqa: BLE001
            pass
        await api.status(mid, "failed", str(exc)[:380])
    finally:
        if line:
            line.cancel()
        await browser.close()


# --------------------------------------------------------------------------
# A self-test that needs no meeting: a loopback call carrying a tone.
# --------------------------------------------------------------------------

SELFTEST_PAGE = """
<html><body><script>
(async () => {
  const a = new RTCPeerConnection(), b = new RTCPeerConnection();
  a.onicecandidate = e => e.candidate && b.addIceCandidate(e.candidate);
  b.onicecandidate = e => e.candidate && a.addIceCandidate(e.candidate);
  const ctx = new AudioContext();
  const gain = ctx.createGain(); gain.gain.value = 0.9;
  const dest = ctx.createMediaStreamDestination();
  gain.connect(dest);
  if (window.__speechB64) {
    const bin = atob(window.__speechB64); const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const buf = await ctx.decodeAudioData(bytes.buffer);
    const src = ctx.createBufferSource(); src.buffer = buf; src.connect(gain); src.start();
  } else {
    const osc = ctx.createOscillator(); osc.frequency.value = 440;
    osc.connect(gain); osc.start();
  }
  dest.stream.getTracks().forEach(t => a.addTrack(t, dest.stream));
  b.ontrack = e => { const el = new Audio(); el.srcObject = e.streams[0]; el.play().catch(()=>{}); window.__remote = true; };
  const offer = await a.createOffer(); await a.setLocalDescription(offer);
  await b.setRemoteDescription(offer);
  const answer = await b.createAnswer(); await b.setLocalDescription(answer);
  await a.setRemoteDescription(answer);
})();
</script></body></html>
"""


async def selftest(meeting_id: str) -> None:
    """Record 8 seconds of a loopback tone in 3-second segments and upload them."""
    api = Api()
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required"]
        )
        ctx = await browser.new_context()
        await ctx.add_init_script(INIT_JS)
        sample = os.environ.get("MANKS_SELFTEST_PCM", "")
        if sample and os.path.exists(sample):
            import struct

            pcm = open(sample, "rb").read()  # 16 kHz mono signed 16-bit
            header = (
                b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
                + b"data" + struct.pack("<I", len(pcm))
            )
            await ctx.add_init_script(
                "window.__speechB64='" + base64.b64encode(header + pcm).decode() + "'"
            )
        page = await ctx.new_page()
        page.on('console', lambda m: log('page:', m.text))
        page.on('pageerror', lambda e: log('page error:', e))
        got: list[int] = []

        async def on_segment(index, start, seconds, b64):
            data = base64.b64decode(b64)
            got.append(len(data))
            await api.segment(meeting_id, int(index), float(start), float(seconds), data)

        await page.expose_function("__manksSegment", on_segment)
        await page.goto("data:text/html," + SELFTEST_PAGE.replace("\n", " ").replace("#", "%23"))
        await asyncio.sleep(3)
        tracks = await page.evaluate("window.__manks.tracks")
        log("remote audio tracks hooked:", tracks)
        seg = int(os.environ.get("MANKS_SELFTEST_SEGMENT", "3"))
        await page.evaluate(f"window.__manks.start({seg})")
        await asyncio.sleep(float(os.environ.get("MANKS_SELFTEST_SECONDS", "8")))
        level = await page.evaluate("window.__manks.level")
        await page.evaluate("window.__manks.stop()")
        for _ in range(40):
            if await page.evaluate("window.__manks.done"):
                break
            await asyncio.sleep(0.5)
        log(f"level {level:.3f}; segments uploaded: {len(got)} sizes {got}")
        await browser.close()
    if not got or tracks < 1:
        sys.exit(1)


# --------------------------------------------------------------------------
# A BigBlueButton meeting joined without a browser
# --------------------------------------------------------------------------


async def native_talk_line(bot, mid: str):  # noqa: ANN001
    """The API's voice line for a bot with no browser: PCM both ways."""
    import websockets
    url = API.replace("https://", "wss://").replace("http://", "ws://") + f"/manks/bot/{mid}/talk"
    ws = await websockets.connect(url, additional_headers=HEADERS, max_size=None)
    first = json.loads(await ws.recv())
    if first.get("type") != "ready":
        await ws.close()
        raise RuntimeError(first.get("detail") or "The voice line refused the bot.")
    up: asyncio.Queue = asyncio.Queue(maxsize=400)

    def heard(pcm: bytes) -> None:
        try:
            up.put_nowait(pcm)
        except asyncio.QueueFull:  # the line is stuck; drop, never block the recorder
            pass

    bot.on_pcm = heard

    async def send_up() -> None:
        while True:
            pcm = await up.get()
            await ws.send(pcm)

    async def down() -> None:
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    bot.voice.push(message, 24_000)
                else:
                    event = json.loads(message)
                    if event.get("type") == "stop":
                        bot.voice.hush()
                    elif event.get("type") == "say":
                        log("said:", str(event.get("text"))[:80])
        except Exception as exc:  # noqa: BLE001
            log("voice line ended:", str(exc)[:80])

    tasks = [asyncio.create_task(send_up()), asyncio.create_task(down())]

    class Line:
        async def hello(self) -> None:
            await ws.send(json.dumps({"type": "joined"}))

        def cancel(self) -> None:
            for t in tasks:
                t.cancel()
            asyncio.create_task(ws.close())

    return Line()


async def run_native(api: Api, job: dict) -> None:
    from urllib.parse import urlparse

    from bbb_native import BBBNative, JoinFailed

    mid, url = job["id"], job["url"]
    talk = bool(job.get("talk"))
    log("meeting", mid, "bbb_native", urlparse(url).netloc, "talking" if talk else "")
    bot = BBBNative(url, segment_seconds=SEGMENT, talk=talk)
    line = None

    async def sink(index: int, start: float, seconds: float, data: bytes) -> None:
        await api.segment(mid, index, start, seconds, data)

    try:
        await api.status(mid, "joining", "Connecting without a browser")
        if talk:
            line = await native_talk_line(bot, mid)
        await bot.join(sink)
        await api.status(mid, "in_meeting", "Listening and talking" if talk else "Listening")
        if line:
            await line.hello()
        started = time.monotonic()
        reason = "The meeting ended."
        while not bot.ended.is_set():
            await asyncio.sleep(3)
            if await api.leave_requested(mid):
                reason = "Asked to leave."
                break
            if time.monotonic() - started > MAX_MINUTES * 60:
                reason = f"Left after the {MAX_MINUTES}-minute limit."
                break
            if time.monotonic() - bot.last_sound > SILENCE_MINUTES * 60:
                reason = f"Left after {SILENCE_MINUTES} minutes of silence."
                break
        else:
            reason = bot.end_reason or reason
        log("leaving:", mid, reason)
        if line:
            line.cancel()
        await bot.close()
        await api.status(mid, "ended", reason)
    except JoinFailed as exc:
        log("failed:", mid, exc)
        await bot.close()
        await api.status(mid, "failed", str(exc)[:380])
    except Exception as exc:  # noqa: BLE001
        log("failed:", mid, type(exc).__name__, exc)
        try:
            await bot.close()
        except Exception:  # noqa: BLE001
            pass
        await api.status(mid, "failed", f"{type(exc).__name__}: {exc}"[:380])


async def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "selftest":
        await selftest(sys.argv[2])
        return
    if not SECRET:
        sys.exit("MANKS_BOT_SECRET is not set.")
    api = Api()
    native = MODE == "native"
    limit = MAX_NATIVE if native else MAX_BROWSER
    platforms = ["bbb_native"] if native else ["meet", "bbb"]
    log(f"manks bot ready ({MODE}, up to {limit} at once); polling", API)

    running: set[asyncio.Task] = set()

    async def drive(job: dict, pw) -> None:
        try:
            if native:
                await run_native(api, job)
            else:
                await run_meeting(pw, api, job)
        except Exception as exc:  # noqa: BLE001 - one meeting must not take the others down
            log("meeting crashed:", job.get("id"), type(exc).__name__, exc)
            await api.status(job["id"], "failed", f"{type(exc).__name__}: {exc}"[:380])

    async def loop(pw) -> None:
        while True:
            running.difference_update({t for t in running if t.done()})
            if len(running) >= limit:
                await asyncio.sleep(2)
                continue
            try:
                job = await api.claim(platforms)
            except Exception as exc:  # noqa: BLE001 - the API may be restarting
                log("claim failed:", exc)
                await asyncio.sleep(10)
                continue
            if job is None:
                await asyncio.sleep(3)
                continue
            running.add(asyncio.create_task(drive(job, pw)))

    if native:
        await loop(None)
    else:
        async with async_playwright() as p:
            await loop(p)


if __name__ == "__main__":
    asyncio.run(main())
