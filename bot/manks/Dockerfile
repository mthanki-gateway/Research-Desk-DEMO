# The Manks browser bot: a real Chromium that joins a Google Meet (or a BigBlueButton
# room) and records the room.
#
# Its own image and its own container, on purpose. A browser is hundreds of
# megabytes, has a different failure profile from the API, and should not share
# a process with request handling. It holds NO model keys: it authenticates to
# the API with MANKS_BOT_SECRET and uploads audio, nothing more.
#
# The base image's tag must match the `playwright` version below, because the
# browsers inside it are built for exactly that release.
FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /bot
RUN pip install --no-cache-dir playwright==1.49.0 httpx websockets
COPY bot.py ./

ENV MANKS_MODE=browser
CMD ["python", "-u", "bot.py"]
