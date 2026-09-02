# Test fixtures

The two documents every measurement in the main `README.md` refers to. Committed
so results stay reproducible after `docker compose down -v` wipes the volumes.

```powershell
# re-ingest both
curl.exe -s -X POST http://localhost:8000/documents -F "file=@fixtures/acme-report.md;type=text/markdown"
curl.exe -s -X POST http://localhost:8000/documents -F "file=@fixtures/antiquity.md;type=text/markdown"
```

Wait for `status: ready` (a few seconds for these — the limiter throttles to
~133 chunks/minute, and these are 6 and 5 chunks).

## Why these two

**`acme-report.md`** — a fake annual report whose facts sit in *different
sections*, which is what makes multi-hop retrieval testable. It also has one
segment that is "roughly flat", so negation can be tested without ambiguity.

**`antiquity.md`** — four unrelated topics, and critically it contains **neither
the word "Egypt" nor "pyramid"**. Searching for either and getting the Giza
section back is the cleanest possible demonstration that retrieval is
concept-based rather than keyword-based.

## The canonical regression test

```powershell
$b = @{ question = 'What was operating income in 2024, and what capital expenditure is planned for 2025?'; top_k = 1 } | ConvertTo-Json -Compress
Invoke-RestMethod http://localhost:8000/ask      -Method Post -ContentType 'application/json' -Body $b
Invoke-RestMethod http://localhost:8000/research -Method Post -ContentType 'application/json' -Body $b
```

Expected at `top_k=1`:

| endpoint | result |
|---|---|
| `/ask` (baseline) | operating income only — **misses the capex**, by design |
| `/research` (agent) | **both figures**, correct |

The baseline failing here is the point: one retrieval pass can only surface
`top_k` passages. If `/research` stops passing, the agent is broken.

## Other useful checks

| Query | Expect |
|---|---|
| `Egypt` | Giza / mummification sections — word absent from the corpus |
| `preserving a dead body` | Mummification, ~0.70 — zero shared words |
| `167.2` | correct chunk, but by a ~0.007 margin — the exact-value weakness |
| `quantum computing error correction` | ~0.57 against an unrelated corpus — proves scores are not thresholdable |
| `What is the CEO salary?` | an honest "not in the documents" |
