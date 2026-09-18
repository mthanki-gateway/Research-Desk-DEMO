import { getAccessToken } from "./supabase";

/**
 * Two base URLs on purpose: server components resolve `api` over the compose
 * network, browsers can only reach `localhost`. Getting this wrong is the
 * classic first-day Docker + Next.js bug.
 */
const serverBase = process.env.API_URL_INTERNAL ?? "http://localhost:8000";
export const browserBase =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Every browser-side request goes through here, so the Authorization header is
 * attached in exactly one place. Patching ~15 individual fetch calls would
 * have guaranteed that one of them was missed — and a missed header is a 401
 * on an endpoint that worked yesterday.
 *
 * With auth disabled `getAccessToken()` returns null and no header is sent,
 * which is precisely what the backend's anonymous mode expects.
 */
async function authedFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const token = await getAccessToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${browserBase}${path}`, { ...init, headers });
}

/** JSON POST/PATCH helper — sets the content type and serialises the body. */
async function authedJson(
  path: string,
  method: string,
  body?: unknown,
): Promise<Response> {
  return authedFetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export type Health = {
  status: string;
  llm_model: string;
  embedding_provider: string;
  google_api_key_present: boolean;
};

// ---------------------------------------------------------------------------
// Evaluation (Tier 1 — retrieval metrics)
//
// Mirrors backend/app/services/eval_runner.py. Hand-written, like every other
// type here, which means it can drift from the Python if a field is renamed;
// generating these from the OpenAPI schema is the standing fix.
// ---------------------------------------------------------------------------

/** Suite-level means at one k. Null where a metric is undefined. */
export type EvalAggregate = {
  k: number;
  n_questions: number;
  /** Answerable questions only — unanswerable ones are excluded, not zeroed. */
  n_scored: number;
  hit_rate: number | null;
  precision: number | null;
  recall: number | null;
  mrr: number | null;
  map: number | null;
  ndcg: number | null;
};

export type EvalQuestionScores = {
  k: number;
  n_retrieved: number;
  total_relevant: number;
  hit: boolean;
  precision: number;
  recall: number | null;
  reciprocal_rank: number;
  average_precision: number | null;
  ndcg: number | null;
};

export type EvalHit = {
  rank: number;
  /** For opening the full chunk in the side panel — the preview is truncated. */
  chunk_id: string;
  filename: string;
  heading: string | null;
  score: number;
  relevant: boolean;
  preview: string;
  n_chars: number;
};

/** A labelled fact the question needs, and whether retrieval found it. */
export type EvalExpectedFact = {
  index: number;
  file: string;
  must_contain: string[];
  /** null = not satisfied by any retrieved chunk, at any depth. */
  found_at_rank: number | null;
  /** Chunks that DO satisfy this label, found by SQL rather than retrieval.
   *  Empty means the label matches nothing in the corpus — a broken label, not
   *  a retrieval failure. */
  matching_chunk_ids: string[];
};

export type EvalQuestionResult = {
  id: string;
  question: string;
  tags: string[];
  answerable: boolean;
  total_specs: number;
  specs_satisfied: number;
  expected: EvalExpectedFact[];
  expected_answer: string;
  expect_refusal: boolean;
  /** spec index -> rank it was first found at. A fact at rank 9 with top_k=5
   *  is a RANKING failure, not a retrieval one. */
  satisfied_at: Record<string, number>;
  unsatisfied_specs: number[];
  hits: EvalHit[];
  /** keyed by k as a string */
  scores: Record<string, EvalQuestionScores>;
};

export type EvalReport = {
  config: {
    top_k: number;
    k_values: number[];
    multi_query: boolean;
    n_questions: number;
    scoped_to_documents: string[] | null;
    /** Empty = the full suite ran. Anything else means these metrics cover a
     *  SUBSET and must not be compared against a full-suite number. */
    filters: { tags?: string[]; ids?: string[] };
    question_ids: string[];
  };
  elapsed_seconds: number;
  n_embedding_calls: number;
  aggregates: Record<string, EvalAggregate>;
  questions: EvalQuestionResult[];
};

export type GoldenSet = {
  n_questions: number;
  n_answerable: number;
  n_unanswerable: number;
  tags: string[];
  questions: {
    id: string;
    question: string;
    tags: string[];
    answerable: boolean;
    n_facts: number;
  }[];
};

export type EvalCorpus = {
  documents: { filename: string; chunks: number }[];
  n_documents: number;
  n_chunks: number;
};

export async function getGoldenSet(): Promise<GoldenSet> {
  const res = await authedFetch("/eval/golden", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function getEvalCorpus(): Promise<EvalCorpus> {
  const res = await authedFetch("/eval/corpus", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function runTier1(opts: {
  topK?: number | null;
  kValues?: number[];
  multiQuery?: boolean;
  tags?: string[];
}): Promise<EvalReport> {
  const res = await authedJson("/eval/tier1", "POST", {
    top_k: opts.topK ?? null,
    k_values: opts.kValues ?? [1, 3, 5, 10],
    multi_query: opts.multiQuery ?? false,
    tags: opts.tags ?? [],
    ids: [],
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export type DocStatus =
  | "pending"
  | "parsing"
  | "embedding"
  | "ready"
  | "failed";

export type Document = {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: DocStatus;
  error: string | null;
  n_pages: number;
  n_chunks: number;
  n_embedded: number;
  created_at: string;
  owner_id: string | null;
  meta: Record<string, unknown>;
};

export type SearchHit = {
  chunk_id: string;
  document_id: string;
  filename: string;
  page: number | null;
  chunk_index: number;
  heading: string | null;
  text: string;
  score: number;
  meta: Record<string, unknown>;
  rrf_score: number | null;
  found_by: string[] | null;
  source?: "document" | "web";
  url?: string | null;
};

export type Stats = {
  documents: number;
  chunks_in_postgres: number;
  vectors_in_qdrant: number;
  embedding_provider: string;
  embedding_dim: number;
};

export async function getHealth(): Promise<Health | { error: string }> {
  try {
    const res = await fetch(`${serverBase}/health`, { cache: "no-store" });
    if (!res.ok) return { error: `API returned ${res.status}` };
    return (await res.json()) as Health;
  } catch (e) {
    return { error: e instanceof Error ? e.message : "unreachable" };
  }
}

/** Pull the API's error detail out, rather than showing a bare status code. */
async function detail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    return `Request failed (${res.status})`;
  }
}

export async function listDocuments(): Promise<Document[]> {
  const res = await authedFetch("/documents", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export type UploadResult = {
  document: Document;
  /** The API already had these exact bytes — 200, not 201. Nothing was
   *  ingested and no embedding quota was spent. */
  duplicate: boolean;
};

export async function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  // No Content-Type here on purpose: the browser must set it, because it has
  // to append the multipart boundary.
  const res = await authedFetch("/documents", { method: "POST", body: form });
  if (!res.ok) throw new Error(await detail(res));
  // The status code carries the answer rather than a field in the body: it is
  // what HTTP already means by 201-created versus 200-here-it-is, and it keeps
  // DocumentOut a description of the document rather than of the request.
  return { document: await res.json(), duplicate: res.status === 200 };
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await authedFetch(`/documents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await detail(res));
}

export async function search(q: string, limit = 5): Promise<SearchHit[]> {
  const res = await authedFetch(
    `/search?q=${encodeURIComponent(q)}&limit=${limit}`,
    { cache: "no-store" },
  );
  if (!res.ok) throw new Error(await detail(res));
  const body = await res.json();
  return body.hits as SearchHit[];
}

export type AskResult = {
  question: string;
  answer: string;
  sources: SearchHit[];
  sources_used: number[];
};

export async function ask(
  question: string,
  opts: { topK?: number; documentIds?: string[]; multiQuery?: boolean } = {},
): Promise<AskResult> {
  const res = await authedJson("/ask", "POST", {
    question,
    top_k: opts.topK ?? null,
    document_ids: opts.documentIds ?? null,
    multi_query: opts.multiQuery ?? null,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export type TraceStep = {
  node: string;
  sub_questions?: string[];
  queries?: { query: string; n: number }[];
  cited?: number[];
  unanswered?: string[];
  sufficient?: boolean;
  /** Answer is knowingly incomplete: resolve kept what was supported and named the gap. */
  partial?: boolean;
  /** "remember" when the turn stored a preference instead of searching. */
  intent?: string;
  /** Preferences this turn saved, so the UI can announce the change. */
  memory_saved?: string[];
  missing?: string[];
  assessment?: string;
  iteration?: number;
  n_sources?: number;
};

export type ResearchResult = AskResult & {
  sub_questions: string[];
  critique: string;
  sufficient: boolean;
  partial?: boolean;
  iterations: number;
  trace: TraceStep[];
};

export async function research(
  question: string,
  opts: { topK?: number; documentIds?: string[]; multiQuery?: boolean } = {},
): Promise<ResearchResult> {
  const res = await authedJson("/research", "POST", {
    question,
    top_k: opts.topK ?? null,
    document_ids: opts.documentIds ?? null,
    multi_query: opts.multiQuery ?? null,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

// ------------------------------------------------------------------ chunks

export type Chunk = {
  id: string;
  document_id: string;
  chunk_index: number;
  page: number | null;
  heading: string | null;
  text: string;
  n_chars: number;
  filename: string;
};

export async function getChunk(id: string): Promise<Chunk> {
  const res = await authedFetch(`/chunks/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function getDocumentChunks(id: string): Promise<Chunk[]> {
  const res = await authedFetch(`/documents/${id}/chunks`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

// ---------------------------------------------------------------- sessions

export type ChatSession = {
  id: string;
  title: string;
  document_ids: string[];
  summary: string | null;
  summarised_upto: number;
  owner_id: string | null;
  created_at: string;
  updated_at: string;
  n_messages: number;
};

/**
 * One numbered citation on an assistant message. `n` is the number the model
 * writes as `[n]` in its prose, which is what lets the answer renderer resolve
 * an inline marker back to the passage it refers to.
 */
export type MessageSource = {
  n: number;
  chunk_id: string;
  filename: string;
  heading: string | null;
  page: number | null;
  score: number;
  /** "document" or "web". Absent on turns stored before web search
   *  existed, hence optional. */
  source?: "document" | "web";
  /** Present only for web sources — there is no chunk to open. */
  url?: string | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: MessageSource[];
  agent_meta: {
    sources_used?: number[];
    sub_questions?: string[];
    iterations?: number;
    sufficient?: boolean;
    partial?: boolean;
    /** "remember" when the turn stored a preference instead of searching. */
    intent?: string;
    /** Preferences this turn saved, so the UI can announce the change. */
    memory_saved?: string[];
    critique?: string;
    /** Set only when the user was asked to clarify and answered. */
    clarification?: string | null;
    /** Langfuse trace id. Present only on turns that ran with tracing on. */
    trace_id?: string | null;
  };
  created_at: string;
};

export type SessionDetail = ChatSession & { messages: ChatMessage[] };

export async function createSession(
  documentIds: string[] = [],
): Promise<ChatSession> {
  const res = await authedJson("/sessions", "POST", {
    document_ids: documentIds,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/**
 * Research Desk's chats — Parley's spoken conversations are excluded
 * server-side, by `kind`.
 *
 * Paginated. `total` comes from a header rather than a wrapper object so the
 * body stays a plain array and every existing caller keeps working.
 */
export async function listSessions(
  limit = 25,
  offset = 0,
): Promise<{ sessions: ChatSession[]; total: number }> {
  const res = await authedFetch(`/sessions?limit=${limit}&offset=${offset}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await detail(res));
  const sessions = await res.json();
  return {
    sessions,
    // Falls back to the page length when the header is missing — a proxy that
    // strips it should degrade to "no more pages", not to zero results.
    total: Number(res.headers.get("X-Total-Count") ?? sessions.length),
  };
}

export async function getSession(id: string): Promise<SessionDetail> {
  const res = await authedFetch(`/sessions/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function updateSession(
  id: string,
  patch: { title?: string; document_ids?: string[] },
): Promise<ChatSession> {
  const res = await authedJson(`/sessions/${id}`, "PATCH", patch);
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  const res = await authedFetch(`/sessions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await detail(res));
}

/** One remembered instruction. `source_message` is the turn it came from. */
export type Memory = {
  id: string;
  text: string;
  source_message: string | null;
  active: boolean;
  created_at: string;
};

/**
 * What is remembered about one conversation. `summary` and `preferences` are
 * different kinds of memory and stay separate: the summary is a lossy
 * compression of what was discussed, the preferences are instructions kept
 * verbatim and reapplied every turn. Only the latter can be forgotten.
 */
export type ConversationMemory = {
  session_id: string;
  title: string;
  summary: string | null;
  preferences: Memory[];
};

export type ProfileMemory = {
  user_preferences: Memory[];
  conversations: ConversationMemory[];
};

export async function getMemory(): Promise<ProfileMemory> {
  const res = await authedFetch("/profile/memory");
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function forgetMemory(id: string): Promise<void> {
  const res = await authedFetch(`/profile/memory/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await detail(res));
}

export type DoneEvent = {
  answer: string;
  sources: SearchHit[];
  sources_used: number[];
  sub_questions: string[];
  critique: string;
  sufficient: boolean;
  iterations: number;
  trace: TraceStep[];
  context_chars: number;
  /** What the user said when asked to clarify. Null on an ordinary turn. */
  clarification?: string | null;
  /** Langfuse trace id, so feedback can be attached to this turn later. */
  trace_id?: string | null;
};

export type ClarifyOption = { label: string; description: string };

/**
 * The graph paused to ask the user what they meant.
 *
 * `thread_id` is the checkpoint key and the only way back to this state — the
 * pause lives in Postgres, not in the open connection, so resuming is a fresh
 * request that may land on a different worker.
 */
export type InterruptEvent = {
  type: string;
  /** The clarifying question, written by the model. */
  question: string;
  /** 2–4 concrete choices, grounded in the documents' actual headings. */
  options: ClarifyOption[];
  /** What the user originally typed. */
  original: string;
  actions: string[];
  thread_id: string;
};

/**
 * A stream ends one of two ways, and callers must handle both. A discriminated
 * union rather than a nullable `done`, so TypeScript forces the paused branch
 * to be considered instead of letting it be forgotten.
 */
export type TurnOutcome =
  | { status: "done"; done: DoneEvent }
  | { status: "paused"; interrupt: InterruptEvent };

export type ClarifyDecision =
  /** Narrow the search. `answer` is a chosen option's label or free text — the
   *  graph treats both identically, so "something else" is not a special case. */
  | { action: "answer"; answer: string }
  /** Search the original question as written. */
  | { action: "skip" }
  /** Stop without searching. */
  | { action: "cancel" };

/**
 * Shared SSE reader for both starting and resuming a turn.
 *
 * EventSource can only issue GET requests, so this uses fetch + a ReadableStream
 * and parses the SSE frames by hand. That is the standard workaround for POST
 * server-sent events.
 */
async function readTurnStream(
  res: Response,
  onProgress?: (node: string, detail: string) => void,
  onActivity?: (activity: Activity) => void,
): Promise<TurnOutcome> {
  if (!res.ok) throw new Error(await detail(res));
  if (!res.body) throw new Error("No response body to stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let outcome: TurnOutcome | null = null;

  while (true) {
    const { value, done: finished } = await reader.read();
    if (finished) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. A frame may arrive split
    // across reads, so keep the trailing partial in the buffer.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const eventLine = frame.split("\n").find((l) => l.startsWith("event:"));
      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!eventLine || !dataLine) continue;

      const event = eventLine.slice(6).trim();
      const payload = JSON.parse(dataLine.slice(5).trim());

      if (event === "progress") onProgress?.(payload.node, payload.detail);
      else if (event === "activity") onActivity?.(payload as Activity);
      else if (event === "done") outcome = { status: "done", done: payload };
      else if (event === "interrupt")
        outcome = { status: "paused", interrupt: payload };
      else if (event === "error") throw new Error(payload.detail);
    }
  }

  if (!outcome) throw new Error("Stream ended without a result");
  return outcome;
}

/**
 * One search or rerank happening inside a node.
 *
 * Separate from node progress because they answer different questions: a node
 * event says WHICH STAGE is running, an activity says WHAT IT IS DOING. A turn
 * that spends six seconds on the web showed one motionless "retrieve" line
 * without these.
 */
export type Activity = {
  /**
   * "search" | "search_done" — reading inside documents or the web.
   * "lookup" | "lookup_done" — collection metadata: names, counts, sizes.
   * "remember"               — storing a preference. The one tool that writes.
   * "rerank"                 — no query to show; not rendered.
   *
   * Lookups are a separate verb from searches because they are a different
   * claim. Counting documents is not reading them, and showing both as
   * "Searched your documents" is what made an answer derived entirely from
   * metadata look like it came from retrieval.
   */
  kind: string;
  /** "documents" | "web" */
  source?: string;
  query?: string;
  n?: number;
  /** For lookups: which metadata tool ran. */
  tool?: string;
  /** For "remember": what was stored. */
  text?: string;
};

/** Start a turn. Resolves either with an answer or with a pause. */
export async function streamTurn(
  sessionId: string,
  question: string,
  opts: {
    topK?: number;
    multiQuery?: boolean;
    clarify?: boolean;
    react?: boolean;
    modelProfile?: string | null;
  } = {},
  onProgress?: (node: string, detail: string) => void,
  onActivity?: (activity: Activity) => void,
): Promise<TurnOutcome> {
  const res = await authedJson(`/sessions/${sessionId}/stream`, "POST", {
    question,
    top_k: opts.topK ?? null,
    multi_query: opts.multiQuery ?? null,
    // null = use the server's AGENT_CLARIFY default rather than asserting a
    // value the UI has no opinion about.
    clarify: opts.clarify ?? null,
    react: opts.react ?? null,
    // Dev-only; the server ignores it unless APP_ENV=dev.
    model_profile: opts.modelProfile ?? null,
  });
  return readTurnStream(res, onProgress, onActivity);
}

/**
 * Answer the clarifying question and stream the rest of the turn.
 *
 * `question` is echoed back because the resumed turn is persisted against it —
 * the graph holds it too, but sending it keeps the endpoint self-contained.
 */
export async function resumeTurn(
  sessionId: string,
  threadId: string,
  question: string,
  decision: ClarifyDecision,
  onProgress?: (node: string, detail: string) => void,
  onActivity?: (activity: Activity) => void,
): Promise<TurnOutcome> {
  const res = await authedJson(`/sessions/${sessionId}/resume/stream`, "POST", {
    thread_id: threadId,
    question,
    action: decision.action,
    answer: decision.action === "answer" ? decision.answer : "",
  });
  return readTurnStream(res, onProgress, onActivity);
}

/**
 * Thumbs up/down on one answer.
 *
 * Identified by MESSAGE id -- the server looks up the trace id from the stored
 * message rather than trusting one from the client, since a client-supplied
 * trace id would let anyone score any trace.
 *
 * Fire-and-forget by design: feedback failing must not interrupt reading the
 * answer, and there is nothing useful to tell the user if it does.
 */
export async function sendFeedback(
  sessionId: string,
  messageId: string,
  helpful: boolean,
): Promise<void> {
  await authedJson(`/sessions/${sessionId}/feedback`, "POST", {
    message_id: messageId,
    helpful,
  });
}

export async function getStats(): Promise<Stats> {
  const res = await authedFetch("/stats", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

// ------------------------------------------------------------------- account

export type WhoAmI = {
  authenticated: boolean;
  auth_enabled: boolean;
  user_id: string | null;
  email: string | null;
  n_documents: number;
  n_sessions: number;
  unclaimed_documents: number;
  unclaimed_sessions: number;
};

export async function whoAmI(): Promise<WhoAmI> {
  const res = await authedFetch("/auth/me", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/** Take ownership of rows created before auth was switched on. Idempotent. */
export async function claimUnowned(): Promise<{
  documents: number;
  sessions: number;
  vectors: number;
}> {
  const res = await authedJson("/auth/claim", "POST");
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

// --- Model Lab: an open-weights model over an OpenAI-compatible API ---------

export type PlaygroundStatus = {
  enabled: boolean;
  default_model: string;
  /** Shown in the UI: this is the one line that changes to self-host. */
  base_url: string;
};

export type GroqModel = {
  id: string;
  owned_by: string | null;
  context_window: number | null;
};

export type Completion = {
  text: string;
  finish_reason: string | null;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  elapsed_ms: number;
  tokens_per_second: number | null;
};

export async function playgroundStatus(): Promise<PlaygroundStatus> {
  const res = await authedFetch("/playground/status");
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function playgroundModels(): Promise<GroqModel[]> {
  const res = await authedFetch("/playground/models");
  if (!res.ok) throw new Error(await detail(res));
  return (await res.json()).models;
}

export async function playgroundComplete(body: {
  prompt: string;
  model?: string;
  system?: string;
  temperature?: number;
  max_tokens?: number;
}): Promise<Completion> {
  const res = await authedFetch("/playground/complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export type Transcription = {
  text: string;
  language: string | null;
  duration_seconds: number;
  elapsed_ms: number;
  realtime_factor: number | null;
  /** Whisper's own confidence. Shown, never trusted: on pure silence it
   *  returns "Thank you." with no_speech_prob 0.000. */
  no_speech_prob: number | null;
  avg_logprob: number | null;
  model: string;
};

/**
 * Multipart, so no Content-Type header is set by hand — the browser has to
 * add its own `boundary` and setting the type manually omits it, which the
 * server then cannot parse.
 */
export async function transcribeAudio(
  blob: Blob,
  opts: {
    model?: string;
    language?: string;
    filename?: string;
    /** Tail of the transcript so far, to keep spelling consistent across a
     *  cut Whisper cannot see across. Ignored by NeMo, which has no
     *  equivalent decoder-context parameter. */
    prompt?: string;
    /** "groq" (Whisper) or "nvidia" (NeMo / Parakeet). */
    provider?: string;
  } = {},
): Promise<Transcription> {
  const form = new FormData();
  form.append("file", blob, opts.filename ?? "audio.webm");
  if (opts.model) form.append("model", opts.model);
  if (opts.language) form.append("language", opts.language);
  if (opts.prompt) form.append("prompt", opts.prompt);
  if (opts.provider) form.append("provider", opts.provider);

  const res = await authedFetch("/playground/transcribe", {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export type NvidiaStatus = {
  enabled: boolean;
  base_url: string;
  function_id: string;
};

export async function nvidiaStatus(): Promise<NvidiaStatus> {
  const res = await authedFetch("/playground/nvidia/status");
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export type NvidiaFunction = {
  id: string;
  name: string;
  status: string | null;
  protocol: string | null;
  speech: boolean;
};

export async function nvidiaFunctions(): Promise<NvidiaFunction[]> {
  const res = await authedFetch("/playground/nvidia/functions");
  if (!res.ok) throw new Error(await detail(res));
  return (await res.json()).functions;
}

// --- Corpus atlas: the embedding space as geometry -------------------------

export type AtlasPoint = {
  chunk_id: string;
  document_id: string;
  filename: string;
  heading: string | null;
  chunk_index: number;
  n_chars: number;
  preview: string;
  /** The whole passage, for the expanded reading card. */
  text: string;
  x: number;
  y: number;
  z: number;
  /** Most similar OTHER chunk. Precomputed so the UI need not scan the matrix. */
  nearest: {
    filename: string;
    heading: string | null;
    chunk_index: number;
    score: number;
  } | null;
};

export type Atlas = {
  points: AtlasPoint[];
  /** Row-major cosine similarity, same order as `points`. */
  similarity: number[][];
  /** Share of variance each of the three axes accounts for. */
  explained_variance: number[];
  n_documents: number;
  truncated: boolean;
};

export async function getAtlas(): Promise<Atlas> {
  const res = await authedFetch("/corpus/atlas");
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/** One retrieved chunk, as a line from the query to a point in `points`. */
export type AtlasRay = {
  /** Index into `points` — not a chunk id, so the UI never re-derives a
   *  position and risks a second, disagreeing answer. */
  index: number;
  rank: number;
  score: number;
  filename: string;
  heading: string | null;
  chunk_index: number;
};

export type QueryRay = Atlas & {
  question: string;
  /** Where the question itself landed, in the corpus's own projection. */
  query: { x: number; y: number; z: number } | null;
  rays: AtlasRay[];
  /** Nearest points that were NOT retrieved — the recall misses, which appear
   *  in no log: a near miss and a distant miss are both simply absent. */
  near_misses: { index: number; distance: number }[];
  /** Stored citations with no point to attach to (a web result, or a chunk
   *  deleted since). Surfaced so the UI can say why it drew fewer lines. */
  dropped: number;
};

export async function getQueryRay(messageId: string): Promise<QueryRay> {
  const res = await authedFetch(`/corpus/atlas/ray/${messageId}`);
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}


// ---------------------------------------------------------------------------
// Parley — the audio-only app
// ---------------------------------------------------------------------------

export type VoiceOption = {
  id: string;
  label: string;
  /** How it actually sounds. The names alone are unreadable as a menu. */
  character: string;
};

export type VoiceStatus = {
  enabled: boolean;
  stt_model: string;
  /** Tried in order — free-tier TTS quota is per model and small, so the
   *  first is not always the one that answers. */
  tts_models: string[];
  voices: VoiceOption[];
  default_voice: string;
  /** Shown because "did not search the web" and "cannot search the web" are
   *  indistinguishable from a spoken answer. */
  web_search: boolean;
};

export type VoiceSource = {
  label: string;
  kind: "document" | "web";
  url: string | null;
  /** How many passages from this one source. Grouped per document, because to
   *  a listener eight chunks of one handbook is one source. */
  passages: number;
};

export type VoiceTurn = {
  transcript: string;
  /** The written answer, markup and all. */
  answer: string;
  /** What was actually SAID — markup stripped, length capped. Differs from
   *  `answer`, so the screen shows the words being spoken rather than a
   *  different text that merely resembles them. */
  spoken: string;
  /** base64 WAV. Inline rather than a second request: one turn, one trip. */
  audio: string;
  mime: string;
  sample_rate: number;
  sources: VoiceSource[];
  /** Nothing intelligible was heard. Answered with speech, not an error. */
  heard_nothing: boolean;
  truncated: boolean;
  iterations: number;
  partial: boolean;
};

export async function getVoiceStatus(): Promise<VoiceStatus> {
  const res = await authedFetch("/voice/status", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function askByVoice(
  wav: Blob,
  voiceName: string,
): Promise<VoiceTurn> {
  const form = new FormData();
  form.append("audio", wav, "question.wav");
  form.append("voice_name", voiceName);
  const res = await authedFetch("/voice/ask", { method: "POST", body: form });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/** base64 WAV -> a URL an <audio> element can play. */
export function audioUrl(turn: VoiceTurn): string {
  const raw = atob(turn.audio);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: turn.mime }));
}


// ---------------------------------------------------------------------------
// Parley — the live, native-audio session
//
// The cascade endpoints above (`/voice/ask`) are kept: they are a working
// reference implementation of the other architecture, and the difference
// between the two is the most instructive thing in this project. The app uses
// the live path.
// ---------------------------------------------------------------------------

export type LiveStatus = {
  enabled: boolean;
  /** The native audio model — audio in, audio out, no transcript between. */
  model: string;
  voices: VoiceOption[];
  default_voice: string;
  web_search: boolean;
  input_rate: number;
  output_rate: number;
};

export async function getLiveStatus(): Promise<LiveStatus> {
  const res = await authedFetch("/live/status", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}


/**
 * Which spoken app a conversation belongs to.
 *
 * Speak and Interview run the same pipeline against a different system
 * prompt, and this is the whole of the difference on the client: it picks the
 * prompt server-side, and separates the two lists of conversations.
 */
export type Mode = "speak" | "interview" | "howler";

/** A spoken conversation, as a row in the "continue" list. */
export type ParleyConversation = {
  id: string;
  title: string;
  turns: number;
  updated_at: string;
  /** Interview only: every required field is filled. */
  complete: boolean;
  /** Whether the live CONTEXT can be restored, not merely the transcript read.
   *  Without a handle it can be reread but not continued. */
  resumable: boolean;
};

/** Where a spoken answer came from. Grouped per document, not per passage —
 *  to a listener, eight chunks of one handbook is one source. */
export type ParleySource = {
  label: string;
  kind: "document" | "web";
  url: string | null;
};

export type ParleyTurn = {
  question: string;
  answer: string;
  sources: ParleySource[];
  /** Tool NAMES only. The hit counts are live-only telemetry and are not
   *  worth a column. */
  tools: string[];
};

export async function getParleyConversations(
  mode: Mode = "speak",
): Promise<ParleyConversation[]> {
  const res = await authedFetch(`/live/conversations?mode=${mode}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/** One field the interview is trying to fill. */
export type ProfileField = {
  name: string;
  required: boolean;
  kind: string;
};

/**
 * What has been gathered.
 *
 * Most values are a string, a number or a list. `notes` is the exception: a
 * map of field name to observations about HOW that answer was given, plus
 * `general` for anything about the person rather than one answer. It is what
 * makes the profile more than a spreadsheet, and it is shaped differently
 * because it hangs off the other fields rather than sitting beside them.
 */
export type ProfileNotes = Record<string, string[]>;
export type Profile = Record<
  string,
  string | number | string[] | ProfileNotes | undefined
>;

/** A field Howler generated from a brief, rather than one written in code. */
export type BlueprintField = {
  name: string;
  label: string;
  description: string;
  type: "STRING" | "NUMBER" | "ARRAY";
  required: boolean;
};

/** What a brief WOULD produce, without creating anything.
 *  Separate from creating, so a brief can be adjusted and re-read — otherwise
 *  every attempt leaves an abandoned conversation in the drawer. */
export async function previewHowl(brief: string): Promise<BlueprintField[]> {
  const res = await authedJson("/live/howler/preview", "POST", { brief });
  if (!res.ok) throw new Error(await detail(res));
  return (await res.json()).fields;
}

export async function createHowl(
  brief: string,
  participant: string,
): Promise<{ id: string; fields: BlueprintField[] }> {
  const res = await authedJson("/live/howler", "POST", { brief, participant });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function getProfileFields(): Promise<ProfileField[]> {
  const res = await authedFetch("/live/profile-fields", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function getParleyConversation(
  id: string,
): Promise<{
  id: string;
  title: string;
  resumable: boolean;
  turns: ParleyTurn[];
  profile: Profile;
  missing: string[];
  complete: boolean;
  /** Howler only: the schema this conversation was given. Empty elsewhere. */
  fields: BlueprintField[];
  brief: string;
  participant: string;
  /** Howler only: the project this conversation belongs to, so reading a
   *  result has a way back to it. Null for a session with no project. */
  project_id: string | null;
}> {
  const res = await authedFetch(`/live/conversations/${id}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function renameParleyConversation(
  id: string,
  title: string,
): Promise<void> {
  const res = await authedJson(`/live/conversations/${id}`, "PATCH", { title });
  if (!res.ok) throw new Error(await detail(res));
}

export async function deleteParleyConversation(id: string): Promise<void> {
  const res = await authedFetch(`/live/conversations/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await detail(res));
}

/* ------------------------------------------------------------------ Howler
 *
 * A PROJECT, not a conversation. One brief is worth interviewing several
 * people against, so the brief, the data points it produced, the links sent
 * out and the results they returned all outlive any single session.
 *
 * The designer chat is a plain request/response — no streaming, no socket.
 * Each turn returns the reply AND the revised project together, because the
 * panel beside the chat has to update in the same round trip that answers
 * you: watching the data points change as you talk is the entire reason this
 * is a conversation rather than a form.
 */

export type DesignMessage = { role: "user" | "assistant"; content: string };

export type HowlerProject = {
  id: string;
  title: string;
  brief: string;
  participant: string;
  fields: BlueprintField[];
  /** Spellings for the microphone, not data to gather. Passed to the live
   *  session as `custom_vocabulary` and listed in the interviewer's prompt. */
  vocabulary: string[];
  design: DesignMessage[];
  created_at: string | null;
  /** Only on the list endpoint. */
  invites?: number;
};

/** What came back through one link. Null until somebody opens it. */
export type InviteResult = {
  session_id: string;
  title: string;
  turns: number;
  summary: string;
  complete: boolean;
  missing: string[];
  /** What was gathered. */
  profile: Profile;
  /**
   * How the whole conversation SOUNDED, from the only thing that heard it.
   *
   * Observations about delivery, not conclusions about the person — the
   * interviewer is told in as many words to record "took time over each
   * answer" and never "lacks confidence". Null when the interview ended
   * without one, which an interview cut short usually does.
   */
  affect: { demeanour: string; moments: string[] } | null;
  /** MEASURED from the audio, as opposed to `affect` which is what the
   *  interviewer heard. Null until a recording exists and has been analysed. */
  voice: {
    model: string;
    baseline: Record<string, number>;
    turns: { turn: number; arousal: number; dominance: number; valence: number }[];
    moments: { turn: number; dimension: string; delta: number; direction: string }[];
    caveat: string;
  } | null;
  /** The background job, so the tab can say "queued" rather than showing
   *  nothing and looking broken. */
  analysis: {
    status: "queued" | "running" | "done" | "failed";
    error: string;
    result: Record<string, unknown>;
  } | null;
  /** The schema THIS conversation was given, which is not necessarily the
   *  project's current one — a link opened last week gathered last week's
   *  data points, and rendering it against today's would invent empty rows
   *  for fields nobody was asked about. */
  fields: BlueprintField[];
};

export type HowlerInvite = {
  id: string;
  token: string;
  label: string;
  participant: string;
  opens: number;
  last_opened_at: string | null;
  revoked: boolean;
  /** Derived server-side from the conversation, never stored. */
  status: "unopened" | "in_progress" | "complete" | "revoked";
  result: InviteResult | null;
};

export type DesignTurn = {
  reply: string;
  ready: boolean;
  project: HowlerProject;
  /**
   * The link, when one was made this turn.
   *
   * Always present from `synthesiseHowlerProject`. From `designHowlerProject`
   * it is null MOST turns and set on the one where the designer decided the
   * data points were finished and settled them itself — it does not need
   * permission, and a button to confirm what it has just said is done is the
   * form this replaced.
   */
  invite?: HowlerInvite | null;
};

export async function listHowlerProjects(): Promise<HowlerProject[]> {
  const res = await authedFetch("/howler/projects", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function getHowlerProject(id: string): Promise<HowlerProject> {
  const res = await authedFetch(`/howler/projects/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/** A brief is optional — the designer conversation is where one gets written. */
export async function createHowlerProject(
  brief = "",
): Promise<HowlerProject> {
  const res = await authedJson("/howler/projects", "POST", { brief });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function deleteHowlerProject(id: string): Promise<void> {
  const res = await authedFetch(`/howler/projects/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await detail(res));
}

export async function designHowlerProject(
  id: string,
  message: string,
): Promise<DesignTurn> {
  const res = await authedJson(`/howler/projects/${id}/design`, "POST", {
    message,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/** "That is enough talking." Commits to a schema and returns a link with it. */
export async function synthesiseHowlerProject(
  id: string,
  message?: string,
): Promise<DesignTurn> {
  const res = await authedJson(`/howler/projects/${id}/synthesise`, "POST", {
    message,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function listHowlerInvites(id: string): Promise<HowlerInvite[]> {
  const res = await authedFetch(`/howler/projects/${id}/invites`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function createHowlerInvite(
  id: string,
  label: string,
  participant = "",
): Promise<HowlerInvite> {
  const res = await authedJson(`/howler/projects/${id}/invites`, "POST", {
    label,
    participant,
  });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function revokeHowlerInvite(inviteId: string): Promise<void> {
  const res = await authedFetch(`/howler/invites/${inviteId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await detail(res));
}
