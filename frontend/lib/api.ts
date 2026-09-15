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

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData();
  form.append("file", file);
  // No Content-Type here on purpose: the browser must set it, because it has
  // to append the multipart boundary.
  const res = await authedFetch("/documents", { method: "POST", body: form });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
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

export async function listSessions(): Promise<ChatSession[]> {
  const res = await authedFetch("/sessions", { cache: "no-store" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
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
  /** "search" | "search_done" | "rerank" */
  kind: string;
  /** "documents" | "web" */
  source?: string;
  query?: string;
  n?: number;
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
