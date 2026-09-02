"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type ChatSession,
  type Chunk,
  type Document,
  getChunk,
  listDocuments,
  listSessions,
} from "@/lib/api";
import { authEnabled, getSupabase } from "@/lib/supabase";

/**
 * One place that owns sessions and documents.
 *
 * The sidebar, the chat page and the library page all need this data. Without
 * a shared owner each would poll on its own, so the sidebar's ingest progress
 * and the library's list could disagree.
 */
type AppData = {
  sessions: ChatSession[];
  documents: Document[];
  loading: boolean;
  error: string | null;
  refreshSessions: () => Promise<void>;
  refreshDocuments: () => Promise<void>;
  /** Documents finished ingesting and searchable. */
  readyDocuments: Document[];
  /** True while anything is still parsing/embedding. */
  ingesting: boolean;
  /** Citation drill-down: the chunk shown in the side panel, if any. */
  openChunk: Chunk | null;
  /** chunk_id currently being fetched, so the rail can show it is working. */
  chunkLoading: string | null;
  showChunk: (chunkId: string) => Promise<void>;
  closeChunk: () => void;
  /**
   * True while a page renders its own right-hand rail (the chat view). The
   * global slide-over then stands down, so citations open in the rail's Source
   * tab instead of a modal covering the conversation. Elsewhere -- the Lab --
   * the modal is still the right answer.
   */
  railActive: boolean;
  setRailActive: (active: boolean) => void;
  /**
   * Rail collapse lives here, not in the chat page, for two reasons: it must
   * survive switching sessions, and holding it in a per-session component made
   * it reset to false and re-animate on every switch.
   */
  /** Signed-in account, or null. Always null when auth is disabled. */
  account: { id: string; email: string | null } | null;
  /** False until the initial session lookup finishes, so the auth gate does
   *  not redirect a signed-in user during the first render. */
  authReady: boolean;
  authEnabled: boolean;
  signOut: () => Promise<void>;
  railCollapsed: boolean;
  setRailCollapsed: (v: boolean) => void;
  /**
   * False until the persisted collapse state has been read. Transitions stay
   * off until then, so the stored value applies without an opening animation.
   */
  railReady: boolean;
};

const RAIL_KEY = "rd:rail-collapsed";

const Ctx = createContext<AppData | null>(null);

export function useApp(): AppData {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useApp must be used inside <Providers>");
  return ctx;
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openChunk, setOpenChunk] = useState<Chunk | null>(null);
  const [chunkLoading, setChunkLoading] = useState<string | null>(null);
  /**
   * Chunk text is immutable after ingestion, so this never needs invalidating.
   * A ref, not state: writing to it must not re-render, and every read is
   * followed by a setState that does.
   */
  const chunkCache = useRef(new Map<string, Chunk>());
  const [railActive, setRailActive] = useState(false);
  const [railCollapsed, setRailCollapsedState] = useState(false);
  const [railReady, setRailReady] = useState(false);
  const [account, setAccount] = useState<AppData["account"]>(null);
  // With auth disabled there is nothing to look up, so treat it as resolved.
  const [authReady, setAuthReady] = useState(!authEnabled);

  // Track the session for the lifetime of the app. onAuthStateChange also
  // fires on token refresh and on sign-out in another tab, so this stays
  // correct without polling.
  useEffect(() => {
    const supabase = getSupabase();
    if (!supabase) return;

    const apply = (user: { id: string; email?: string | null } | undefined) =>
      setAccount(user ? { id: user.id, email: user.email ?? null } : null);

    void supabase.auth.getSession().then(({ data }) => {
      apply(data.session?.user);
      setAuthReady(true);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) => {
      apply(session?.user);
      setAuthReady(true);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  const signOut = useCallback(async () => {
    const supabase = getSupabase();
    if (!supabase) return;
    await supabase.auth.signOut();
    // Clear cached data so the next account never sees the previous one's.
    setSessions([]);
    setDocuments([]);
  }, []);

  // Read once at app mount, not per session.
  useEffect(() => {
    try {
      setRailCollapsedState(localStorage.getItem(RAIL_KEY) === "1");
    } catch {
      // private browsing or blocked storage — the default is fine
    }
    setRailReady(true);
  }, []);

  const setRailCollapsed = useCallback((v: boolean) => {
    setRailCollapsedState(v);
    try {
      localStorage.setItem(RAIL_KEY, v ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load sessions");
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load documents");
    }
  }, []);

  // Load only once we know who we are. Fetching before the session resolves
  // would 401 on every endpoint; keying on account.id also reloads the data
  // when a different user signs in.
  const canFetch = authReady && (!authEnabled || account !== null);

  useEffect(() => {
    if (!canFetch) return;
    void (async () => {
      await Promise.all([refreshSessions(), refreshDocuments()]);
      setLoading(false);
    })();
  }, [canFetch, account?.id, refreshSessions, refreshDocuments]);

  const ingesting = documents.some(
    (d) => d.status !== "ready" && d.status !== "failed",
  );

  // Poll only while something is ingesting. Embedding is capped near 133
  // chunks/minute, so a large PDF sits in `embedding` for minutes -- but an
  // idle app must not hammer the API.
  useEffect(() => {
    if (!ingesting) return;
    const timer = setInterval(() => void refreshDocuments(), 2000);
    return () => clearInterval(timer);
  }, [ingesting, refreshDocuments]);

  /**
   * Citation drill-down.
   *
   * Every click used to be a full round-trip with no cache and no feedback, so
   * clicking between three citations meant three waits staring at the previous
   * passage -- and against a cold backend that is seconds each. Chunk text is
   * immutable once ingested, so it caches indefinitely and re-opening a passage
   * is now instant.
   *
   * `chunkLoading` exists because a cache miss still takes a moment, and
   * without it the rail showed the *previous* chunk during the fetch, which
   * reads as the click having selected the wrong source.
   */
  const showChunk = useCallback(async (chunkId: string) => {
    const cached = chunkCache.current.get(chunkId);
    if (cached) {
      setOpenChunk(cached);
      return;
    }
    setChunkLoading(chunkId);
    try {
      const chunk = await getChunk(chunkId);
      chunkCache.current.set(chunkId, chunk);
      setOpenChunk(chunk);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load chunk");
    } finally {
      setChunkLoading(null);
    }
  }, []);

  // Must be stable. As an inline arrow inside the memo below, its identity
  // changed whenever any context value changed -- which re-ran the chat page's
  // mount effect and cleared the chunk the instant a citation set it.
  const closeChunk = useCallback(() => setOpenChunk(null), []);

  const value = useMemo<AppData>(
    () => ({
      sessions,
      documents,
      loading,
      error,
      refreshSessions,
      refreshDocuments,
      readyDocuments: documents.filter((d) => d.status === "ready"),
      ingesting,
      openChunk,
      chunkLoading,
      showChunk,
      closeChunk,
      railActive,
      setRailActive,
      account,
      authReady,
      authEnabled,
      signOut,
      railCollapsed,
      setRailCollapsed,
      railReady,
    }),
    [
      sessions,
      documents,
      loading,
      error,
      refreshSessions,
      refreshDocuments,
      ingesting,
      openChunk,
      chunkLoading,
      showChunk,
      closeChunk,
      railActive,
      account,
      authReady,
      signOut,
      railCollapsed,
      setRailCollapsed,
      railReady,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
