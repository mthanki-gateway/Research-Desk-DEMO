"use client";

import { useRef, useState } from "react";
import {
  type Chunk,
  type Document,
  deleteDocument,
  getDocumentChunks,
  uploadDocument,
} from "@/lib/api";
import { useApp } from "../providers";
import { Button, ConfirmButton, LinearProgress, Ripplable } from "../md";
import {
  IconChevron,
  IconDocument,
  IconSpinner,
  IconTrash,
  IconUpload,
} from "../icons";

export default function Library() {
  const { documents, loading, refreshDocuments } = useApp();
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFiles(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    try {
      // Sequential: parallel uploads queue behind the same token budget anyway,
      // and failures get attributed to the right file.
      for (const file of Array.from(files)) await uploadDocument(file);
      await refreshDocuments();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  const totalChunks = documents.reduce((n, d) => n + d.n_chunks, 0);

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-6 py-9">
      <header className="flex items-end justify-between gap-6">
        <div>
          <h1 className="md-headline-small">Library</h1>
          <p
            className="md-body-medium mt-1"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            Documents available to every conversation.
          </p>
        </div>
        {documents.length > 0 && (
          <div className="flex shrink-0 gap-2">
            <span className="md-badge">{documents.length} docs</span>
            <span className="md-badge">{totalChunks} chunks</span>
          </div>
        )}
      </header>

      {/* No M3 dropzone exists, so this is an outlined card with a tonal
          action inside it. */}
      <Ripplable
        as="div"
        onDragOver={(e: React.DragEvent) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e: React.DragEvent) => {
          e.preventDefault();
          setDragging(false);
          void handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        className="md-card cursor-pointer px-6 py-10 text-center"
        style={{
          border: `2px dashed ${
            dragging ? "var(--md-primary)" : "var(--md-outline-variant)"
          }`,
          background: dragging
            ? "var(--md-primary-container)"
            : "var(--md-surface)",
        }}
      >
        <span
          className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-[var(--md-shape-full)]"
          style={{
            background: "var(--md-primary-container)",
            color: "var(--md-on-primary-container)",
          }}
        >
          {busy ? (
            <IconSpinner className="h-6 w-6" />
          ) : (
            <IconUpload className="h-6 w-6" />
          )}
        </span>
        <p className="md-title-small">
          {busy ? "Uploading" : "Drop files here, or click to browse"}
        </p>
        <p
          className="md-body-small mt-1"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          PDF, Markdown or plain text · up to 20MB · scanned PDFs need OCR
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md"
          multiple
          hidden
          onChange={(e) => void handleFiles(e.target.files)}
        />
      </Ripplable>

      {error && (
        <p
          className="md-body-medium rounded-[var(--md-shape-md)] px-4 py-3"
          style={{
            background: "var(--md-error-container)",
            color: "var(--md-on-error-container)",
          }}
        >
          {error}
        </p>
      )}

      {loading ? (
        <div className="space-y-3" aria-hidden>
          {[0, 1].map((i) => (
            <div key={i} className="md-skeleton h-[88px]" />
          ))}
        </div>
      ) : documents.length === 0 ? (
        <p
          className="md-body-medium"
          style={{ color: "var(--md-on-surface-variant)" }}
        >
          Nothing indexed yet.
        </p>
      ) : (
        <ul className="space-y-3">
          {documents.map((doc) => (
            <DocumentCard
              key={doc.id}
              doc={doc}
              expanded={expanded === doc.id}
              onToggle={() =>
                setExpanded((cur) => (cur === doc.id ? null : doc.id))
              }
              onDeleted={async () => {
                if (expanded === doc.id) setExpanded(null);
                await refreshDocuments();
              }}
              onError={setError}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function fileKind(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return ext === "md" ? "MD" : ext === "pdf" ? "PDF" : "TXT";
}

function StatusBadge({ doc }: { doc: Document }) {
  if (doc.status === "ready")
    return <span className="md-badge md-badge-primary">Indexed</span>;
  if (doc.status === "failed")
    return <span className="md-badge md-badge-error">Failed</span>;
  if (doc.status === "embedding") {
    const pct = doc.n_chunks
      ? Math.round((doc.n_embedded / doc.n_chunks) * 100)
      : 0;
    return <span className="md-badge md-badge-tertiary">Embedding {pct}%</span>;
  }
  return <span className="md-badge md-badge-tertiary">{doc.status}</span>;
}

function DocumentCard({
  doc,
  expanded,
  onToggle,
  onDeleted,
  onError,
}: {
  doc: Document;
  expanded: boolean;
  onToggle: () => void;
  onDeleted: () => Promise<void>;
  onError: (m: string) => void;
}) {
  const [chunks, setChunks] = useState<Chunk[] | null>(null);
  const [loadingChunks, setLoadingChunks] = useState(false);

  const ready = doc.status === "ready";

  async function toggle() {
    if (!ready) return;
    onToggle();
    if (!expanded && !chunks) {
      setLoadingChunks(true);
      try {
        setChunks(await getDocumentChunks(doc.id));
      } catch (e) {
        onError(e instanceof Error ? e.message : "Could not load chunks");
      } finally {
        setLoadingChunks(false);
      }
    }
  }

  async function remove() {
    try {
      await deleteDocument(doc.id);
      await onDeleted();
    } catch (err) {
      onError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  const pct = doc.n_chunks
    ? Math.round((doc.n_embedded / doc.n_chunks) * 100)
    : 0;

  return (
    <li className={`md-card ${ready ? "md-card-elevated" : "md-card-outlined"}`}>
      {/* The state layer and ripple cover the WHOLE row, with the card's own
          radius. Previously they sat on an inner box, so hover painted a
          mismatched grey rectangle inside the card. */}
      <Ripplable
        as="div"
        onClick={() => void toggle()}
        role={ready ? "button" : undefined}
        tabIndex={ready ? 0 : undefined}
        className={`flex items-center gap-4 p-4 ${ready ? "cursor-pointer" : ""}`}
        style={{
          borderRadius: expanded
            ? "var(--md-shape-md) var(--md-shape-md) 0 0"
            : "var(--md-shape-md)",
        }}
      >
        <div className="flex min-w-0 flex-1 items-center gap-4">
          <span
            className="md-label-small grid h-11 w-11 shrink-0 place-items-center rounded-[var(--md-shape-md)]"
            style={{
              background: "var(--md-primary-container)",
              color: "var(--md-on-primary-container)",
            }}
          >
            {fileKind(doc.filename)}
          </span>

          <span className="min-w-0 flex-1">
            <span className="flex items-center gap-1.5">
              {ready && (
                <IconChevron className="h-4 w-4 shrink-0" open={expanded} />
              )}
              <span className="md-title-small truncate">{doc.filename}</span>
            </span>

            <span
              className="md-body-small mt-1 flex flex-wrap items-center gap-x-3 gap-y-1"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              <StatusBadge doc={doc} />
              {ready && (
                <>
                  <span className="tabular-nums">{doc.n_chunks} chunks</span>
                  {doc.n_pages > 1 && (
                    <span className="tabular-nums">{doc.n_pages} pages</span>
                  )}
                  <span className="tabular-nums">
                    {Math.max(1, Math.round(doc.size_bytes / 1024))} KB
                  </span>
                </>
              )}
            </span>

            {doc.status === "embedding" && (
              <span className="mt-2 block">
                <LinearProgress value={pct} />
              </span>
            )}

            {doc.error && (
              <span
                className="md-body-small mt-1.5 block"
                style={{ color: "var(--md-error)" }}
              >
                {doc.error}
              </span>
            )}
          </span>
        </div>

        {/* Inside the ripple surface, so it aligns with the row — but its own
            clicks must not also toggle the card open. */}
        <span
          className="shrink-0"
          onClick={(e) => e.stopPropagation()}
          role="presentation"
        >
          <ConfirmButton
            label=""
            title="Delete this document?"
            body="Its chunks and vectors will be removed. Re-adding it later costs embedding quota."
            onConfirm={remove}
            icon={<IconTrash className="h-4 w-4" />}
          />
        </span>
      </Ripplable>

      {expanded && (
        <div
          className="border-t px-4 pb-4 pt-3"
          style={{ borderColor: "var(--md-outline-variant)" }}
        >
          {loadingChunks && (
            <p
              className="md-body-small flex items-center gap-2"
              style={{ color: "var(--md-on-surface-variant)" }}
            >
              <IconSpinner className="h-4 w-4" />
              Loading chunks
            </p>
          )}
          {chunks && (
            <>
              <p
                className="md-body-small mb-3 flex items-start gap-2"
                style={{ color: "var(--md-on-surface-variant)" }}
              >
                <IconDocument className="mt-0.5 h-4 w-4 shrink-0" />
                Split on markdown headings and prefixed with the heading — this
                is verbatim what the model receives.
              </p>
              <ol className="space-y-2">
                {chunks.map((c) => (
                  <li
                    key={c.id}
                    className="rounded-[var(--md-shape-md)] p-3"
                    style={{ background: "var(--md-surface-container-high)" }}
                  >
                    <div className="mb-1.5 flex items-baseline justify-between gap-3">
                      <span className="flex min-w-0 items-baseline gap-2">
                        <span
                          className="md-label-small shrink-0 rounded-[var(--md-shape-xs)] px-1.5 tabular-nums"
                          style={{
                            background: "var(--md-primary-container)",
                            color: "var(--md-on-primary-container)",
                          }}
                        >
                          {c.chunk_index}
                        </span>
                        <span className="md-body-medium truncate font-medium">
                          {c.heading
                            ? c.heading.replace(/^#+\s*/, "")
                            : "untitled section"}
                          {c.page !== null && (
                            <span
                              style={{ color: "var(--md-on-surface-variant)" }}
                            >
                              {" "}
                              · p.{c.page}
                            </span>
                          )}
                        </span>
                      </span>
                      <span
                        className="md-body-small shrink-0 tabular-nums"
                        style={{ color: "var(--md-on-surface-variant)" }}
                      >
                        {c.n_chars} ch
                      </span>
                    </div>
                    <p
                      className="md-body-small whitespace-pre-wrap"
                      style={{ color: "var(--md-on-surface-variant)" }}
                    >
                      {c.text}
                    </p>
                  </li>
                ))}
              </ol>
            </>
          )}
        </div>
      )}
    </li>
  );
}
