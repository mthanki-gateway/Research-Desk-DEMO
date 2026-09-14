"use client";

import type { MessageSource } from "@/lib/api";

/**
 * Renders an assistant answer.
 *
 * Two jobs the previous `whitespace-pre-wrap` paragraph could not do:
 *
 *  1. **Structure.** Gemma returns markdown-ish prose -- blank-line separated
 *     paragraphs, `- ` bullets, `**bold**`, occasional `### ` headings. Held in
 *     one <p> those all collapsed into a wall of text, so a multi-part answer
 *     looked like a single run-on statement.
 *
 *  2. **Clickable citations.** The `[5]` markers were inert text. They are the
 *     most useful thing on screen -- the one affordance that takes a claim back
 *     to the passage it came from -- so each becomes a button that opens that
 *     chunk in the rail.
 *
 * Deliberately NOT a markdown library. The input is a constrained subset from a
 * known model, and `react-markdown` plus remark would add ~40KB to the bundle
 * to parse tables and footnotes that never appear. It also means no
 * `dangerouslySetInnerHTML` anywhere: every node below is a real React element,
 * so model output cannot inject markup.
 */

/** `[3]` or `[3, 5]` or `[3][5]` -- Gemma is inconsistent about grouping. */
const CITE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
const BOLD = /\*\*([^*]+)\*\*/g;

type Props = {
  content: string;
  sources: MessageSource[];
  activeChunkId: string | null;
  onCite: (chunkId: string) => void;
};

export function Answer({ content, sources, activeChunkId, onCite }: Props) {
  // n -> chunk_id, so a marker can be resolved to the passage it refers to.
  const byNumber = new Map(sources.map((s) => [s.n, s]));

  return (
    <div className="md-body-large md-prose">
      {blocks(content).map((block, i) => (
        <Block
          key={i}
          block={block}
          byNumber={byNumber}
          activeChunkId={activeChunkId}
          onCite={onCite}
        />
      ))}
    </div>
  );
}

/**
 * `p` and `h3` are separate members rather than `{kind: "p" | "h3"}`, so that
 * excluding both by early return leaves TypeScript with only the list member
 * and `block.items` narrows. With the kinds combined it stayed a union and
 * `items` was reported as missing.
 */
type BlockNode =
  | { kind: "p"; text: string }
  | { kind: "h3"; text: string }
  | { kind: "ul" | "ol"; items: string[] }
  | { kind: "table"; head: string[]; rows: string[][] };

/**
 * `| a | b |` -> ["a", "b"]. Null when the line is not a table row.
 *
 * The outer pipes are optional because models emit both forms, often in the
 * same answer.
 */
function cells(line: string): string[] | null {
  const trimmed = line.trim();
  if (!trimmed.includes("|")) return null;
  return trimmed
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => c.trim());
}

/** The `| --- | --- |` line that turns the row above it into a header. */
function isDivider(line: string): boolean {
  const parts = cells(line);
  return (
    !!parts && parts.length > 1 && parts.every((c) => /^:?-{2,}:?$/.test(c))
  );
}

/**
 * Put structure back on its own line when the model emitted it inline.
 *
 * The drafter replies under a JSON schema, so its answer is a string literal
 * and a line break has to be escaped as \n. A model that does not escape them
 * tends to emit NONE -- producing real markdown markers run together on a
 * single line: "the main pain points: - Customer concentration... - Hardware
 * supply chain...". Every marker the prompt asked for is present; only the
 * newlines are missing, so `blocks` below has nothing to split on and renders
 * the whole answer as one paragraph.
 *
 * The schema description now tells the model to escape them, which fixes new
 * answers. This fixes the ones ALREADY STORED, and stays as the net for any
 * model that slips -- the alternative, rewriting rows in the database, would
 * still leave the next unescaped answer broken.
 *
 * Deliberately narrow. It only breaks before a marker that is preceded by
 * whitespace and, for bullets, followed by one -- so a hyphenated phrase
 * ("well-known"), a negative number, and an inline "1." in a date survive
 * untouched. A document that already contains newlines is left alone
 * entirely, because then the model escaped correctly and any inline dash it
 * wrote is prose, not a list.
 */
const MARKER = /^(?:[-*]\s|\d{1,2}\.\s|#{1,6}\s|\|)/;

function reflow(content: string): string {
  if (content.includes("\n")) return content;
  const lines = content
    // "text - item" -> break before the bullet. Requires the space AFTER the
    // dash, which is what separates a list marker from a hyphen.
    .replace(/\s+([-*])\s+/g, "\n$1 ")
    // "text 1. item" -> ordered item. Anchored on a space before the digit so
    // "v1. " inside a word is not caught.
    .replace(/\s+(\d{1,2})\.\s+/g, "\n$1. ")
    // Headings and table rows, which are unambiguous wherever they appear.
    .replace(/\s+(#{1,6}\s+)/g, "\n$1")
    .replace(/\s+(\|)/g, "\n$1")
    .trim()
    .split("\n");

  // A blank line before the FIRST marker only.
  //
  // The lead-in sentence wants separating from the list it introduces, but a
  // blank line BETWEEN items is not cosmetic -- `blocks` treats it as a
  // paragraph break, which would close the list and open a new one per item,
  // renumbering every ordered item back to 1.
  const out: string[] = [];
  for (const line of lines) {
    const prev = out[out.length - 1];
    if (MARKER.test(line) && prev !== undefined && !MARKER.test(prev)) {
      out.push("");
    }
    out.push(line);
  }
  return out.join("\n");
}

/**
 * Group lines into blocks. Blank lines separate paragraphs; consecutive
 * bullet or numbered lines gather into one list rather than becoming a
 * paragraph each, which is what made lists render as stacked fragments.
 */
function blocks(content: string): BlockNode[] {
  const out: BlockNode[] = [];
  // \r\n first: an answer that round-tripped through a Windows client would
  // otherwise leave a stray \r that defeats the blank-line test.
  const lines = reflow(content.replace(/\r\n/g, "\n")).split("\n");
  let para: string[] = [];
  let list: { kind: "ul" | "ol"; items: string[] } | null = null;

  const flushPara = () => {
    if (para.length) {
      out.push({ kind: "p", text: para.join(" ") });
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      out.push(list);
      list = null;
    }
  };

  // An index loop, not for-of, because a table needs LOOKAHEAD: a row of pipes
  // is only a header once the NEXT line is a divider, and consuming its body
  // rows means advancing past them. (An earlier version used
  // `lines.indexOf(raw)` to fake this, which silently finds the FIRST matching
  // line -- so two identical rows in one answer sent it back to the start.)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) {
      flushPara();
      flushList();
      continue;
    }

    const row = cells(line);
    if (row && row.length > 1 && isDivider(lines[i + 1] ?? "")) {
      flushPara();
      flushList();
      const rows: string[][] = [];
      let j = i + 2; // skip the header and its divider
      for (; j < lines.length; j++) {
        const next = cells(lines[j]);
        if (!next || next.length < 2) break;
        rows.push(next);
      }
      out.push({ kind: "table", head: row, rows });
      i = j - 1; // the outer loop increments
      continue;
    }

    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    const numbered = /^\d+[.)]\s+(.*)$/.exec(line);
    const heading = /^#{1,6}\s+(.*)$/.exec(line);

    if (heading) {
      flushPara();
      flushList();
      out.push({ kind: "h3", text: heading[1] });
    } else if (bullet) {
      flushPara();
      if (list?.kind !== "ul") {
        flushList();
        list = { kind: "ul", items: [] };
      }
      list.items.push(bullet[1]);
    } else if (numbered) {
      flushPara();
      if (list?.kind !== "ol") {
        flushList();
        list = { kind: "ol", items: [] };
      }
      list.items.push(numbered[1]);
    } else if (list) {
      // A wrapped continuation of the current bullet, not a new paragraph.
      list.items[list.items.length - 1] += ` ${line}`;
    } else {
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return out;
}

function Block({
  block,
  byNumber,
  activeChunkId,
  onCite,
}: {
  block: BlockNode;
  byNumber: Map<number, MessageSource>;
  activeChunkId: string | null;
  onCite: (chunkId: string) => void;
}) {
  const render = (text: string) => (
    <Inline
      text={text}
      byNumber={byNumber}
      activeChunkId={activeChunkId}
      onCite={onCite}
    />
  );

  if (block.kind === "h3") return <h3>{render(block.text)}</h3>;
  if (block.kind === "p") return <p>{render(block.text)}</p>;

  if (block.kind === "table") {
    return (
      // Wrapped in its own scroll container. A wide table must never make the
      // whole message scroll sideways -- the rest of the answer would move with
      // it, which is far more disorienting than scrolling the table alone.
      <div className="md-scroll md-table-wrap">
        <table className="md-table">
          <thead>
            <tr>
              {block.head.map((cell, i) => (
                <th key={i}>{render(cell)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, r) => (
              <tr key={r}>
                {row.map((cell, c) => (
                  <td key={c}>{render(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const List = block.kind === "ul" ? "ul" : "ol";
  return (
    <List>
      {block.items.map((item, i) => (
        <li key={i}>{render(item)}</li>
      ))}
    </List>
  );
}

/**
 * Inline pass: turns `**bold**` into <strong> and every `[n]` into a button.
 *
 * A marker whose number has no matching source stays plain text. That happens
 * when the model cites a passage it was not actually given, and showing it as
 * dead text rather than a broken button is the honest rendering -- a citation
 * that leads nowhere should not look clickable.
 */
function Inline({
  text,
  byNumber,
  activeChunkId,
  onCite,
}: {
  text: string;
  byNumber: Map<number, MessageSource>;
  activeChunkId: string | null;
  onCite: (chunkId: string) => void;
}) {
  const out: React.ReactNode[] = [];
  let last = 0;
  let key = 0;

  for (const m of text.matchAll(CITE)) {
    const at = m.index ?? 0;
    if (at > last) out.push(...bold(text.slice(last, at), key++));

    const numbers = m[1].split(",").map((n) => Number(n.trim()));
    const resolved = numbers.filter((n) => byNumber.has(n));

    if (resolved.length === 0) {
      out.push(<span key={`c${key++}`}>{m[0]}</span>);
    } else {
      for (const n of resolved) {
        const source = byNumber.get(n)!;

        // A web source opens its URL in a new tab; a document source opens
        // the chunk in the rail. Rendered as a real <a> rather than a
        // button calling window.open, so middle-click, ctrl-click and
        // “copy link address” all behave — the same reason the nav uses
        // <Link> instead of router.push.
        if (source.source === "web" && source.url) {
          out.push(
            <a
              key={`c${key++}`}
              className="md-cite md-cite-web"
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              title={`${source.filename} — ${source.url}`}
            >
              {n}
            </a>,
          );
          continue;
        }

        out.push(
          <button
            key={`c${key++}`}
            type="button"
            className="md-cite"
            aria-current={source.chunk_id === activeChunkId}
            onClick={() => onCite(source.chunk_id)}
            title={
              source.heading
                ? `${source.filename} — ${source.heading.replace(/^#+\s*/, "")}`
                : source.filename
            }
          >
            {n}
          </button>,
        );
      }
    }
    last = at + m[0].length;
  }

  if (last < text.length) out.push(...bold(text.slice(last), key++));
  return <>{out}</>;
}

/** Splits a plain run on `**bold**`. Returns nodes, never HTML. */
function bold(text: string, seed: number): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const m of text.matchAll(BOLD)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    out.push(<strong key={`b${seed}-${i++}`}>{m[1]}</strong>);
    last = at + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out.length ? out : [text];
}
