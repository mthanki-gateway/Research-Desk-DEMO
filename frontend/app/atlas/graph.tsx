"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import {
  buildGraph,
  getChunk,
  getGraph,
  getHydration,
  hydrateEntity,
  addTopic,
  exploreTopic,
  deleteTopic,
  type Chunk,
  type Hydration,
  type GraphEdge,
  type GraphNode,
  type KnowledgeGraph,
} from "@/lib/api";
import { Button } from "../md";
import { IconSpinner } from "../icons";
import { GROUND, INK, backdropTexture, type PlotTheme } from "./scatter";

/**
 * Who and what the corpus mentions, and what it SAYS connects them -- as a
 * space you fly through, built the same way as the scatter above it.
 *
 * The scatter shows passages that are near each other in meaning; this shows
 * claims. Every edge is a sentence in a document, and the reading panel opens
 * it -- an edge that cannot be checked would be the model's opinion dressed as
 * the corpus's.
 *
 * SAME CONTROLS AS THE SCATTER, deliberately: drag to orbit, shift- or
 * right-drag to pan, scroll to zoom, eased fly-to, Reset, Expand, Light/Dark.
 * Two views on one page that steer differently is two things to learn.
 *
 * LAYOUT IS COMPUTED ONCE, UP FRONT, then frozen. A 3D force simulation runs
 * a few hundred iterations before the first frame (tens of milliseconds at the
 * 250-node cap) and the result is scaled to fit the camera. A layout that keeps
 * settling on screen is the most distracting thing a graph can do, and a moving
 * node cannot be clicked.
 */

/** Entity-type colours, one set per ground -- same reasoning as PALETTES. */
const TYPE_COLOURS: Record<PlotTheme, Record<string, string>> = {
  dark: {
    person: "#ff8ae2",
    organization: "#4fd1ff",
    place: "#34e3a4",
    product: "#ffc247",
    concept: "#a78bfa",
    event: "#ffa06b",
    other: "#b8bfd6",
  },
  light: {
    person: "#b52f93",
    organization: "#0369a1",
    place: "#00855a",
    product: "#a86400",
    concept: "#5b3fd6",
    event: "#c2410c",
    other: "#5a6178",
  },
};
const TYPES = Object.keys(TYPE_COLOURS.dark);

/** Category colours for the Categories view, one set per ground. */
const CATEGORY_COLOURS: Record<PlotTheme, Record<string, string>> = {
  dark: {
    Technology: "#4fd1ff",
    Science: "#a78bfa",
    Business: "#ffc247",
    "People & Society": "#ff8ae2",
    Nature: "#34e3a4",
    Places: "#9be0b4",
    History: "#ffa06b",
    "Arts & Culture": "#f2a7c3",
    Health: "#ff6b81",
    Food: "#b6ef6a",
    Other: "#b8bfd6",
  },
  light: {
    Technology: "#0369a1",
    Science: "#5b3fd6",
    Business: "#a86400",
    "People & Society": "#b52f93",
    Nature: "#00855a",
    Places: "#3f7d4f",
    History: "#c2410c",
    "Arts & Culture": "#9d3b6b",
    Health: "#c62348",
    Food: "#4d7c0f",
    Other: "#5a6178",
  },
};

/**
 * The Categories view: the same graph re-drawn as category trees.
 *
 * Each category becomes a hub node; every term is linked to its category's
 * hub, and the term-to-term relations stay. So a category is a tree you can
 * click into -- Technology, and under it CUDA, the Toolkit, NVIDIA and how
 * they connect -- rather than a colour in a legend. Hubs and their links are
 * synthetic: they carry no passage, because no document said "CUDA is
 * Technology"; the extraction classified it.
 */
type Source = "all" | "documents" | "topics";

/**
 * Which trees to show. "documents": what the files say. "topics": the
 * concepts you added or explored, plus whatever they link to -- the learning
 * trees. "all": both, joined wherever a topic links to a document term.
 */
function bySource(data: KnowledgeGraph, source: Source): KnowledgeGraph {
  if (source === "all") return data;
  if (source === "documents") {
    const nodes = data.nodes.filter((n) => n.origin !== "topic");
    const ids = new Set(nodes.map((n) => n.id));
    return {
      ...data,
      nodes,
      edges: data.edges.filter((e) => e.origin !== "topic" && ids.has(e.source) && ids.has(e.target)),
    };
  }
  const edges = data.edges.filter((e) => e.origin === "topic");
  const ids = new Set<string>();
  for (const n of data.nodes) if (n.origin === "topic" || n.origin === "both") ids.add(n.id);
  for (const e of edges) {
    ids.add(e.source);
    ids.add(e.target);
  }
  return { ...data, nodes: data.nodes.filter((n) => ids.has(n.id)), edges };
}

function byCategory(data: KnowledgeGraph): KnowledgeGraph {
  const cats = new Map<string, number>();
  for (const n of data.nodes) cats.set(n.category || "Other", (cats.get(n.category || "Other") ?? 0) + 1);
  const hubs: GraphNode[] = [...cats].map(([c, count]) => ({
    id: `cat:${c}`,
    name: c,
    type: c,
    category: c,
    mentions: count,
    degree: count,
    documents: [],
  }));
  const nodes: GraphNode[] = [
    ...hubs,
    ...data.nodes.map((n) => ({ ...n, type: n.category || "Other" })),
  ];
  const edges: GraphEdge[] = [
    ...data.edges,
    ...data.nodes.map((n) => ({
      source: n.id,
      target: `cat:${n.category || "Other"}`,
      predicate: "is in",
      chunk_id: null,
      filename: "",
    })),
  ];
  return { ...data, nodes, edges };
}

/** World radius the layout is fitted to, and the default camera distance. */
const FIT = 3.2;
const HOME_RADIUS = 8.5;
/**
 * The scatter's point sizes and fog, exactly -- so a node and a chunk are the
 * same kind of object on screen. Sizes are before the 1/depth divide and the
 * clamp in the shader.
 */
const BASE_SIZE = 70;
const HOVER_SIZE = 115;
const SELECTED_SIZE = 165;
const FOG_NEAR = 7;
const FOG_FAR = 34;

/** Labels drawn at once. Past this they overlap into a wall of text. */
const MAX_LABELS = 28;

type Props = { theme: PlotTheme; onThemeChange: (t: PlotTheme) => void };

export default function GraphView({ theme, onThemeChange }: Props) {
  const [data, setData] = useState<KnowledgeGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [queuing, setQueuing] = useState(false);
  // Categories view: the graph redrawn as category trees (see byCategory).
  const [categories, setCategories] = useState(false);
  const [source, setSource] = useState<Source>("all");
  // A node to select once the graph reloads -- after adding or exploring, the
  // thing you just made is what you want to look at.
  const [focusKey, setFocusKey] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setData(await getGraph());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the graph");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Poll only while extraction is actually running, and slowly.
  useEffect(() => {
    if (!data?.pending) return;
    const t = setTimeout(() => void load(), 4000);
    return () => clearTimeout(t);
  }, [data, load]);

  async function backfill(rebuild = false) {
    setQueuing(true);
    try {
      await buildGraph(rebuild);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not queue extraction");
    } finally {
      setQueuing(false);
    }
  }

  if (error) {
    return <p className="md-body-medium" style={{ color: "var(--md-error)" }}>{error}</p>;
  }
  if (!data) return <div className="md-skeleton h-[30rem]" aria-hidden />;

  const missing = data.n_documents - data.n_documents_with_graph;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <p className="md-body-small flex-1" style={{ color: "var(--md-on-surface-variant)" }}>
          {data.nodes.length} entities, {data.edges.length} stated relations from{" "}
          {data.n_documents_with_graph} of {data.n_documents} documents.
          {data.truncated && ` Showing the ${data.nodes.length} best connected of ${data.n_entities}.`}
          {data.pending > 0 && ` Extracting ${data.pending} more…`}
        </p>
        {missing > 0 && data.pending === 0 && (
          <Button variant="outlined" onClick={() => void backfill()} disabled={queuing}>
            {queuing && <IconSpinner />}
            Build graph for {missing} document{missing === 1 ? "" : "s"}
          </Button>
        )}
        {missing === 0 && data.pending === 0 && data.n_documents > 0 && (
          <Button
            variant="text"
            onClick={() => void backfill(true)}
            disabled={queuing}
            title="Read every document again with the current extraction"
          >
            {queuing && <IconSpinner />}
            Rebuild graph
          </Button>
        )}
      </div>

      {data.nodes.length === 0 ? (
        <div
          className="flex h-[20rem] items-center justify-center rounded-[var(--md-shape-lg)]"
          style={{ background: GROUND[theme], color: INK[theme].faint }}
        >
          <p className="md-body-medium max-w-sm text-center">
            {data.pending > 0
              ? "Reading your documents for entities and relationships…"
              : "No graph yet. New uploads are read automatically; build one for existing documents above."}
          </p>
        </div>
      ) : (
        <Scene
          key={`${categories ? "categories" : "entities"}-${source}`}
          data={categories ? byCategory(bySource(data, source)) : bySource(data, source)}
          source={source}
          onSource={setSource}
          onReload={async (key?: string) => {
            if (key) setFocusKey(key);
            await load();
          }}
          focusKey={focusKey}
          onFocused={() => setFocusKey(null)}
          theme={theme}
          onThemeChange={onThemeChange}
          palette={categories ? CATEGORY_COLOURS : TYPE_COLOURS}
          categories={categories}
          onToggleCategories={() => setCategories((v) => !v)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

/** Positions for every node, fitted into a sphere of radius FIT. */
function layout(nodes: GraphNode[], edges: GraphEdge[]): Float32Array {
  const n = nodes.length;
  const index = new Map(nodes.map((nd, i) => [nd.id, i]));
  const links = edges
    .map((e) => [index.get(e.source), index.get(e.target)] as const)
    .filter((l): l is readonly [number, number] => l[0] !== undefined && l[1] !== undefined);

  // Deterministic start on a Fibonacci sphere, so the same graph settles into
  // the same shape on every visit.
  const p = new Float32Array(n * 3);
  const v = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / n;
    const r = Math.sqrt(1 - y * y);
    const a = i * 2.39996;
    p[i * 3] = Math.cos(a) * r * 10;
    p[i * 3 + 1] = y * 10;
    p[i * 3 + 2] = Math.sin(a) * r * 10;
  }

  const ITER = 320;
  for (let it = 0; it < ITER; it++) {
    const alpha = 1 - it / ITER;
    // Repulsion, every pair. O(n^2) is fine at 250 nodes.
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = p[i * 3] - p[j * 3];
        const dy = p[i * 3 + 1] - p[j * 3 + 1];
        const dz = p[i * 3 + 2] - p[j * 3 + 2];
        const d2 = dx * dx + dy * dy + dz * dz + 0.01;
        const f = (4 / d2) * alpha;
        v[i * 3] += dx * f; v[i * 3 + 1] += dy * f; v[i * 3 + 2] += dz * f;
        v[j * 3] -= dx * f; v[j * 3 + 1] -= dy * f; v[j * 3 + 2] -= dz * f;
      }
    }
    // Springs along edges.
    for (const [i, j] of links) {
      const dx = p[j * 3] - p[i * 3];
      const dy = p[j * 3 + 1] - p[i * 3 + 1];
      const dz = p[j * 3 + 2] - p[i * 3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      const f = ((d - 1.6) / d) * 0.08 * alpha;
      v[i * 3] += dx * f; v[i * 3 + 1] += dy * f; v[i * 3 + 2] += dz * f;
      v[j * 3] -= dx * f; v[j * 3 + 1] -= dy * f; v[j * 3 + 2] -= dz * f;
    }
    // Gravity to the centre, so disconnected islands do not drift away --
    // this is what kept the earlier 2D version from fitting on screen.
    for (let i = 0; i < n * 3; i++) {
      v[i] -= p[i] * 0.02 * alpha;
      p[i] += v[i];
      v[i] *= 0.55;
    }
  }

  // Fit: centre on the mean, scale so the 95th-percentile node sits at FIT.
  // A percentile rather than the max, so one far outlier cannot shrink the
  // whole graph to a dot.
  const c = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) c[k] += p[i * 3 + k] / n;
  const radii: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) p[i * 3 + k] -= c[k];
    radii.push(Math.hypot(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]));
  }
  radii.sort((a, b) => a - b);
  const r95 = radii[Math.floor(radii.length * 0.95)] || 1;
  for (let i = 0; i < n * 3; i++) p[i] *= FIT / r95;
  return p;
}

function Scene({
  data,
  theme,
  onThemeChange,
  palette,
  categories,
  onToggleCategories,
  source,
  onSource,
  onReload,
  focusKey,
  onFocused,
}: {
  source: Source;
  onSource: (s: Source) => void;
  onReload: (key?: string) => Promise<void>;
  focusKey: string | null;
  onFocused: () => void;
  data: KnowledgeGraph;
  theme: PlotTheme;
  onThemeChange: (t: PlotTheme) => void;
  palette: Record<PlotTheme, Record<string, string>>;
  categories: boolean;
  onToggleCategories: () => void;
}) {
  const host = useRef<HTMLDivElement | null>(null);
  // The passage behind a relation, shown INSIDE the graph's own panel. The
  // app's chunk side bar lives outside this element, so in fullscreen it
  // opened behind the graph where nobody could see it.
  const [passage, setPassage] = useState<Chunk | "loading" | null>(null);
  const [passageError, setPassageError] = useState<string | null>(null);
  // The selected entity's saved web profile. undefined = not looked up yet.
  const [hydration, setHydration] = useState<Hydration | null | undefined>(undefined);
  const [hydrating, setHydrating] = useState(false);
  const [hydrateError, setHydrateError] = useState<string | null>(null);
  const openPassage = useCallback(async (id: string) => {
    setPassage("loading");
    setPassageError(null);
    try {
      setPassage(await getChunk(id));
    } catch (e) {
      setPassage(null);
      setPassageError(e instanceof Error ? e.message : "Could not load the passage");
    }
  }, []);
  const labelLayer = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [newTopic, setNewTopic] = useState("");
  const [topicBusy, setTopicBusy] = useState<"add" | "explore" | "remove" | null>(null);
  const [topicError, setTopicError] = useState<string | null>(null);
  // The reading panel's width, dragged from its left edge and remembered.
  const [panelWidth, setPanelWidth] = useState(352);
  useEffect(() => {
    try {
      const w = Number(localStorage.getItem("graph.panelWidth"));
      if (w >= 260 && w <= 900) setPanelWidth(w);
    } catch {
      /* private mode: the default width is fine */
    }
  }, []);
  const startResize = (e: React.PointerEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = panelWidth;
    const max = Math.min(900, (host.current?.clientWidth ?? 1200) * 0.7);
    let last = startW;
    const move = (ev: PointerEvent) => {
      last = Math.max(260, Math.min(max, startW + (startX - ev.clientX)));
      setPanelWidth(last);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      try {
        localStorage.setItem("graph.panelWidth", String(Math.round(last)));
      } catch {
        /* not persisted; still applied for this visit */
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  const [typeFocus, setTypeFocus] = useState<string | null>(null);
  const [labels, setLabels] = useState(true);
  const [bridges, setBridges] = useState(false);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [nativeFull, setNativeFull] = useState(false);
  const big = expanded || nativeFull;

  const { nodes, edges } = data;
  const index = useMemo(() => new Map(nodes.map((n, i) => [n.id, i])), [nodes]);
  const positions = useMemo(() => layout(nodes, edges), [nodes, edges]);
  const neighbours = useMemo(() => {
    const out = nodes.map(() => new Set<number>());
    for (const e of edges) {
      const a = index.get(e.source), b = index.get(e.target);
      if (a === undefined || b === undefined) continue;
      out[a].add(b);
      out[b].add(a);
    }
    return out;
  }, [nodes, edges, index]);

  const api = useRef<{
    emphasise: (sel: number | null, hov: number | null, type: string | null, bridges: boolean) => void;
    flyToNode: (i: number) => void;
    flyToType: (t: string | null) => void;
    reset: () => void;
    setLabels: (on: boolean) => void;
  }>({ emphasise: () => {}, flyToNode: () => {}, flyToType: () => {}, reset: () => {}, setLabels: () => {} });

  const selectRef = useRef(setSelected);
  selectRef.current = setSelected;
  const labelsRef = useRef(labels);
  labelsRef.current = labels;
  const bigRef = useRef(big);
  bigRef.current = big;

  useEffect(() => {
    const el = host.current;
    const layer = labelLayer.current;
    if (!el || !layer || nodes.length === 0) return;
    const n = nodes.length;
    const colours = palette[theme];
    const ink = INK[theme];

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const view = renderer.domElement;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.insertBefore(view, el.firstChild);

    const scene = new THREE.Scene();
    const backdrop = backdropTexture(theme);
    scene.background = backdrop;
    const camera = new THREE.PerspectiveCamera(50, el.clientWidth / Math.max(1, el.clientHeight), 0.01, 120);

    // ---- nodes ----------------------------------------------------------
    const colourAttr = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const alphas = new Float32Array(n).fill(1);
    const base = new Float32Array(n);
    const c = new THREE.Color();
    nodes.forEach((nd, i) => {
      c.set(colours[nd.type] ?? colours.other ?? colours.Other);
      colourAttr.set([c.r, c.g, c.b], i * 3);
      // Category hubs are drawn larger: they are the roots of the trees.
      base[i] = nd.id.startsWith("cat:") ? BASE_SIZE * 2.1 : BASE_SIZE;
      sizes[i] = base[i];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colourAttr, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("alpha", new THREE.BufferAttribute(alphas, 1));

    // The scatter's shader, verbatim: soft light-source falloff, depth-scaled
    // size clamped to 6..90px, and fog towards the backdrop's edge colour.
    const ground = new THREE.Color(GROUND[theme]);
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      vertexColors: true,
      uniforms: {
        uFogColor: { value: ground },
        uFogNear: { value: FOG_NEAR },
        uFogFar: { value: FOG_FAR },
        uGlow: { value: theme === "dark" ? 1 : 0 },
        uScale: { value: renderer.getPixelRatio() },
      },
      vertexShader: [
        "attribute float size;",
        "attribute float alpha;",
        "varying vec3 vColor;",
        "varying float vAlpha;",
        "varying float vFog;",
        "uniform float uFogNear;",
        "uniform float uFogFar;",
        "uniform float uScale;",
        "void main() {",
        "  vColor = color;",
        "  vAlpha = alpha;",
        "  vec4 mv = modelViewMatrix * vec4(position, 1.0);",
        "  gl_PointSize = clamp(size / -mv.z, 6.0, 90.0) * uScale;",
        "  vFog = smoothstep(uFogNear, uFogFar, -mv.z);",
        "  gl_Position = projectionMatrix * mv;",
        "}",
      ].join("\n"),
      fragmentShader: [
        "uniform vec3 uFogColor;",
        "uniform float uGlow;",
        "varying vec3 vColor;",
        "varying float vAlpha;",
        "varying float vFog;",
        "void main() {",
        "  vec2 d = gl_PointCoord - vec2(0.5);",
        "  float r = length(d);",
        "  if (r > 0.5) discard;",
        "  float falloff = 1.0 - smoothstep(0.0, 0.5, r);",
        "  vec3 lift = mix(vec3(0.0), vec3(1.0), uGlow);",
        "  vec3 col = mix(vColor, lift, pow(falloff, 6.0) * 0.5);",
        "  col = mix(col, uFogColor, vFog * 0.85);",
        "  gl_FragColor = vec4(col, vAlpha * (0.25 + 0.75 * falloff));",
        "}",
      ].join("\n"),
    });
    const cloud = new THREE.Points(geometry, material);
    cloud.renderOrder = 2;
    scene.add(cloud);

    // ---- edges ----------------------------------------------------------
    const pairs = edges
      .map((e) => [index.get(e.source), index.get(e.target)] as const)
      .filter((l): l is readonly [number, number] => l[0] !== undefined && l[1] !== undefined);
    const edgePos = new Float32Array(pairs.length * 6);
    const edgeCol = new Float32Array(pairs.length * 6);
    const edgeInk = new THREE.Color(theme === "dark" ? 0x9aa3c8 : 0x5b6489);
    pairs.forEach(([a, b], k) => {
      edgePos.set(positions.subarray(a * 3, a * 3 + 3), k * 6);
      edgePos.set(positions.subarray(b * 3, b * 3 + 3), k * 6 + 3);
    });
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));
    edgeGeometry.setAttribute("color", new THREE.BufferAttribute(edgeCol, 3));
    const edgeMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: theme === "dark" ? 0.55 : 0.42,
      depthWrite: false,
    });
    const lines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    lines.renderOrder = 1;
    scene.add(lines);

    // ---- stars ----------------------------------------------------------
    // The same quiet field as the scatter: far outside the graph, fixed pixel
    // size, mostly dim, drifting too slowly to notice.
    let seed = 20261005;
    const rand = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
    const STARS = 1400;
    const starPos = new Float32Array(STARS * 3);
    const starCol = new Float32Array(STARS * 3);
    const starInk = new THREE.Color(theme === "dark" ? 0xdfe6ff : 0x8e98c4);
    for (let i = 0; i < STARS; i++) {
      const r = 60 + rand() * 25, th = rand() * Math.PI * 2, u = rand() * 2 - 1, w = Math.sqrt(1 - u * u);
      starPos.set([r * w * Math.cos(th), r * u, r * w * Math.sin(th)], i * 3);
      const b = 0.25 + Math.pow(rand(), 3) * 0.75;
      starCol.set([starInk.r * b, starInk.g * b, starInk.b * b], i * 3);
    }
    const starGeometry = new THREE.BufferGeometry();
    starGeometry.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    starGeometry.setAttribute("color", new THREE.BufferAttribute(starCol, 3));
    const stars = new THREE.Points(
      starGeometry,
      new THREE.PointsMaterial({
        size: 1.4,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: true,
        opacity: theme === "dark" ? 0.75 : 0.22,
        depthWrite: false,
      }),
    );
    stars.renderOrder = -1;
    scene.add(stars);

    // ---- labels ---------------------------------------------------------
    // HTML, positioned each frame, rather than sprites: crisp at any zoom,
    // real text, and the same font as the rest of the page. Only a few are
    // shown at once (see `emphasise`).
    const labelEls: HTMLDivElement[] = nodes.map((nd) => {
      const d = document.createElement("div");
      d.textContent = nd.name;
      d.style.cssText =
        `position:absolute;left:0;top:0;white-space:nowrap;font-size:11px;` +
        `pointer-events:none;color:${ink.text};text-shadow:0 0 4px ${GROUND[theme]},0 0 2px ${GROUND[theme]};` +
        `display:none;will-change:transform;`;
      layer.appendChild(d);
      return d;
    });
    let shown = new Set<number>();
    const byDegree = nodes.map((_, i) => i).sort((a, b) => nodes[b].degree - nodes[a].degree);

    // ---- camera ---------------------------------------------------------
    // Current and desired values eased together each frame, exactly as in the
    // scatter: input moves the desired value and the camera chases it.
    const target = new THREE.Vector3();
    const wantTarget = new THREE.Vector3();
    let yaw = 0.8, pitch = 0.35, radius = HOME_RADIUS;
    let wantYaw = yaw, wantPitch = pitch, wantRadius = radius;
    const place = () => {
      camera.position.set(
        target.x + radius * Math.cos(pitch) * Math.sin(yaw),
        target.y + radius * Math.sin(pitch),
        target.z + radius * Math.cos(pitch) * Math.cos(yaw),
      );
      camera.lookAt(target);
    };

    // ---- interaction ----------------------------------------------------
    let dragging = false, panning = false, moved = false, lastX = 0, lastY = 0;
    const pointer = new THREE.Vector2();
    const ray = new THREE.Raycaster();
    const pickAt = (x: number, y: number): number | null => {
      const rect = view.getBoundingClientRect();
      pointer.set(((x - rect.left) / rect.width) * 2 - 1, -((y - rect.top) / rect.height) * 2 + 1);
      ray.params.Points = { threshold: 0.03 * radius };
      ray.setFromCamera(pointer, camera);
      const hits = ray.intersectObject(cloud).filter((h) => alphas[h.index ?? 0] > 0.5);
      return hits.length ? hits[0].index ?? null : null;
    };
    const onDown = (e: PointerEvent) => {
      dragging = true;
      panning = e.shiftKey || e.button === 1 || e.button === 2;
      moved = false;
      lastX = e.clientX;
      lastY = e.clientY;
      view.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) {
        const i = pickAt(e.clientX, e.clientY);
        const rect = view.getBoundingClientRect();
        setHover(i === null ? null : { i, x: e.clientX - rect.left, y: e.clientY - rect.top });
        return;
      }
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      lastX = e.clientX;
      lastY = e.clientY;
      if (panning) {
        const right = new THREE.Vector3(), up = new THREE.Vector3();
        camera.matrixWorld.extractBasis(right, up, new THREE.Vector3());
        const k = radius * 0.0016;
        wantTarget.addScaledVector(right, -dx * k);
        wantTarget.addScaledVector(up, dy * k);
      } else {
        wantYaw -= dx * 0.005;
        wantPitch = Math.max(-1.45, Math.min(1.45, wantPitch + dy * 0.005));
      }
    };
    const onUp = (e: PointerEvent) => {
      dragging = false;
      panning = false;
      view.releasePointerCapture(e.pointerId);
      if (moved) return;
      selectRef.current(pickAt(e.clientX, e.clientY));
    };
    const onLeave = () => setHover(null);
    // passive:false is what makes scroll ZOOM instead of scrolling the page --
    // the earlier 2D version listened through React, whose wheel handler is
    // passive, so preventDefault was ignored and the page moved instead.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      wantRadius = Math.max(0.6, Math.min(30, wantRadius * Math.exp(e.deltaY * 0.0012)));
    };
    const onContext = (e: Event) => e.preventDefault();

    // WASD flight, EXPANDED ONLY. In the page it would steal keys from
    // scrolling and from every text field; fullscreen is where you explore.
    // W/S along the view, A/D sideways, E/Q up and down, Shift for speed.
    const keys = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => {
      if (!bigRef.current) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      const k = e.key.toLowerCase();
      if ("wasdqe".includes(k) && k.length === 1) {
        keys.add(k);
        e.preventDefault();
      }
      if (e.key === "Shift") keys.add("shift");
    };
    const onKeyUp = (e: KeyboardEvent) => {
      keys.delete(e.key.toLowerCase());
      if (e.key === "Shift") keys.delete("shift");
    };
    const onBlur = () => keys.clear();
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    const fwd = new THREE.Vector3();
    const side = new THREE.Vector3();
    const upv = new THREE.Vector3(0, 1, 0);

    view.addEventListener("pointerdown", onDown);
    view.addEventListener("pointermove", onMove);
    view.addEventListener("pointerup", onUp);
    view.addEventListener("pointerleave", onLeave);
    view.addEventListener("wheel", onWheel, { passive: false });
    view.addEventListener("contextmenu", onContext);

    // ---- imperative API -------------------------------------------------
    api.current.emphasise = (sel, hov, type, onlyBridges) => {
      // HOVER NEVER DIMS. It only grows the dot under the cursor, as in the
      // scatter; fading the whole graph every time the mouse crossed a node
      // made it flicker as you moved. Only a click, a type or Bridges fades.
      const focus = sel;
      // The WHOLE connected tree, not just the node and its neighbours: a
      // chain like CSC -> Toolkit -> NVIDIA is one idea, and lighting only
      // the first hop cut it in half.
      const near = focus !== null ? treeOf(focus) : null;
      const sizeAttr = geometry.getAttribute("size") as THREE.BufferAttribute;
      const alphaAttr = geometry.getAttribute("alpha") as THREE.BufferAttribute;
      for (let i = 0; i < n; i++) {
        let on = true;
        if (near) on = near.has(i);
        else if (type) on = nodes[i].type === type;
        else if (onlyBridges) on = nodes[i].documents.length > 1;
        alphas[i] = on ? 1 : 0.1;
        sizes[i] = i === sel ? SELECTED_SIZE : i === hov ? HOVER_SIZE : base[i];
      }
      sizeAttr.needsUpdate = true;
      alphaAttr.needsUpdate = true;
      // Edges: bright if both ends are lit, faint otherwise.
      pairs.forEach(([a, b], k) => {
        const lit = alphas[a] > 0.5 && alphas[b] > 0.5;
        const touchesFocus = focus !== null && (a === focus || b === focus);
        const s = touchesFocus ? 1.25 : lit ? 0.8 : 0.08;
        for (const end of [0, 3]) {
          edgeCol[k * 6 + end] = edgeInk.r * s;
          edgeCol[k * 6 + end + 1] = edgeInk.g * s;
          edgeCol[k * 6 + end + 2] = edgeInk.b * s;
        }
      });
      (edgeGeometry.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
      // Labels: the focus and its neighbours when there is one; otherwise the
      // best-connected lit nodes.
      const pool = near ? [...near] : byDegree.filter((i) => alphas[i] > 0.5);
      shown = new Set(pool.slice(0, MAX_LABELS));
    };

    // Breadth-first over relations: every node reachable from i.
    const treeOf = (i: number): Set<number> => {
      const seen = new Set<number>([i]);
      const queue = [i];
      while (queue.length) {
        const k = queue.shift()!;
        for (const j of neighbours[k]) {
          if (!seen.has(j)) {
            seen.add(j);
            queue.push(j);
          }
        }
      }
      return seen;
    };

    // Frame the clicked node's whole tree: centre on its centroid and back off
    // far enough to see all of it.
    api.current.flyToNode = (i) => {
      const tree = [...treeOf(i)];
      const c = new THREE.Vector3();
      for (const k of tree) c.add(new THREE.Vector3(positions[k * 3], positions[k * 3 + 1], positions[k * 3 + 2]));
      c.divideScalar(tree.length);
      let extent = 0;
      for (const k of tree) {
        extent = Math.max(extent, c.distanceTo(new THREE.Vector3(positions[k * 3], positions[k * 3 + 1], positions[k * 3 + 2])));
      }
      wantTarget.copy(c);
      // ZOOM IN ONLY. Pulled out to frame a big tree, a click would undo
      // the person's own zooming every time; it may come closer if they are
      // further out than the tree needs, never further away.
      const fit = Math.max(2.5, Math.min(HOME_RADIUS, extent * 2.8 + 1.2));
      wantRadius = Math.min(wantRadius, fit);
    };
    api.current.flyToType = (t) => {
      if (!t) {
        wantTarget.set(0, 0, 0);
        wantRadius = HOME_RADIUS;
        return;
      }
      const m = new THREE.Vector3();
      let k = 0;
      nodes.forEach((nd, i) => {
        if (nd.type !== t) return;
        m.x += positions[i * 3]; m.y += positions[i * 3 + 1]; m.z += positions[i * 3 + 2];
        k++;
      });
      if (k) wantTarget.copy(m.divideScalar(k));
      wantRadius = 6;
    };
    api.current.reset = () => {
      wantTarget.set(0, 0, 0);
      wantYaw = 0.8;
      wantPitch = 0.35;
      wantRadius = HOME_RADIUS;
    };
    api.current.setLabels = () => {};
    api.current.emphasise(null, null, null, false);

    // ---- loop -----------------------------------------------------------
    const onResize = () => {
      if (!el.clientWidth) return;
      camera.aspect = el.clientWidth / Math.max(1, el.clientHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(el);

    const v = new THREE.Vector3();
    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      if (keys.size && bigRef.current) {
        // Scaled by distance, like pan, so flight feels the same at any zoom.
        const step = radius * 0.012 * (keys.has("shift") ? 3 : 1);
        camera.getWorldDirection(fwd);
        side.crossVectors(fwd, upv).normalize();
        if (keys.has("w")) wantTarget.addScaledVector(fwd, step);
        if (keys.has("s")) wantTarget.addScaledVector(fwd, -step);
        if (keys.has("d")) wantTarget.addScaledVector(side, step);
        if (keys.has("a")) wantTarget.addScaledVector(side, -step);
        if (keys.has("e")) wantTarget.addScaledVector(upv, step);
        if (keys.has("q")) wantTarget.addScaledVector(upv, -step);
      }
      const k = 0.11;
      yaw += (wantYaw - yaw) * k;
      pitch += (wantPitch - pitch) * k;
      radius += (wantRadius - radius) * k;
      target.lerp(wantTarget, k);
      place();
      stars.rotation.y += 0.00004;
      renderer.render(scene, camera);

      const w = el.clientWidth, h = el.clientHeight;
      for (let i = 0; i < n; i++) {
        const d = labelEls[i];
        if (!labelsRef.current || !shown.has(i)) {
          if (d.style.display !== "none") d.style.display = "none";
          continue;
        }
        v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).project(camera);
        if (v.z > 1 || v.z < -1) {
          d.style.display = "none";
          continue;
        }
        d.style.display = "block";
        d.style.transform = `translate(${((v.x + 1) / 2) * w + 9}px, ${((1 - v.y) / 2) * h - 7}px)`;
      }
    };
    loop();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      view.removeEventListener("pointerdown", onDown);
      view.removeEventListener("pointermove", onMove);
      view.removeEventListener("pointerup", onUp);
      view.removeEventListener("pointerleave", onLeave);
      view.removeEventListener("wheel", onWheel);
      view.removeEventListener("contextmenu", onContext);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      geometry.dispose();
      material.dispose();
      edgeGeometry.dispose();
      edgeMaterial.dispose();
      starGeometry.dispose();
      (stars.material as THREE.Material).dispose();
      backdrop.dispose();
      renderer.dispose();
      labelEls.forEach((d) => d.remove());
      view.remove();
    };
  }, [nodes, edges, positions, neighbours, index, theme]);

  useEffect(() => {
    api.current.emphasise(selected, hover?.i ?? null, typeFocus, bridges);
  }, [selected, hover, typeFocus, bridges, theme]);

  useEffect(() => {
    if (!focusKey) return;
    const i = index.get(focusKey);
    if (i !== undefined) {
      setSelected(i);
      onFocused();
    }
  }, [focusKey, index, onFocused]);

  useEffect(() => {
    setPassage(null);
    setHydration(undefined);
    setHydrateError(null);
    if (selected === null) return;
    api.current.flyToNode(selected);
    // A saved profile shows up the moment the entity is opened; nothing is
    // generated without a click.
    let live = true;
    if (nodes[selected].id.startsWith("cat:")) {
      setHydration(null);
      return;
    }
    getHydration(nodes[selected].id)
      .then((h) => live && setHydration(h))
      .catch(() => live && setHydration(null));
    return () => {
      live = false;
    };
  }, [selected, nodes]);

  const hydrate = useCallback(async (key: string) => {
    setHydrating(true);
    setHydrateError(null);
    try {
      setHydration(await hydrateEntity(key));
    } catch (e) {
      setHydrateError(e instanceof Error ? e.message : "Could not hydrate");
    } finally {
      setHydrating(false);
    }
  }, []);

  useEffect(() => {
    api.current.flyToType(typeFocus);
  }, [typeFocus]);

  // Fullscreen with a CSS fallback, as in the scatter.
  useEffect(() => {
    const onChange = () => setNativeFull(document.fullscreenElement === host.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (passage) setPassage(null);
      else if (selected !== null) setSelected(null);
      else if (expanded) setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded, selected, passage]);
  const toggleBig = useCallback(async () => {
    if (document.fullscreenElement) {
      await document.exitFullscreen().catch(() => {});
      return;
    }
    if (expanded) {
      setExpanded(false);
      return;
    }
    try {
      await host.current?.requestFullscreen();
    } catch {
      setExpanded(true);
    }
  }, [expanded]);

  const ink = INK[theme];
  const colours = palette[theme];
  const panel = theme === "dark" ? "rgba(16,18,26,0.94)" : "rgba(255,255,255,0.95)";
  const hovered = hover ? nodes[hover.i] : null;
  const chosen = selected !== null ? nodes[selected] : null;
  const relations = chosen
    ? edges.filter((e) => e.source === chosen.id || e.target === chosen.id)
    : [];
  const nameOf = (id: string) => nodes[index.get(id) ?? -1]?.name ?? id;
  const typeCounts = useMemo(() => {
    const m: Record<string, number> = {};
    for (const nd of nodes) m[nd.type] = (m[nd.type] ?? 0) + 1;
    return m;
  }, [nodes]);
  const matches = query.trim()
    ? nodes
        .map((nd, i) => ({ nd, i }))
        .filter(({ nd }) => nd.name.toLowerCase().includes(query.trim().toLowerCase()))
        .slice(0, 8)
    : [];

  const chip = (on: boolean) => ({
    background: on ? ink.on : ink.chip,
    color: on ? ink.onText : ink.text,
  });

  return (
    <div className="space-y-3">
      <details className="md-body-small" style={{ color: "var(--md-on-surface-variant)" }}>
        <summary className="md-label-large" style={{ color: "var(--md-on-surface)" }}>
          How to use the graph
        </summary>
        <div className="mt-2 space-y-1.5">
          <p>
            Each dot is an entity your documents name, coloured by type. Each line is a relationship a document states, such
            as &ldquo;Acme acquired Beta&rdquo;. Bridges shows only the entities that appear in more than one document, which is where documents connect.
          </p>
          <p>
            Drag to orbit, shift-drag or right-drag to pan, and scroll to zoom. Hover a dot
            for its details; click it to fly there, light up its neighbours and list what
            the documents say about it. Click a relationship in that list to open the
            passage it came from. Click a type in the legend to show only that type, or use
            Bridges to show only cross-document entities. Search jumps to an entity by
            name. In Expand, fly with W/A/S/D, rise and sink with E/Q, and hold Shift to
            go faster. Esc closes a passage, then the selection; Reset brings the whole
            graph back.
          </p>
        </div>
      </details>

      <div
        ref={host}
        className={
          big
            ? "fixed inset-0 z-[60] w-full touch-none overflow-hidden"
            : "relative h-[34rem] w-full touch-none overflow-hidden rounded-[var(--md-shape-lg)]"
        }
        style={{ background: GROUND[theme], cursor: hovered ? "pointer" : "grab" }}
      >
        <div ref={labelLayer} className="pointer-events-none absolute inset-0 z-[5] overflow-hidden" />

        {/* Top-left: search and legend. */}
        <div className="pointer-events-none absolute left-3 top-3 z-10 flex w-56 flex-col gap-2">
          <div className="pointer-events-auto flex rounded-[var(--md-shape-sm)] p-0.5 text-xs" style={{ background: panel }} role="tablist" aria-label="Which trees to show">
            {(["all", "documents", "topics"] as Source[]).map((s) => (
              <button
                key={s}
                type="button"
                role="tab"
                aria-selected={source === s}
                onClick={() => onSource(s)}
                className="flex-1 rounded-[var(--md-shape-sm)] px-2 py-1"
                style={{ background: source === s ? ink.chip : "transparent", color: ink.text }}
              >
                {s === "all" ? "All" : s === "documents" ? "Documents" : "Topics"}
              </button>
            ))}
          </div>
          <form
            className="pointer-events-auto flex gap-1"
            onSubmit={async (e) => {
              e.preventDefault();
              const name = newTopic.trim();
              if (!name || topicBusy) return;
              setTopicBusy("add");
              setTopicError(null);
              try {
                const r = await addTopic(name);
                setNewTopic("");
                await onReload(r.key);
              } catch (err) {
                setTopicError(err instanceof Error ? err.message : "Could not add it");
              } finally {
                setTopicBusy(null);
              }
            }}
          >
            <input
              value={newTopic}
              onChange={(e) => setNewTopic(e.target.value)}
              placeholder="Add a topic to learn…"
              aria-label="Add a topic"
              className="min-w-0 flex-1 rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs outline-none"
              style={{ background: panel, color: ink.text, border: `1px solid ${ink.chip}` }}
            />
            <button type="submit" disabled={!newTopic.trim() || !!topicBusy} className="rounded-[var(--md-shape-sm)] px-2 text-xs" style={chip(true)}>
              {topicBusy === "add" ? "…" : "Add"}
            </button>
          </form>
          {topicError && (
            <p className="pointer-events-auto rounded-[var(--md-shape-sm)] p-1.5 text-xs" style={{ background: panel, color: ink.text }}>
              {topicError}
            </p>
          )}
          <div className="pointer-events-auto relative">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Find an entity…"
              aria-label="Find an entity"
              className="w-full rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs outline-none"
              style={{ background: panel, color: ink.text, border: `1px solid ${ink.chip}` }}
            />
            {matches.length > 0 && (
              <ul
                className="absolute left-0 right-0 top-full mt-1 overflow-hidden rounded-[var(--md-shape-sm)]"
                style={{ background: panel, border: `1px solid ${ink.chip}` }}
              >
                {matches.map(({ nd, i }) => (
                  <li key={nd.id}>
                    <button
                      type="button"
                      className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs"
                      style={{ color: ink.text }}
                      onClick={() => {
                        setTypeFocus(null);
                        setSelected(i);
                        setQuery("");
                      }}
                    >
                      <span className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: colours[nd.type] ?? colours.other }} />
                      <span className="min-w-0 flex-1 truncate">{nd.name}</span>
                      <span style={{ opacity: 0.55 }}>{nd.degree}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="pointer-events-auto flex flex-col gap-0.5 rounded-[var(--md-shape-md)] p-1.5" style={{ background: panel }}>
            {(categories ? Object.keys(CATEGORY_COLOURS.dark) : TYPES).filter((t) => typeCounts[t]).map((t) => {
              const on = typeFocus === t;
              return (
                <button
                  key={t}
                  type="button"
                  aria-pressed={on}
                  onClick={() => {
                    setSelected(null);
                    setTypeFocus(on ? null : t);
                  }}
                  className="flex items-center gap-2 rounded-[var(--md-shape-sm)] px-2 py-1 text-left text-xs"
                  style={{ background: on ? ink.chip : "transparent", color: ink.text }}
                >
                  <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: colours[t] }} />
                  <span className="flex-1">{t}</span>
                  <span style={{ opacity: 0.55 }}>{typeCounts[t]}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Top-right: the scatter's controls, in the same order. */}
        <div className="pointer-events-none absolute right-3 top-3 z-10 flex gap-2">
          <button type="button" className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs" style={chip(false)} onClick={() => onThemeChange(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          <button type="button" className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs" style={chip(labels)} onClick={() => setLabels((v) => !v)} title="Show names next to the best-connected entities">
            Labels
          </button>
          <button type="button" className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs" style={chip(categories)} onClick={onToggleCategories} title="Group the graph into category trees: Technology, Nature, Business...">
            Categories
          </button>
          <button type="button" className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs" style={chip(bridges)} onClick={() => { setSelected(null); setTypeFocus(null); setBridges((v) => !v); }} title="Show only entities that appear in more than one document">
            Bridges
          </button>
          <button type="button" className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs" style={chip(false)} onClick={() => { setSelected(null); setTypeFocus(null); setBridges(false); api.current.reset(); }}>
            Reset view
          </button>
          <button type="button" className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs" style={chip(false)} onClick={() => void toggleBig()}>
            {big ? "Exit" : "Expand"}
          </button>
        </div>

        <p className="pointer-events-none absolute bottom-3 left-3 z-10 text-xs" style={{ color: ink.faint }}>
          drag to orbit · shift-drag or right-drag to pan · scroll to zoom · click an entity
          {big ? " · WASD to fly, E/Q up/down, Shift faster · Esc to exit" : ""}
          {bridges && " · showing entities found in more than one document"}
        </p>

        {/* The reading panel: everything the documents say about the
            selected entity, each statement one click from its passage. */}
        {chosen && (
          <div
            className="absolute bottom-10 right-3 top-14 z-10 flex flex-col rounded-[var(--md-shape-md)]"
            style={{ background: panel, color: ink.text, width: panelWidth, maxWidth: "70%" }}
          >
            {/* Drag handle: the panel's left edge. Width is remembered. */}
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize panel"
              onPointerDown={startResize}
              className="absolute -left-1 bottom-0 top-0 z-20 w-2 cursor-col-resize"
              title="Drag to resize"
            />
            <div className="flex items-start gap-2 p-4 pb-2">
              <div className="min-w-0 flex-1">
                <p className="break-words text-sm font-medium">{chosen.name}</p>
                <p className="mt-0.5 text-xs" style={{ color: ink.faint }}>
                  {chosen.type} · {chosen.mentions} mention{chosen.mentions === 1 ? "" : "s"} ·{" "}
                  {chosen.degree} relation{chosen.degree === 1 ? "" : "s"}
                </p>
                <p className="mt-1 break-words text-xs" style={{ color: ink.faint }}>
                  {chosen.documents.length
                    ? `in ${chosen.documents.join(", ")}`
                    : chosen.id.startsWith("cat:")
                      ? "category"
                      : "your topic · links are general knowledge, not from a document"}
                </p>
              </div>
              <button type="button" onClick={() => setSelected(null)} className="rounded-[var(--md-shape-sm)] px-2 py-1 text-xs" style={chip(false)}>
                Close
              </button>
            </div>
            {!chosen.id.startsWith("cat:") && (
              <div className="flex flex-wrap gap-2 px-4 pb-3">
                <button
                  type="button"
                  disabled={!!topicBusy}
                  className="rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs font-medium"
                  style={chip(true)}
                  title="Pull in the concepts around this one as new connected topics"
                  onClick={async () => {
                    setTopicBusy("explore");
                    setTopicError(null);
                    try {
                      await exploreTopic(chosen.id);
                      await onReload(chosen.id);
                    } catch (err) {
                      setTopicError(err instanceof Error ? err.message : "Could not explore");
                    } finally {
                      setTopicBusy(null);
                    }
                  }}
                >
                  {topicBusy === "explore" ? "Exploring…" : "Explore related topics"}
                </button>
                {chosen.origin === "topic" && (
                  <button
                    type="button"
                    disabled={!!topicBusy}
                    className="rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs"
                    style={chip(false)}
                    onClick={async () => {
                      setTopicBusy("remove");
                      try {
                        await deleteTopic(chosen.id);
                        setSelected(null);
                        await onReload();
                      } finally {
                        setTopicBusy(null);
                      }
                    }}
                  >
                    Remove topic
                  </button>
                )}
              </div>
            )}
            {passage ? (
              <div className="flex min-h-0 flex-1 flex-col px-4 pb-4">
                <button type="button" onClick={() => setPassage(null)} className="mb-2 self-start rounded-[var(--md-shape-sm)] px-2 py-1 text-xs" style={chip(false)}>
                  ← Back to relations
                </button>
                {passage === "loading" ? (
                  <p className="text-xs" style={{ color: ink.faint }}>Loading the passage…</p>
                ) : (
                  <>
                    <p className="text-xs" style={{ color: ink.faint }}>
                      {passage.filename} · {passage.heading ?? "no heading"} · chunk {passage.chunk_index}
                      {passage.page ? ` · page ${passage.page}` : ""}
                    </p>
                    <div className="mt-2 min-h-0 flex-1 overflow-y-auto">
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{passage.text}</p>
                    </div>
                  </>
                )}
              </div>
            ) : (
            <div className="min-h-0 flex-1 overflow-y-auto">
            <div className="px-4 pb-3">
              {chosen.id.startsWith("cat:") ? (
                <p className="text-xs" style={{ color: ink.faint }}>
                  A category: every term below was classified into it. Go to any term to see how it connects.
                </p>
              ) : hydration ? (
                <div className="space-y-2.5 rounded-[var(--md-shape-sm)] p-3" style={{ background: ink.chip }}>
                  <p className="text-[11px] uppercase tracking-wide" style={{ color: ink.faint }}>
                    From the web
                  </p>
                  {hydration.paragraphs.map((para, k) => (
                    <p key={k} className="text-xs leading-relaxed">{para}</p>
                  ))}
                  {hydration.facts.length > 0 && (
                    <>
                      <p className="pt-1 text-[11px] uppercase tracking-wide" style={{ color: ink.faint }}>
                        Interesting facts
                      </p>
                      <ul className="list-disc space-y-1 pl-4">
                        {hydration.facts.map((f, k) => (
                          <li key={k} className="text-xs leading-relaxed">{f}</li>
                        ))}
                      </ul>
                    </>
                  )}
                  {hydration.sources.length > 0 && (
                    <ol className="space-y-0.5 pt-1 text-[11px]" style={{ color: ink.faint }}>
                      {hydration.sources.map((src, k) => (
                        <li key={k} className="truncate">
                          [{k + 1}]{" "}
                          <a href={src.url} target="_blank" rel="noreferrer" className="underline" style={{ color: ink.text }}>
                            {src.title}
                          </a>
                        </li>
                      ))}
                    </ol>
                  )}
                  <button
                    type="button"
                    onClick={() => void hydrate(chosen.id)}
                    disabled={hydrating}
                    className="text-[11px] underline"
                    style={{ color: ink.faint }}
                  >
                    {hydrating ? "Refreshing…" : "Refresh from the web"}
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => void hydrate(chosen.id)}
                  disabled={hydrating || hydration === undefined}
                  className="flex w-full items-center justify-center gap-2 rounded-[var(--md-shape-sm)] px-3 py-2 text-xs font-medium"
                  style={chip(true)}
                  title="Search the web for this entity and save a short profile"
                >
                  {hydrating && <IconSpinner />}
                  {hydrating ? "Reading the web…" : "Hydrate from web knowledge"}
                </button>
              )}
              {hydrateError && (
                <p className="mt-1.5 text-xs" style={{ color: ink.faint }}>{hydrateError}</p>
              )}
            </div>
            <ul className="space-y-1.5 px-4 pb-4">
              {passageError && (
                <li className="text-xs" style={{ color: ink.faint }}>{passageError}</li>
              )}
              {relations.length === 0 && (
                <li className="text-xs" style={{ color: ink.faint }}>
                  Named in the documents, but no relationship to another entity was stated.
                </li>
              )}
              {relations.map((r, k) => {
                const other = r.source === chosen.id ? r.target : r.source;
                return (
                  <li key={k}>
                    <div className="rounded-[var(--md-shape-sm)] p-2 text-xs" style={{ background: ink.chip }}>
                      <p className="leading-snug">
                        <strong>{nameOf(r.source)}</strong> {r.predicate} <strong>{nameOf(r.target)}</strong>
                      </p>
                      <div className="mt-1.5 flex flex-wrap items-center gap-2" style={{ color: ink.faint }}>
                        <span className="min-w-0 flex-1 truncate">{r.filename}</span>
                        {r.chunk_id && (
                          <button type="button" className="underline" style={{ color: ink.text }} onClick={() => void openPassage(r.chunk_id!)}>
                            Open passage
                          </button>
                        )}
                        <button
                          type="button"
                          className="underline"
                          style={{ color: ink.text }}
                          onClick={() => {
                            const j = index.get(other);
                            if (j !== undefined) setSelected(j);
                          }}
                        >
                          Go to {nameOf(other)}
                        </button>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
            </div>
            )}
          </div>
        )}

        {hovered && hover && hover.i !== selected && (
          <div
            className="pointer-events-none absolute z-20 w-[16rem] rounded-[var(--md-shape-md)] p-3"
            style={{
              left: Math.max(8, Math.min(hover.x + 16, (host.current?.clientWidth ?? 0) - 270)),
              top: Math.min(hover.y + 16, (host.current?.clientHeight ?? 0) - 110),
              background: panel,
              color: ink.text,
              border: `1px solid ${ink.chip}`,
            }}
          >
            <p className="break-words text-xs font-medium">{hovered.name}</p>
            <p className="mt-0.5 text-xs" style={{ color: ink.faint }}>
              {hovered.type} · {hovered.degree} relation{hovered.degree === 1 ? "" : "s"} ·{" "}
              {hovered.documents.length} document{hovered.documents.length === 1 ? "" : "s"}
            </p>
            <p className="mt-1 break-words text-xs" style={{ color: ink.faint }}>
              {hovered.documents.slice(0, 3).join(", ")}
              {hovered.documents.length > 3 ? "…" : ""}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
