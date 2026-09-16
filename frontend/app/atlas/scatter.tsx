"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { AtlasPoint } from "@/lib/api";

/**
 * The corpus as a point cloud you can fly through.
 *
 * RAW three.js, NOT react-three-fiber. Fiber declares a peer range of
 * `react >=19 <19.3` and this project is on 19.3, so installing it means
 * overriding a constraint its authors set deliberately. The React binding also
 * buys little here: there is one scene, built once, and a component that
 * re-renders sixty times a second to move a camera is the thing to avoid.
 *
 * So the scene lives in one effect, and React owns only the container, the
 * tooltip and the overlay buttons.
 */

/**
 * Document colours, one set per background.
 *
 * The first version reused Material's surface palette, which is tuned for text
 * on light backgrounds: at four pixels across those hues were indistinguishable
 * from one another. These are spread around the wheel at high chroma, because
 * the only job a point's colour has is to say which document it came from.
 *
 * TWO SETS RATHER THAN ONE, because a colour legible on near-black is washed
 * out on near-white and the reverse. The light set is the same hues taken
 * darker and deeper so they hold against a pale ground.
 */
export const PALETTES = {
  dark: [
    "#a78bfa", // violet
    "#34e3a4", // mint
    "#ff6b81", // coral
    "#ffc247", // amber
    "#4fd1ff", // cyan
    "#ff8ae2", // orchid
    "#b6ef6a", // lime
    "#ffa06b", // tangerine
  ],
  light: [
    "#5b3fd6", // violet
    "#00855a", // mint
    "#c62348", // coral
    "#a86400", // amber
    "#0369a1", // cyan
    "#b52f93", // orchid
    "#4d7c0f", // lime
    "#c2410c", // tangerine
  ],
} as const;

export type PlotTheme = keyof typeof PALETTES;

/**
 * How far the unit-box projection is spread before drawing.
 *
 * The backend scales its coordinates into a unit-ish box so the camera does
 * not have to guess a zoom for an arbitrary embedding space. At that size every
 * point sits within a few pixels of its neighbours once the camera is far
 * enough back to see all of them.
 *
 * Exported because the query-ray view plots the SAME projection: a different
 * factor there would place a query convincingly in the wrong place.
 */
export const SPREAD = 2.4;

/** The plot's own ground, deliberately independent of the app theme. */
export const GROUND: Record<PlotTheme, string> = {
  dark: "#0e1016",
  light: "#eef0f6",
};

/**
 * The backdrop, as a radial gradient: lit near the middle, deep at the edges.
 *
 * WHY A GRADIENT AND NOT A FLAT COLOUR
 *
 * Removing the floor left the cloud hanging in an even black field, and an
 * even field gives the eye nothing: no centre, no sense of depth, no clue that
 * the space continues past the points. A flat colour reads as a wall behind a
 * picture. A vignette reads as a volume you are inside, because that is what
 * distance looks like -- and unlike a floor it asserts no direction, which
 * matters when the third axis of a PCA is not a height.
 *
 * `edge` doubles as the fog colour, so a point receding into the distance
 * arrives at exactly the shade the background is already showing there.
 */
export const BACKDROP: Record<PlotTheme, { core: string; edge: string }> = {
  dark: { core: "#222a44", edge: "#06070b" },
  light: { core: "#ffffff", edge: "#c9cfe0" },
};

/**
 * The gradient, baked into a texture.
 *
 * `scene.background` accepts a texture and stretches it across the viewport
 * without any geometry, so this costs one 512-pixel canvas and no draw call of
 * its own -- considerably less machinery than the usual trick of a huge
 * inside-out sphere that has to be kept centred on the camera every frame.
 */
export function backdropTexture(theme: PlotTheme): THREE.Texture {
  const size = 512;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const { core, edge } = BACKDROP[theme];
  const g = ctx.createRadialGradient(
    size / 2,
    size * 0.46,
    0,
    size / 2,
    size * 0.46,
    size * 0.62,
  );
  g.addColorStop(0, core);
  g.addColorStop(1, edge);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  // The gradient is smooth, so linear filtering is what it wants -- the
  // opposite of the territory map, where the hard boundaries ARE the data.
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}


/** Overlay text and chrome, which must sit on whichever ground is active. */
export const INK: Record<
  PlotTheme,
  { text: string; faint: string; chip: string; on: string; onText: string }
> = {
  dark: {
    text: "#e8e9f0",
    faint: "rgba(232,233,240,0.6)",
    chip: "rgba(255,255,255,0.12)",
    on: "#8b7bf5",
    onText: "#ffffff",
  },
  light: {
    text: "#16181f",
    faint: "rgba(22,24,31,0.6)",
    chip: "rgba(0,0,0,0.07)",
    on: "#5b3fd6",
    onText: "#ffffff",
  },
};

type Props = {
  points: AtlasPoint[];
  selected: number | null;
  onSelect: (index: number | null) => void;
  /** Filename to fly to, or null for the whole corpus. */
  focus: string | null;
  /** Needed so the expanded view can carry its own legend and fly-to. */
  onFocusChange: (filename: string | null) => void;
  theme: PlotTheme;
  onThemeChange: (theme: PlotTheme) => void;
};

export default function Scatter({
  points,
  selected,
  onSelect,
  focus,
  onFocusChange,
  theme,
  onThemeChange,
}: Props) {
  // Also computed inside the scene effect, from the same sorted order, so the
  // legend's colours cannot disagree with the points'.
  const files = useMemo(
    () => [...new Set(points.map((p) => p.filename))].sort(),
    [points],
  );
  const host = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(
    null,
  );
  /**
   * Expansion, with a CSS fallback behind the Fullscreen API.
   *
   * The API is tried first because hiding the browser chrome is genuinely
   * better where the browser allows it. It can still reject -- a
   * Permissions-Policy, a transformed ancestor -- and `requestFullscreen()`
   * rejects through a PROMISE, so a discarded rejection leaves a button that
   * does nothing and says nothing. The fallback removes that failure mode.
   *
   * NOTE: this was NOT why the button appeared broken. That was pointer
   * capture on the container swallowing the click; see the listener setup
   * below. The fallback is worth keeping on its own merits, but the API was
   * never the culprit.
   */
  const [expanded, setExpanded] = useState(false);
  const [nativeFull, setNativeFull] = useState(false);
  const big = expanded || nativeFull;

  /**
   * Reading order, drawn as a line through each document's chunks.
   *
   * What it shows that nothing else does: chunking quality. A document whose
   * consecutive sections are about related things traces a compact, slowly
   * wandering curve. One that ricochets across the space was cut mid-argument
   * -- or genuinely changes subject every few hundred characters, which is
   * equally worth knowing before blaming retrieval for missing it.
   */
  const [threads, setThreads] = useState(false);

  /**
   * Which document owns each patch of ground.
   *
   * Answers a question that has no other form: for a query nobody has written
   * yet, which document will win? A document with a large territory dominates
   * a whole region of concept space. A document with none is one retrieval
   * will effectively never reach, and it sits in the library looking perfectly
   * healthy.
   */
  const [territory, setTerritory] = useState(false);

  // Handlers change identity every render; the scene is built once. Refs keep
  // the effect from tearing down the whole WebGL context on a parent render.
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // Imperative hooks into the live scene, assigned when it is built.
  const api = useRef<{
    highlight: (sel: number | null, hovered: number | null) => void;
    flyTo: (filename: string | null) => void;
    reset: () => void;
    setThreads: (on: boolean) => void;
    setTerritory: (on: boolean) => void;
  }>({
    highlight: () => {},
    flyTo: () => {},
    reset: () => {},
    setThreads: () => {},
    setTerritory: () => {},
  });

  // Read inside the scene effect without making it a dependency -- depending
  // on the values would rebuild the whole scene on every toggle.
  const threadsRef = useRef(threads);
  threadsRef.current = threads;
  const territoryRef = useRef(territory);
  territoryRef.current = territory;

  // Toggling visibility rather than rebuilding: both overlays are built once
  // with the scene. Putting them in the scene effect's dependencies would tear
  // down and recreate the WebGL context on every click, losing the camera
  // position with it.
  useEffect(() => {
    api.current.setThreads(threads);
  }, [threads]);
  useEffect(() => {
    api.current.setTerritory(territory);
  }, [territory]);

  useEffect(() => {
    const el = host.current;
    if (!el || points.length === 0) return;

    const ground = new THREE.Color(GROUND[theme]);
    const scene = new THREE.Scene();
    const backdrop = backdropTexture(theme);
    scene.background = backdrop;
    // Fog fades distant points, which is most of what makes a flat scatter
    // read as a volume you are inside rather than a picture you look at.
    // Pushed well out from the original 4-13: at 13 units it swallowed the
    // points themselves at full zoom-out, leaving an empty field.
    //
    // Fogged to the backdrop's EDGE colour, so a receding point arrives at the
    // shade already behind it instead of dissolving into a different grey.
    scene.fog = new THREE.Fog(
      new THREE.Color(BACKDROP[theme].edge),
      FOG_NEAR,
      FOG_FAR,
    );

    const camera = new THREE.PerspectiveCamera(
      50,
      el.clientWidth / Math.max(1, el.clientHeight),
      0.01,
      100,
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const view = renderer.domElement;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);

    // ---- geometry -------------------------------------------------------
    const palette = PALETTES[theme];
    const files = [...new Set(points.map((p) => p.filename))].sort();
    const n = points.length;
    const positions = new Float32Array(n * 3);
    const colours = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const alphas = new Float32Array(n);
    const colour = new THREE.Color();

    points.forEach((p, i) => {
      // Spread out: the projection arrives squeezed into a unit box, which
      // puts every point within a few pixels of its neighbours once the camera
      // is far enough back to see all of it.
      positions[i * 3] = p.x * SPREAD;
      positions[i * 3 + 1] = p.y * SPREAD;
      positions[i * 3 + 2] = p.z * SPREAD;
      colour.set(palette[files.indexOf(p.filename) % palette.length]);
      colours[i * 3] = colour.r;
      colours[i * 3 + 1] = colour.g;
      colours[i * 3 + 2] = colour.b;
      sizes[i] = BASE_SIZE;
      alphas[i] = 1;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("alpha", new THREE.BufferAttribute(alphas, 1));

    // A shader rather than PointsMaterial, for two things it cannot do:
    // per-point SIZE, so hover and selection can grow a dot, and per-point
    // ALPHA, so focusing one document can fade the rest.
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uFogColor: { value: ground.clone() },
        uFogNear: { value: FOG_NEAR },
        uFogFar: { value: FOG_FAR },
        // 1 on dark, 0 on light. The core highlight blends towards white on a
        // dark ground and towards black on a pale one -- blending to white on
        // white is what made points vanish.
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
        // Divided by depth so points shrink with distance -- without it a
        // cloud of equal dots reads as flat however it is rotated -- then
        // CLAMPED, which is what stops a zoomed-out view becoming four-pixel
        // specks and a zoomed-in one filling the screen with two dots.
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
        // A SOFT falloff from the centre, not a hard-edged disc. The disc
        // version read as a UI element; the blur reads as a light source,
        // which is what a point in a cloud should look like. The earlier
        // problem was that these were four pixels across, not that they were
        // soft -- size is fixed in the vertex shader, so the look can stay.
        "  float falloff = 1.0 - smoothstep(0.0, 0.5, r);",
        "  vec3 lift = mix(vec3(0.0), vec3(1.0), uGlow);",
        "  vec3 col = mix(vColor, lift, pow(falloff, 6.0) * 0.5);",
        "  col = mix(col, uFogColor, vFog * 0.85);",
        // Raised off the floor so the blurred rim still carries colour
        // instead of dissolving into the background.
        "  gl_FragColor = vec4(col, vAlpha * (0.25 + 0.75 * falloff));",
        "}",
      ].join("\n"),
      vertexColors: true,
    });

    const cloud = new THREE.Points(geometry, material);
    // EXPLICIT ORDER, because the default flips as you orbit.
    //
    // Every transparent object is drawn after the opaque ones and sorted
    // back-to-front BY OBJECT CENTRE. The cloud, the threads and the territory
    // plane all centre near the origin, so which counts as "nearer" changes as
    // the camera swings around -- and at some angles a layer was drawn last and
    // painted straight over the points. The points also set `depthWrite:
    // false`, so they never occlude anything and cannot win on depth alone.
    cloud.renderOrder = 1;
    scene.add(cloud);

    // NO FLOOR.
    //
    // There was a 220-unit grid here, and it was wrong for this data. A ground
    // plane asserts a DOWN, and a PCA projection has none: the third component
    // is an axis like the other two, not a height, and points sit on both
    // sides of any plane you draw. So roughly a third of the corpus rendered
    // beneath the floor, where the grid's depth writes hid it outright at some
    // angles and the lines cut across it at others.
    //
    // Orientation while flying comes from the cloud's own parallax instead,
    // which is honest about the fact that there is no up.

    // ---- dust -----------------------------------------------------------
    // Faint specks scattered through the surrounding volume.
    //
    // This is what actually replaces the floor. The floor's real job was never
    // decoration: it was MOTION PARALLAX -- near things sliding past faster
    // than far things is how the eye reads depth, and with an empty background
    // there is nothing to slide. Dust restores that without asserting a
    // direction, because it surrounds the cloud evenly instead of lying under
    // it.
    //
    // Deterministic, from a fixed seed, so the field does not reshuffle on
    // every theme change and read as a flicker.
    const DUST = 700;
    const dustPositions = new Float32Array(DUST * 3);
    let seed = 20260916;
    const rand = () => {
      // Mulberry-ish LCG. Math.random would be reseeded on every rebuild.
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let i = 0; i < DUST; i++) {
      // Spherical shell between the data and the fog, so dust never appears
      // among the points (where it would read as corpus) nor beyond the fog
      // (where it would be invisible anyway).
      const r = 7 + rand() * 17;
      const theta = rand() * Math.PI * 2;
      const u = rand() * 2 - 1;
      const w = Math.sqrt(1 - u * u);
      dustPositions[i * 3] = r * w * Math.cos(theta);
      dustPositions[i * 3 + 1] = r * u;
      dustPositions[i * 3 + 2] = r * w * Math.sin(theta);
    }
    const dustGeometry = new THREE.BufferGeometry();
    dustGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(dustPositions, 3),
    );
    const dust = new THREE.Points(
      dustGeometry,
      new THREE.PointsMaterial({
        color: new THREE.Color(theme === "dark" ? 0xaab2d0 : 0x6b73a0),
        size: 0.03,
        sizeAttenuation: true,
        transparent: true,
        opacity: theme === "dark" ? 0.5 : 0.35,
        depthWrite: false,
        fog: true,
      }),
    );
    dust.renderOrder = 0;
    scene.add(dust);


    // ---- reading-order threads ------------------------------------------
    // One polyline per document, through its chunks in chunk_index order.
    //
    // Built here rather than on toggle: the geometry is a few hundred vertices
    // and building it once costs less than the allocation churn of creating
    // and disposing it every time someone flicks the switch.
    const threadGroup = new THREE.Group();
    threadGroup.visible = false;
    for (const file of files) {
      const ordered = points
        .map((p, i) => ({ p, i }))
        .filter(({ p }) => p.filename === file)
        .sort((a, b) => a.p.chunk_index - b.p.chunk_index);
      // A single-chunk document has no reading order to draw. `Line` with one
      // vertex renders nothing anyway, but skipping is clearer than relying on
      // that.
      if (ordered.length < 2) continue;
      const verts = new Float32Array(ordered.length * 3);
      ordered.forEach(({ i }, k) => {
        verts[k * 3] = positions[i * 3];
        verts[k * 3 + 1] = positions[i * 3 + 1];
        verts[k * 3 + 2] = positions[i * 3 + 2];
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
      const line = new THREE.Line(
        g,
        new THREE.LineBasicMaterial({
          color: new THREE.Color(
            palette[files.indexOf(file) % palette.length],
          ),
          transparent: true,
          opacity: 0.7,
        }),
      );
      // Above the grid, below the points: the thread should read as connecting
      // the dots, not as covering them.
      line.renderOrder = 0.5;
      threadGroup.add(line);
    }
    scene.add(threadGroup);

    // ---- territory -------------------------------------------------------
    // A Voronoi partition painted onto the floor: every patch takes the colour
    // of the document whose nearest chunk owns it.
    //
    // HONEST LIMITATION, stated in the legend too: the floor is two
    // dimensional and the projection is three, so this partitions the plot's
    // SHADOW on the XZ plane. Two chunks stacked vertically collapse onto one
    // another here. It is a map of the ground, not a slice through the space,
    // and reading it as the latter would overstate what it knows.
    //
    // Computed on a canvas rather than in a shader because it runs once, on a
    // few hundred points, and a shader would need the points uploaded as a
    // texture to be read in a loop -- considerably more machinery for
    // something that takes about fifty milliseconds here.
    const RES = 192;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = RES;
    const ctx = canvas.getContext("2d");
    let territoryMesh: THREE.Mesh | null = null;
    let stems: THREE.LineSegments | null = null;
    if (ctx) {
      // Painted over the data's own extent plus a margin, NOT over the whole
      // 220-unit grid: at that scale the entire corpus would occupy two
      // texels and the floor would be one flat colour.
      let extent = 0.5;
      for (let i = 0; i < n; i++) {
        extent = Math.max(
          extent,
          Math.abs(positions[i * 3]),
          Math.abs(positions[i * 3 + 2]),
        );
      }
      extent *= 1.25;

      const image = ctx.createImageData(RES, RES);
      const rgb = new THREE.Color();
      for (let py = 0; py < RES; py++) {
        const wz = ((py + 0.5) / RES) * 2 * extent - extent;
        for (let px = 0; px < RES; px++) {
          const wx = ((px + 0.5) / RES) * 2 * extent - extent;
          let best = -1;
          let bestD = Infinity;
          for (let i = 0; i < n; i++) {
            const dx = positions[i * 3] - wx;
            const dz = positions[i * 3 + 2] - wz;
            const d = dx * dx + dz * dz;
            if (d < bestD) {
              bestD = d;
              best = i;
            }
          }
          const file = points[best].filename;
          rgb.set(palette[files.indexOf(file) % palette.length]);
          const o = (py * RES + px) * 4;
          image.data[o] = rgb.r * 255;
          image.data[o + 1] = rgb.g * 255;
          image.data[o + 2] = rgb.b * 255;
          image.data[o + 3] = 255;
        }
      }
      ctx.putImageData(image, 0, 0);

      const texture = new THREE.CanvasTexture(canvas);
      // NearestFilter, deliberately: the boundaries between territories are
      // the information. Linear filtering would blur them into gradients and
      // imply a soft transition that does not exist -- ownership flips at a
      // line.
      texture.magFilter = THREE.NearestFilter;
      texture.minFilter = THREE.LinearMipmapLinearFilter;

      territoryMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(extent * 2, extent * 2),
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          opacity: 0.4,
          depthWrite: false,
          fog: true,
        }),
      );
      territoryMesh.rotation.x = -Math.PI / 2;
      // Just under the LOWEST point rather than at a fixed height. A fixed
      // plane cut through the middle of the cloud, because the projection's
      // third axis is not a height and points sit on both sides of any plane
      // you pick -- so part of the corpus ended up underneath its own map.
      let lowest = 0;
      for (let i = 0; i < n; i++) lowest = Math.min(lowest, positions[i * 3 + 1]);
      territoryMesh.position.y = lowest - 0.35;
      territoryMesh.renderOrder = 0.2;
      territoryMesh.visible = false;
      scene.add(territoryMesh);

      // Drop-lines from each point straight down to the map.
      //
      // Without them the plane reads as an unrelated rectangle floating below
      // the cloud, and the whole claim -- that this is the corpus's SHADOW --
      // has to be taken on trust. The lines are the claim, drawn.
      const stemVerts = new Float32Array(n * 6);
      const stemColours = new Float32Array(n * 6);
      const stemColour = new THREE.Color();
      for (let i = 0; i < n; i++) {
        stemVerts[i * 6] = positions[i * 3];
        stemVerts[i * 6 + 1] = positions[i * 3 + 1];
        stemVerts[i * 6 + 2] = positions[i * 3 + 2];
        stemVerts[i * 6 + 3] = positions[i * 3];
        stemVerts[i * 6 + 4] = territoryMesh.position.y;
        stemVerts[i * 6 + 5] = positions[i * 3 + 2];
        stemColour.set(
          palette[files.indexOf(points[i].filename) % palette.length],
        );
        for (const end of [0, 3]) {
          stemColours[i * 6 + end] = stemColour.r;
          stemColours[i * 6 + end + 1] = stemColour.g;
          stemColours[i * 6 + end + 2] = stemColour.b;
        }
      }
      const stemGeometry = new THREE.BufferGeometry();
      stemGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(stemVerts, 3),
      );
      stemGeometry.setAttribute(
        "color",
        new THREE.BufferAttribute(stemColours, 3),
      );
      stems = new THREE.LineSegments(
        stemGeometry,
        new THREE.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity: 0.18,
          depthWrite: false,
          fog: true,
        }),
      );
      stems.renderOrder = 0.3;
      stems.visible = false;
      scene.add(stems);
    }

    // ---- camera state ---------------------------------------------------
    // Every value has a CURRENT and a DESIRED form, eased together each frame.
    // That is what makes zoom and fly-to continuous rather than stepped: input
    // moves the desired value and the camera chases it.
    const target = new THREE.Vector3();
    const wantTarget = new THREE.Vector3();
    let yaw = 0.8;
    let pitch = 0.45;
    let radius = 6.5;
    let wantYaw = yaw;
    let wantPitch = pitch;
    let wantRadius = radius;

    const place = () => {
      camera.position.set(
        target.x + radius * Math.cos(pitch) * Math.sin(yaw),
        target.y + radius * Math.sin(pitch),
        target.z + radius * Math.cos(pitch) * Math.cos(yaw),
      );
      camera.lookAt(target);
    };

    // ---- interaction ----------------------------------------------------
    let dragging = false;
    let panning = false;
    let moved = false;
    let lastX = 0;
    let lastY = 0;

    const pointer = new THREE.Vector2();
    const ray = new THREE.Raycaster();

    const pickAt = (clientX: number, clientY: number): number | null => {
      const rect = view.getBoundingClientRect();
      pointer.set(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      // Points have no surface area, so picking needs an explicit world-space
      // radius -- and it scales with distance, or a zoomed-out plot becomes
      // impossible to click.
      ray.params.Points = { threshold: 0.022 * radius };
      ray.setFromCamera(pointer, camera);
      const hits = ray.intersectObject(cloud);
      return hits.length ? hits[0].index ?? null : null;
    };

    const onDown = (e: PointerEvent) => {
      dragging = true;
      // Shift, middle or right button pans instead of orbiting. Pan is what
      // "fly somewhere else" needs; orbit alone can only circle one point.
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
        setHover(
          i === null
            ? null
            : { i, x: e.clientX - rect.left, y: e.clientY - rect.top },
        );
        return;
      }
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      // A few pixels of slop, so a click from an unsteady hand still selects
      // rather than being swallowed as a drag.
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      lastX = e.clientX;
      lastY = e.clientY;

      if (panning) {
        // Pan in the camera's own plane, scaled by distance, so a drag moves
        // the same amount of WORLD under the cursor at any zoom level.
        const right = new THREE.Vector3();
        const up = new THREE.Vector3();
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
      if (moved) return; // a manoeuvre, not a pick
      onSelectRef.current(pickAt(e.clientX, e.clientY));
    };

    const onLeave = () => setHover(null);

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // MULTIPLICATIVE, not additive. A fixed step per notch is enormous when
      // close and imperceptible when far, which is exactly the "zooms in
      // steps" feeling. Scaling by current distance makes every notch the same
      // proportional move, so zoom is continuous at any scale.
      wantRadius = Math.max(
        0.5,
        Math.min(18, wantRadius * Math.exp(e.deltaY * 0.0012)),
      );
    };

    const onContext = (e: Event) => e.preventDefault(); // right-drag pans

    // ON THE CANVAS, NOT THE CONTAINER, and this is what broke every overlay
    // button at once.
    //
    // The buttons are children of the container. With the listeners on the
    // container, pressing one ran `onDown` -> `setPointerCapture` on the
    // container, which retargets all subsequent pointer events to it -- so the
    // pointerup never reached the button and no `click` was ever synthesised.
    // Light, Reset and Expand all appeared dead for the same reason, and the
    // Fullscreen API was never the problem.
    //
    // The canvas is a sibling of the overlay, so capture on it cannot swallow
    // a press meant for a control.
    view.addEventListener("pointerdown", onDown);
    view.addEventListener("pointermove", onMove);
    view.addEventListener("pointerup", onUp);
    view.addEventListener("pointerleave", onLeave);
    view.addEventListener("wheel", onWheel, { passive: false });
    view.addEventListener("contextmenu", onContext);

    // ---- imperative API -------------------------------------------------
    api.current.highlight = (sel, hov) => {
      const attr = geometry.getAttribute("size") as THREE.BufferAttribute;
      for (let i = 0; i < n; i++) {
        attr.setX(i, i === sel ? SELECTED_SIZE : i === hov ? HOVER_SIZE : BASE_SIZE);
      }
      attr.needsUpdate = true;
    };

    api.current.flyTo = (filename) => {
      const attr = geometry.getAttribute("alpha") as THREE.BufferAttribute;
      if (!filename) {
        for (let i = 0; i < n; i++) attr.setX(i, 1);
        attr.needsUpdate = true;
        wantTarget.set(0, 0, 0);
        wantRadius = 6.5;
        return;
      }
      const box = new THREE.Box3();
      const v = new THREE.Vector3();
      for (let i = 0; i < n; i++) {
        const mine = points[i].filename === filename;
        // Faded, not hidden. Removing the others would lose the context that
        // makes "this document sits apart from everything else" visible.
        attr.setX(i, mine ? 1 : 0.12);
        if (mine) box.expandByPoint(v.fromArray(positions, i * 3));
      }
      attr.needsUpdate = true;
      if (box.isEmpty()) return;
      box.getCenter(wantTarget);
      // Frame the document's own extent, with a floor so a one-chunk file does
      // not fly the camera inside the point.
      wantRadius = Math.max(1.3, box.getSize(v).length() * 1.7);
    };

    api.current.reset = () => {
      wantYaw = 0.8;
      wantPitch = 0.45;
      api.current.flyTo(null);
    };

    api.current.setThreads = (on) => {
      threadGroup.visible = on;
    };
    api.current.setTerritory = (on) => {
      if (territoryMesh) territoryMesh.visible = on;
      if (stems) stems.visible = on;
    };

    // The scene is built once and the toggles live in React state, so a scene
    // rebuilt after a theme change would come back with both layers off while
    // the buttons still read "on". Applying the current values here keeps them
    // in step.
    threadGroup.visible = threadsRef.current;
    if (territoryMesh) territoryMesh.visible = territoryRef.current;
    if (stems) stems.visible = territoryRef.current;

    // ---- loop -----------------------------------------------------------
    const onResize = () => {
      if (!el.clientWidth) return;
      camera.aspect = el.clientWidth / Math.max(1, el.clientHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(el);

    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      // ONE easing constant for every axis, so a fly-to that changes target,
      // distance and angle at once arrives as a single movement rather than
      // three overlapping ones.
      const k = 0.11;
      yaw += (wantYaw - yaw) * k;
      pitch += (wantPitch - pitch) * k;
      radius += (wantRadius - radius) * k;
      target.lerp(wantTarget, k);
      place();
      renderer.render(scene, camera);
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
      // dispose(), not just removal: a WebGL context holds GPU buffers that
      // garbage collection will not reclaim, and browsers cap how many
      // contexts a page may hold. Without this, navigating back and forth
      // eventually blanks an earlier canvas.
      geometry.dispose();
      material.dispose();
      for (const line of threadGroup.children) {
        const l = line as THREE.Line;
        l.geometry.dispose();
        (l.material as THREE.Material).dispose();
      }
      backdrop.dispose();
      dustGeometry.dispose();
      (dust.material as THREE.Material).dispose();
      if (stems) {
        stems.geometry.dispose();
        (stems.material as THREE.Material).dispose();
      }
      if (territoryMesh) {
        territoryMesh.geometry.dispose();
        const m = territoryMesh.material as THREE.MeshBasicMaterial;
        // The CanvasTexture holds a 192x192 bitmap on the GPU; disposing the
        // material alone leaves it resident.
        m.map?.dispose();
        m.dispose();
      }
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
    // `theme` rebuilds the scene, which is correct and cheap: colours,
    // background, fog and grid all change together, and there are a few
    // hundred points.
  }, [points, theme]);

  // Selection, hover and focus are pushed imperatively. Including them in the
  // effect above would rebuild the WebGL context on every mouse move.
  useEffect(() => {
    api.current.highlight(selected, hover?.i ?? null);
  }, [selected, hover]);

  useEffect(() => {
    api.current.flyTo(focus);
  }, [focus]);

  useEffect(() => {
    const onChange = () =>
      setNativeFull(document.fullscreenElement === host.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  // Escape leaves the CSS overlay. The native API handles its own Escape, so
  // this only matters for the fallback -- and an expanded view with no way out
  // but a mouse is the kind of trap that makes people reload the page.
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

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
      // Attempted first: hiding the browser chrome is genuinely better when
      // the browser allows it.
      await host.current?.requestFullscreen();
    } catch {
      // Rejected -- Permissions-Policy, a transformed ancestor, or a browser
      // that will not grant it here. Fall back rather than doing nothing,
      // which is what the earlier version did, silently.
      setExpanded(true);
    }
  }, [expanded]);

  const hovered = hover ? points[hover.i] : null;
  const chosen = selected !== null ? points[selected] : null;
  const ink = INK[theme];
  const palette = PALETTES[theme];
  const panel = theme === "dark" ? "rgba(16,18,26,0.94)" : "rgba(255,255,255,0.95)";

  return (
    <div
      ref={host}
      className={
        big
          ? "fixed inset-0 z-[60] w-full touch-none overflow-hidden"
          : "relative h-[30rem] w-full touch-none overflow-hidden rounded-[var(--md-shape-lg)]"
      }
      style={{ background: GROUND[theme], cursor: hovered ? "pointer" : "grab" }}
    >
      {/* Overlay controls live INSIDE the container so they survive both
          expansion modes -- a button outside it vanishes the moment the
          element fills the screen. */}
      <div className="pointer-events-none absolute right-3 top-3 z-10 flex gap-2">
        <button
          type="button"
          className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs"
          style={{ background: ink.chip, color: ink.text }}
          onClick={() => onThemeChange(theme === "dark" ? "light" : "dark")}
        >
          {theme === "dark" ? "Light" : "Dark"}
        </button>
        <button
          type="button"
          className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs"
          style={{
            // `chip` is a 12%-opacity OVERLAY colour, not a foreground. Using
            // it as the active label made the text all but invisible against
            // its own inverted background.
            background: threads ? ink.on : ink.chip,
            color: threads ? ink.onText : ink.text,
          }}
          onClick={() => setThreads((v) => !v)}
          title="Join each document's chunks in reading order"
        >
          Threads
        </button>
        <button
          type="button"
          className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs"
          style={{
            // `chip` is a 12%-opacity OVERLAY colour, not a foreground. Using
            // it as the active label made the text all but invisible against
            // its own inverted background.
            background: territory ? ink.on : ink.chip,
            color: territory ? ink.onText : ink.text,
          }}
          onClick={() => setTerritory((v) => !v)}
          title="Colour the floor by which document owns each patch of ground"
        >
          Territory
        </button>
        <button
          type="button"
          className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs"
          style={{ background: ink.chip, color: ink.text }}
          onClick={() => api.current.reset()}
        >
          Reset view
        </button>
        <button
          type="button"
          className="pointer-events-auto rounded-[var(--md-shape-sm)] px-2.5 py-1.5 text-xs"
          style={{ background: ink.chip, color: ink.text }}
          onClick={() => void toggleBig()}
        >
          {big ? "Exit" : "Expand"}
        </button>
      </div>

      <p
        className="pointer-events-none absolute bottom-3 left-3 z-10 text-xs"
        style={{ color: ink.faint }}
      >
        drag to orbit · shift-drag or right-drag to pan · scroll to zoom
        {big ? " · Esc to exit" : ""}
        {threads && " · threads join chunks in reading order"}
        {/* Stated plainly, because the floor looks like a map of the space and
            is not: it partitions the plot's shadow, so two chunks stacked
            vertically collapse onto one another. */}
        {territory && " · territory is the floor's shadow, height ignored"}
      </p>

      {/* EXPANDED ONLY. Out of fullscreen the page already carries a legend
          and a selection panel; in fullscreen the canvas is the whole window,
          so anything outside it is simply gone. */}
      {big && (
        <div
          className="pointer-events-none absolute left-3 top-3 z-10 flex max-h-[80%] w-56 flex-col gap-1 overflow-y-auto rounded-[var(--md-shape-md)] p-2"
          style={{ background: panel, color: ink.text }}
        >
          {files.map((f, i) => {
            const on = focus === f;
            return (
              <button
                key={f}
                type="button"
                onClick={() => onFocusChange(on ? null : f)}
                className="pointer-events-auto flex items-center gap-2 rounded-[var(--md-shape-sm)] px-2 py-1 text-left text-xs"
                style={{ background: on ? ink.chip : "transparent" }}
                aria-pressed={on}
              >
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ background: palette[i % palette.length] }}
                />
                <span className="min-w-0 flex-1 truncate">{f}</span>
                <span style={{ opacity: 0.55 }}>
                  {points.filter((p) => p.filename === f).length}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {big && chosen && (
        <div
          className="absolute bottom-3 right-3 top-14 z-10 flex w-[26rem] max-w-[40vw] flex-col rounded-[var(--md-shape-md)]"
          style={{ background: panel, color: ink.text }}
        >
          <div className="flex items-start gap-2 p-4 pb-2">
            <div className="min-w-0 flex-1">
              <p className="break-words text-sm font-medium">{chosen.filename}</p>
              <p className="mt-0.5 text-xs" style={{ color: ink.faint }}>
                {chosen.heading ?? "no heading"} · chunk {chosen.chunk_index} ·{" "}
                {chosen.n_chars} chars
              </p>
            </div>
            <button
              type="button"
              onClick={() => onSelect(null)}
              className="rounded-[var(--md-shape-sm)] px-2 py-1 text-xs"
              style={{ background: ink.chip, color: ink.text }}
            >
              Close
            </button>
          </div>

          {/* The whole passage, scrollable. A tall card rather than a tooltip
              because the point of clicking is to READ the chunk -- 160
              characters of preview is enough to recognise one and not enough
              to judge whether it was chunked sensibly. */}
          <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-2">
            <p className="whitespace-pre-wrap text-sm leading-relaxed">
              {chosen.text}
            </p>
          </div>

          {chosen.nearest && (
            <div
              className="m-4 mt-2 rounded-[var(--md-shape-sm)] p-2 text-xs"
              style={{ background: ink.chip }}
            >
              nearest {chosen.nearest.score} · {chosen.nearest.filename} chunk{" "}
              {chosen.nearest.chunk_index}
              {chosen.nearest.score >= 0.95 && " — near-duplicate"}
            </div>
          )}
        </div>
      )}

      {hovered && hover && (
        <div
          className="pointer-events-none absolute z-20 w-[19rem] rounded-[var(--md-shape-md)] p-3"
          style={{
            // Offset from the cursor, and clamped to the canvas so the tooltip
            // never covers the point it describes or leaves the viewport.
            left: Math.max(
              8,
              Math.min(hover.x + 16, (host.current?.clientWidth ?? 0) - 320),
            ),
            top: Math.min(hover.y + 16, (host.current?.clientHeight ?? 0) - 150),
            background: theme === "dark" ? "rgba(16,18,26,0.96)" : "rgba(255,255,255,0.97)",
            color: ink.text,
            border: `1px solid ${ink.chip}`,
          }}
        >
          <p className="break-words text-xs font-medium">{hovered.filename}</p>
          <p className="mt-0.5 text-xs" style={{ color: ink.faint }}>
            {hovered.heading ?? "no heading"} · chunk {hovered.chunk_index} ·{" "}
            {hovered.n_chars} chars
          </p>
          <p className="mt-1.5 text-xs leading-snug">{hovered.preview}…</p>
          {hovered.nearest && (
            <p className="mt-1.5 text-xs" style={{ color: ink.faint }}>
              nearest {hovered.nearest.score} · {hovered.nearest.filename} chunk{" "}
              {hovered.nearest.chunk_index}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Point sizes, before the 1/depth divide and the clamp in the shader.
 *
 * The first values were a third of these, which at the default camera distance
 * worked out to about four pixels -- the "I can't see some of the dots"
 * problem. These give roughly a ten-pixel dot at the default framing.
 */
const BASE_SIZE = 70;
const HOVER_SIZE = 115;
const SELECTED_SIZE = 165;

/**
 * Where the fog starts and finishes, in world units.
 *
 * Doing double duty: depth cue for the points, and the horizon for the floor.
 * The far value has to exceed the maximum camera distance (18) or the scene
 * disappears entirely when zoomed out -- which is how the first version ended
 * up showing a black void.
 */
const FOG_NEAR = 7;
const FOG_FAR = 34;

