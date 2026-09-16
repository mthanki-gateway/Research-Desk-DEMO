"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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

/** The plot's own ground, deliberately independent of the app theme. */
const GROUND: Record<PlotTheme, string> = {
  dark: "#0e1016",
  light: "#eef0f6",
};

/**
 * Grid line colours: [centre axes, everything else].
 *
 * Considerably brighter than the first pass, which used near-background greys
 * and effectively drew an invisible floor -- the plot read as points in a void
 * however large the grid was. Legibility of the ground plane is the whole
 * reason it exists: it is what tells you which way is down while you are
 * moving, so it has to be seen rather than merely present.
 */
const GRID: Record<PlotTheme, [number, number]> = {
  dark: [0x7b86a8, 0x4a5270],
  light: [0x8c96b4, 0xb6bed2],
};

/** Overlay text and chrome, which must sit on whichever ground is active. */
const INK: Record<PlotTheme, { text: string; faint: string; chip: string }> = {
  dark: {
    text: "#e8e9f0",
    faint: "rgba(232,233,240,0.6)",
    chip: "rgba(255,255,255,0.12)",
  },
  light: {
    text: "#16181f",
    faint: "rgba(22,24,31,0.6)",
    chip: "rgba(0,0,0,0.07)",
  },
};

type Props = {
  points: AtlasPoint[];
  selected: number | null;
  onSelect: (index: number | null) => void;
  /** Filename to fly to, or null for the whole corpus. */
  focus: string | null;
  theme: PlotTheme;
  onThemeChange: (theme: PlotTheme) => void;
};

export default function Scatter({
  points,
  selected,
  onSelect,
  focus,
  theme,
  onThemeChange,
}: Props) {
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

  // Handlers change identity every render; the scene is built once. Refs keep
  // the effect from tearing down the whole WebGL context on a parent render.
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // Imperative hooks into the live scene, assigned when it is built.
  const api = useRef<{
    highlight: (sel: number | null, hovered: number | null) => void;
    flyTo: (filename: string | null) => void;
    reset: () => void;
  }>({ highlight: () => {}, flyTo: () => {}, reset: () => {} });

  useEffect(() => {
    const el = host.current;
    if (!el || points.length === 0) return;

    const ground = new THREE.Color(GROUND[theme]);
    const scene = new THREE.Scene();
    scene.background = ground;
    // Fog fades distant points, which is most of what makes a flat scatter
    // read as a volume you are inside rather than a picture you look at.
    // Pushed well out from the original 4-13. Fog is what makes the floor
    // read as endless -- the grid fades into the ground colour before its edge
    // is ever reached -- but at 13 units it also swallowed the points
    // themselves at full zoom-out, leaving a black void with nothing in it.
    scene.fog = new THREE.Fog(ground, FOG_NEAR, FOG_FAR);

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
      positions[i * 3] = p.x * 2.4;
      positions[i * 3 + 1] = p.y * 2.4;
      positions[i * 3 + 2] = p.z * 2.4;
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
    scene.add(cloud);

    // A floor grid rather than a wireframe box. The box implied the data had
    // edges, which it does not; a ground plane gives orientation while moving
    // without claiming anything about the data's extent.
    //
    // FAR LARGER THAN THE VIEW, and that is the trick. The 10-unit grid ended
    // in mid-air: you flew past its edge and the world became a black void.
    // At 220 units the edge sits far beyond the fog, so the lines simply fade
    // into the ground colour and the floor reads as endless -- which is what
    // "infinite grid" shaders achieve, without the shader.
    //
    // One line per unit keeps the cell size meaningful: the data spans about
    // five units, so a square is a fifth of the corpus wide.
    const grid = new THREE.GridHelper(GRID_SPAN, GRID_SPAN, GRID[theme][0], GRID[theme][1]);
    // Closer under the data than before. At -2.4 the floor sat far enough
    // below the cloud that a default view showed mostly empty space between
    // the two, which read as the points floating in nothing.
    grid.position.y = -1.8;
    const gridMaterial = grid.material as THREE.Material;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.85;
    // Explicit, though it is the default: without fog on this material the
    // grid would stay crisp to the horizon and look like graph paper rather
    // than a receding plane.
    (gridMaterial as THREE.LineBasicMaterial).fog = true;
    scene.add(grid);

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
      grid.geometry.dispose();
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
  const ink = INK[theme];

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
      </p>

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

/**
 * The floor's extent. Deliberately far beyond FOG_FAR so its edge is never
 * reachable: the grid fades out rather than stopping.
 */
const GRID_SPAN = 220;
