"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { QueryRay } from "@/lib/api";
import {
  BACKDROP,
  GROUND,
  INK,
  PALETTES,
  SPREAD,
  backdropTexture,
  type PlotTheme,
} from "../atlas/scatter";

/**
 * Where one question landed in the corpus, and what it pulled back.
 *
 * WHY THIS IS A PICTURE AND NOT A TABLE
 *
 * A retrieval score of 0.68 looks exactly the same in two situations that
 * could not be more different. In the first the query sat inside a dense
 * cluster and every result is a neighbour of every other: retrieval worked,
 * and the answer is probably good. In the second the query sat in empty space
 * between three documents and dragged one straggler out of each: the model was
 * handed three unrelated passages and will confidently stitch them together.
 * The scores are identical. The geometry is not, and it is obvious at a glance.
 *
 * The other thing only visible here is the near miss. A chunk that scored just
 * below the cutoff and a chunk on the far side of the corpus are both simply
 * absent from the results — no log distinguishes them. Drawn as dashed lines,
 * they are the recall failures, sitting right next to the query.
 *
 * A SEPARATE SCENE FROM THE ATLAS, deliberately. The Atlas is a place to fly
 * around and explore; this is a single fact, read in a few seconds and closed.
 * Sharing the component would mean threading query state through every one of
 * the Atlas's flight, legend and fullscreen paths to serve a dialog that needs
 * none of them. It shares the things that must not diverge — the palette, the
 * spread factor, the ground — and nothing else.
 */

type Props = {
  data: QueryRay;
  theme: PlotTheme;
};

/** The query marker and its rays, in each ground's contrast direction. */
const QUERY_INK: Record<PlotTheme, { query: number; ray: number; miss: number }> =
  {
    dark: { query: 0xffffff, ray: 0xffd166, miss: 0x8b93ad },
    light: { query: 0x111318, ray: 0xb45309, miss: 0x9aa2b8 },
  };

export default function RayView({ data, theme }: Props) {
  const host = useRef<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(
    null,
  );

  const retrieved = useMemo(
    () => new Set(data.rays.map((r) => r.index)),
    [data.rays],
  );
  const missed = useMemo(
    () => new Set(data.near_misses.map((m) => m.index)),
    [data.near_misses],
  );

  useEffect(() => {
    const el = host.current;
    const query = data.query;
    if (!el || !query || data.points.length === 0) return;

    const scene = new THREE.Scene();
    // Same vignette as the Atlas. A flat field gives the eye no centre and no
    // depth, and this view is read in a few seconds -- it has to be legible
    // immediately or not at all.
    const backdrop = backdropTexture(theme);
    scene.background = backdrop;
    // Fogged to the vignette's edge, so a distant point fades into the shade
    // already behind it.
    scene.fog = new THREE.Fog(new THREE.Color(BACKDROP[theme].edge), 6, 26);

    const camera = new THREE.PerspectiveCamera(
      50,
      el.clientWidth / Math.max(1, el.clientHeight),
      0.01,
      100,
    );
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);
    const view = renderer.domElement;

    const at = (i: number) =>
      new THREE.Vector3(
        data.points[i].x * SPREAD,
        data.points[i].y * SPREAD,
        data.points[i].z * SPREAD,
      );
    const queryAt = new THREE.Vector3(
      query.x * SPREAD,
      query.y * SPREAD,
      query.z * SPREAD,
    );

    // ---- the corpus, as context ------------------------------------------
    // Everything not retrieved is drawn small and faint. It is not the subject;
    // it is the shape the query landed in, and at full strength it would bury
    // the dozen points that matter.
    const files = [...new Set(data.points.map((p) => p.filename))].sort();
    const palette = PALETTES[theme];
    const n = data.points.length;
    const positions = new Float32Array(n * 3);
    const colours = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
    const alphas = new Float32Array(n);
    const colour = new THREE.Color();

    data.points.forEach((p, i) => {
      const v = at(i);
      positions[i * 3] = v.x;
      positions[i * 3 + 1] = v.y;
      positions[i * 3 + 2] = v.z;
      colour.set(palette[files.indexOf(p.filename) % palette.length]);
      colours[i * 3] = colour.r;
      colours[i * 3 + 1] = colour.g;
      colours[i * 3 + 2] = colour.b;
      const hit = retrieved.has(i);
      const near = missed.has(i);
      sizes[i] = hit ? 110 : near ? 80 : 46;
      alphas[i] = hit ? 1 : near ? 0.8 : 0.22;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("colour", new THREE.BufferAttribute(colours, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("alpha", new THREE.BufferAttribute(alphas, 1));

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      vertexShader: `
        attribute vec3 colour;
        attribute float size;
        attribute float alpha;
        varying vec3 vColour;
        varying float vAlpha;
        void main() {
          vColour = colour;
          vAlpha = alpha;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = clamp(size / -mv.z, 5.0, 90.0);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying vec3 vColour;
        varying float vAlpha;
        void main() {
          // Soft falloff rather than a hard disc: reads as a glow, and the
          // blur is what stops a few dozen points looking like confetti.
          float d = length(gl_PointCoord - vec2(0.5));
          if (d > 0.5) discard;
          float edge = smoothstep(0.5, 0.12, d);
          gl_FragColor = vec4(vColour, vAlpha * edge);
        }
      `,
    });
    const cloud = new THREE.Points(geometry, material);
    // Both materials are transparent, so three.js sorts them by object centre
    // — which flips as the camera moves. Pinned, or the lines flicker in front
    // of and behind the points depending on the angle.
    cloud.renderOrder = 2;
    scene.add(cloud);

    // ---- the rays --------------------------------------------------------
    const ink = QUERY_INK[theme];
    const rayPositions: number[] = [];
    for (const r of data.rays) {
      const v = at(r.index);
      rayPositions.push(queryAt.x, queryAt.y, queryAt.z, v.x, v.y, v.z);
    }
    const rayGeometry = new THREE.BufferGeometry();
    rayGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(rayPositions, 3),
    );
    const rayLines = new THREE.LineSegments(
      rayGeometry,
      new THREE.LineBasicMaterial({
        color: ink.ray,
        transparent: true,
        opacity: 0.85,
      }),
    );
    rayLines.renderOrder = 1;
    scene.add(rayLines);

    // ---- the near misses -------------------------------------------------
    // Dashed, so they cannot be mistaken for results. These are the chunks the
    // query was close to and did NOT get.
    const missPositions: number[] = [];
    for (const m of data.near_misses) {
      const v = at(m.index);
      missPositions.push(queryAt.x, queryAt.y, queryAt.z, v.x, v.y, v.z);
    }
    const missGeometry = new THREE.BufferGeometry();
    missGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(missPositions, 3),
    );
    const missLines = new THREE.LineSegments(
      missGeometry,
      new THREE.LineDashedMaterial({
        color: ink.miss,
        transparent: true,
        opacity: 0.55,
        dashSize: 0.08,
        gapSize: 0.06,
      }),
    );
    // Required for LineDashedMaterial: without it every dash length is zero
    // and the line renders solid, which would make a miss look like a hit.
    missLines.computeLineDistances();
    missLines.renderOrder = 1;
    scene.add(missLines);

    // ---- the query itself ------------------------------------------------
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 20, 20),
      new THREE.MeshBasicMaterial({ color: ink.query }),
    );
    marker.position.copy(queryAt);
    marker.renderOrder = 3;
    scene.add(marker);

    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(0.11, 20, 20),
      new THREE.MeshBasicMaterial({
        color: ink.query,
        transparent: true,
        opacity: 0.18,
      }),
    );
    halo.position.copy(queryAt);
    scene.add(halo);

    // ---- camera ----------------------------------------------------------
    // The Atlas's camera model, spelled the same way ON PURPOSE.
    //
    // This started with its own spherical parameterisation -- x from
    // cos(theta), z from sin(theta), where the Atlas uses sin for x and cos
    // for z. Swapping which function feeds which axis is a REFLECTION, so with
    // both subtracting on drag the two views orbited in opposite directions.
    // Nothing looked wrong in either one alone; it was only wrong next to the
    // other. Two plots of the same data that answer a drag differently is a
    // bug however defensible each half is, so this now matches term for term.
    //
    // Framed on the QUERY rather than on the corpus centroid: the subject is
    // where this question landed, and centring the cloud would push it off to
    // one side exactly when it landed somewhere unusual.
    let yaw = 0.8;
    let pitch = 0.45;
    let radius = 5.2;
    let wantYaw = yaw;
    let wantPitch = pitch;
    let wantRadius = radius;
    const target = queryAt.clone();
    const wantTarget = queryAt.clone();

    const place = () => {
      camera.position.set(
        target.x + radius * Math.cos(pitch) * Math.sin(yaw),
        target.y + radius * Math.sin(pitch),
        target.z + radius * Math.cos(pitch) * Math.cos(yaw),
      );
      camera.lookAt(target);
    };
    place();

    // ---- interaction -----------------------------------------------------
    // Listeners on the CANVAS, not the container: pointer capture on the
    // container swallows clicks meant for chrome sitting on top of it.
    let dragging = false;
    let panning = false;
    let lastX = 0;
    let lastY = 0;

    const onDown = (e: PointerEvent) => {
      dragging = true;
      // Shift, middle or right button pans instead of orbiting, as in the
      // Atlas. Muscle memory should carry between the two.
      panning = e.shiftKey || e.button === 1 || e.button === 2;
      lastX = e.clientX;
      lastY = e.clientY;
      view.setPointerCapture(e.pointerId);
    };

    const onUp = (e: PointerEvent) => {
      dragging = false;
      panning = false;
      if (view.hasPointerCapture(e.pointerId))
        view.releasePointerCapture(e.pointerId);
    };

    const onMove = (e: PointerEvent) => {
      if (dragging) {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        if (panning) {
          // Pan in the camera's own plane, scaled by distance, so a drag moves
          // the same amount of WORLD under the cursor at any zoom.
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
        return;
      }
      // Hover picking, in screen space. A raycaster against a Points cloud
      // needs a threshold tuned per zoom level; projecting the points and
      // measuring pixels is both simpler and behaves the same at every zoom.
      const rect = view.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let best: number | null = null;
      let bestDist = 18;
      const v = new THREE.Vector3();
      for (let i = 0; i < n; i++) {
        // Only retrieved points and near misses are hoverable. The faint
        // background is context, and making it pickable means the tooltip
        // fires constantly on things the reader did not point at.
        if (!retrieved.has(i) && !missed.has(i)) continue;
        v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        v.project(camera);
        if (v.z > 1) continue;
        const sx = ((v.x + 1) / 2) * rect.width;
        const sy = ((1 - v.y) / 2) * rect.height;
        const d = Math.hypot(sx - mx, sy - my);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      }
      setHover(best === null ? null : { i: best, x: mx, y: my });
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      // MULTIPLICATIVE, as in the Atlas: a fixed step per notch is enormous
      // when close and imperceptible when far.
      wantRadius = Math.max(
        0.5,
        Math.min(18, wantRadius * Math.exp(e.deltaY * 0.0012)),
      );
    };

    const onLeave = () => setHover(null);
    const onContext = (e: Event) => e.preventDefault(); // right-drag pans

    view.addEventListener("pointerdown", onDown);
    view.addEventListener("pointerup", onUp);
    view.addEventListener("pointermove", onMove);
    view.addEventListener("pointerleave", onLeave);
    view.addEventListener("wheel", onWheel, { passive: false });
    view.addEventListener("contextmenu", onContext);

    let frame = 0;
    const loop = () => {
      frame = requestAnimationFrame(loop);
      // ONE easing constant for every axis, so a drag and a zoom arriving
      // together read as a single movement. Same value as the Atlas.
      const k = 0.11;
      yaw += (wantYaw - yaw) * k;
      pitch += (wantPitch - pitch) * k;
      radius += (wantRadius - radius) * k;
      target.lerp(wantTarget, k);
      place();
      renderer.render(scene, camera);
    };
    loop();

    const resize = new ResizeObserver(() => {
      if (!el.clientWidth || !el.clientHeight) return;
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    });
    resize.observe(el);

    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      view.removeEventListener("pointerdown", onDown);
      view.removeEventListener("pointerup", onUp);
      view.removeEventListener("pointermove", onMove);
      view.removeEventListener("pointerleave", onLeave);
      view.removeEventListener("wheel", onWheel);
      view.removeEventListener("contextmenu", onContext);
      geometry.dispose();
      material.dispose();
      rayGeometry.dispose();
      missGeometry.dispose();
      backdrop.dispose();
      renderer.dispose();
      el.removeChild(view);
    };
  }, [data, theme, retrieved, missed]);

  const ink = INK[theme];
  const point = hover ? data.points[hover.i] : null;
  const rank = hover ? data.rays.find((r) => r.index === hover.i) : undefined;

  return (
    <div
      className="relative h-full w-full touch-none overflow-hidden rounded-[var(--md-shape-md)]"
      // Grab, and a pointer over a hoverable point -- the Atlas's cursors. A
      // canvas you can drag has to say so; the default arrow reads as inert.
      style={{
        background: GROUND[theme],
        cursor: hover ? "pointer" : "grab",
      }}
    >
      <div ref={host} className="h-full w-full" />

      {point && hover && (
        <div
          className="pointer-events-none absolute z-10 max-w-[18rem] rounded-[var(--md-shape-sm)] px-3 py-2"
          style={{
            left: Math.min(hover.x + 12, 320),
            top: hover.y + 12,
            background: ink.chip,
            color: ink.text,
          }}
        >
          <p className="md-label-medium truncate">{point.filename}</p>
          <p className="md-body-small mt-0.5" style={{ color: ink.faint }}>
            {rank
              ? `rank ${rank.rank} · score ${rank.score.toFixed(3)}`
              : "not retrieved"}
            {" · chunk "}
            {point.chunk_index}
          </p>
          <p className="md-body-small mt-1 line-clamp-3">{point.preview}</p>
        </div>
      )}

      <div
        className="md-body-small pointer-events-none absolute bottom-2 left-3 right-3 flex flex-wrap gap-x-4 gap-y-1"
        style={{ color: ink.faint }}
      >
        <span>
          <span style={{ color: ink.text }}>●</span> query
        </span>
        <span>— retrieved ({data.rays.length})</span>
        <span>┄ near miss, not retrieved ({data.near_misses.length})</span>
        {data.dropped > 0 && <span>{data.dropped} source(s) not in the corpus</span>}
        <span className="ml-auto">
          drag to orbit · shift-drag to pan · scroll to zoom
        </span>
      </div>
    </div>
  );
}
