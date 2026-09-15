"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { AtlasPoint } from "@/lib/api";

/**
 * The corpus as a 3D point cloud.
 *
 * RAW three.js, NOT react-three-fiber. Fiber declares a peer range of
 * `react >=19 <19.3` and this project is on 19.3, so installing it means
 * overriding a constraint its authors set deliberately. For a scatter plot
 * with orbit controls the React binding buys very little anyway: there is one
 * scene, built once, and none of it belongs in React's render cycle -- a
 * component that re-renders sixty times a second to spin a camera is the
 * thing to avoid, not the thing to build.
 *
 * So the whole scene lives in one effect keyed on the data, and React only
 * owns the container element.
 */

/** Distinct hues per document. Chosen for separability rather than beauty --
 *  the entire point is telling one file's chunks from another's at a glance. */
const PALETTE = [
  0x6750a4, // primary-ish
  0x2e7d32,
  0xc62828,
  0xef6c00,
  0x0277bd,
  0x00838f,
  0x6a1b9a,
  0x827717,
];

export default function Scatter({
  points,
  selected,
  onSelect,
}: {
  points: AtlasPoint[];
  selected: number | null;
  onSelect: (index: number | null) => void;
}) {
  const host = useRef<HTMLDivElement | null>(null);
  // The click handler changes identity on every render; the scene is built
  // once. A ref keeps the effect from tearing down and rebuilding the whole
  // WebGL context every time the parent re-renders.
  const onSelectRef = useRef(onSelect);
  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);
  const highlight = useRef<(index: number | null) => void>(() => {});

  useEffect(() => {
    const el = host.current;
    if (!el || points.length === 0) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      55,
      el.clientWidth / Math.max(1, el.clientHeight),
      0.01,
      100,
    );
    camera.position.set(2.2, 1.6, 2.2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);

    // --- geometry ---------------------------------------------------------
    const files = [...new Set(points.map((p) => p.filename))].sort();
    const positions = new Float32Array(points.length * 3);
    const colours = new Float32Array(points.length * 3);
    const sizes = new Float32Array(points.length);
    const base = new THREE.Color();

    points.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
      base.setHex(PALETTE[files.indexOf(p.filename) % PALETTE.length]);
      colours[i * 3] = base.r;
      colours[i * 3 + 1] = base.g;
      colours[i * 3 + 2] = base.b;
      sizes[i] = 14;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    // A shader rather than PointsMaterial, for one reason: per-point SIZE.
    // PointsMaterial takes a single size for the whole cloud, so the selected
    // chunk could only be distinguished by colour -- and colour already means
    // "which document".
    const material = new THREE.ShaderMaterial({
      transparent: true,
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          // Divided by depth so points shrink with distance, which is what
          // makes a flat cloud of dots read as a 3D volume at all.
          gl_PointSize = size * (1.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          // Discard outside the unit circle: gl_PointCoord is a square, and
          // without this every point is a visible box.
          vec2 d = gl_PointCoord - vec2(0.5);
          if (dot(d, d) > 0.25) discard;
          gl_FragColor = vec4(vColor, 0.95);
        }
      `,
      vertexColors: true,
    });

    const cloud = new THREE.Points(geometry, material);
    scene.add(cloud);

    // A faint box, so rotation has something to be relative to. Without a
    // reference the cloud appears to wobble rather than turn.
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
      new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.18 }),
    );
    scene.add(box);

    highlight.current = (index) => {
      const attr = geometry.getAttribute("size") as THREE.BufferAttribute;
      for (let i = 0; i < points.length; i++) attr.setX(i, i === index ? 34 : 14);
      attr.needsUpdate = true;
    };

    // --- interaction ------------------------------------------------------
    // Hand-rolled orbit rather than three's OrbitControls: the example addon
    // is an extra import path that has moved between versions, and "drag to
    // rotate, wheel to zoom" is a dozen lines.
    let dragging = false;
    let moved = false;
    let lastX = 0;
    let lastY = 0;
    let yaw = 0.7;
    let pitch = 0.5;
    let radius = 3.4;

    const place = () => {
      camera.position.set(
        radius * Math.cos(pitch) * Math.sin(yaw),
        radius * Math.sin(pitch),
        radius * Math.cos(pitch) * Math.cos(yaw),
      );
      camera.lookAt(0, 0, 0);
    };
    place();

    const onDown = (e: PointerEvent) => {
      dragging = true;
      moved = false;
      lastX = e.clientX;
      lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      // A few pixels of slop, so a click with a trembling hand still selects
      // rather than being swallowed as a drag.
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      lastX = e.clientX;
      lastY = e.clientY;
      yaw -= dx * 0.005;
      pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.005));
      place();
    };
    const onUp = (e: PointerEvent) => {
      dragging = false;
      el.releasePointerCapture(e.pointerId);
      if (moved) return; // a rotate, not a pick

      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      const ray = new THREE.Raycaster();
      // Points have no surface area to hit, so the picker needs an explicit
      // radius in world units. Too small and the plot feels broken; this is
      // roughly the on-screen dot.
      ray.params.Points = { threshold: 0.055 };
      ray.setFromCamera(ndc, camera);
      const hits = ray.intersectObject(cloud);
      onSelectRef.current(hits.length ? (hits[0].index ?? null) : null);
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      radius = Math.max(1.4, Math.min(8, radius + e.deltaY * 0.002));
      place();
    };

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });

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
      renderer.render(scene, camera);
    };
    loop();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("wheel", onWheel);
      // dispose(), not just removal: a WebGL context holds GPU buffers that
      // garbage collection will not reclaim, and browsers cap how many
      // contexts a page may have. Navigating away repeatedly without this
      // eventually loses the oldest context and blanks an earlier canvas.
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [points]);

  // Selection is pushed imperatively rather than rebuilding the scene: the
  // effect above is keyed on `points`, and including `selected` would tear
  // down and recreate the WebGL context on every click.
  useEffect(() => {
    highlight.current(selected);
  }, [selected]);

  return (
    <div
      ref={host}
      className="h-[26rem] w-full cursor-grab touch-none rounded-[var(--md-shape-lg)] active:cursor-grabbing"
      style={{ background: "var(--md-surface-container-high)" }}
    />
  );
}
