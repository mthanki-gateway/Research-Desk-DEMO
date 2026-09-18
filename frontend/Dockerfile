# syntax=docker/dockerfile:1
FROM node:22-alpine AS base
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json* ./
# No lockfile on the first build, so fall back to `install`.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY tsconfig.json next.config.ts postcss.config.mjs ./
# `predev` and `prebuild` both run scripts/copyOrt.mjs, which lifts
# onnxruntime-web's WASM out of node_modules into public/ort. Without this the
# container has the npm scripts but not the file they call.
COPY scripts ./scripts

# --- dev: app/ and lib/ are bind-mounted by compose --------------------------
FROM base AS dev
EXPOSE 3000
# --webpack, not the default Turbopack. Turbopack gets no inotify events across
# a Windows bind mount and has no polling escape hatch, so edits are silently
# ignored until a container restart. Webpack honours WATCHPACK_POLLING (set in
# compose) and does detect them. For Turbopack speed, run the frontend natively
# on the host instead -- see README "Can I skip Docker".
CMD ["npm", "run", "dev", "--", "--webpack"]

# --- prod: only used if you self-host instead of deploying to Vercel ---------
FROM base AS build
COPY app ./app
COPY lib ./lib
# `public` carries the turn-detection models (~11MB of ONNX). It was missing
# here, which broke this stage in two ways at once: the build produced no
# `/app/public` for the prod stage to copy on line 33, and any page fetching
# `/models/*.onnx` got a 404 from a container that had never seen the files.
COPY public ./public
RUN npm run build

FROM node:22-alpine AS prod
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
