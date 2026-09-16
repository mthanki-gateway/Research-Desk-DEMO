# Project conventions

Rules that apply to this repo. Personal/global preferences live in
`~/.claude/CLAUDE.md`; this file is about the code.

## Frontend

### Clickable things show a pointer cursor

Anything a user can click, tap or activate uses `cursor: pointer` — buttons,
dropdowns (`<select>`), nav items, cards that navigate, chips, tabs, custom
controls built from `<div role="button">`. Disabled controls use
`cursor: not-allowed`.

This is handled **once** in `@layer base` in `frontend/app/globals.css`, not
per component. It covers native controls (`button`, `select`, `summary`,
`a[href]`, `label[for]`) and the interactive ARIA roles — `button`, `link`,
`tab`, `option`, `checkbox`, `switch`.

So: **a clickable `<div>` gets `role="button"`, not a `cursor-pointer`
utility.** It needs the role for keyboard and screen-reader users anyway, and
the cursor then follows for free. If something has the wrong cursor, the base
rule is missing a selector — fix it there.

The one legitimate use of the utility is a cursor that depends on STATE rather
than on what the element is, e.g. a row that is only clickable once its
document has finished indexing (`library/page.tsx`). A few older call sites
still carry a redundant `cursor-pointer` next to a role the base rule already
covers; harmless, and not worth a sweep.

**Why the rule exists:** Tailwind v4 removed the `button { cursor: pointer }`
reset that v3 shipped, so a bare `<button>` gets the arrow cursor. Every
component class here had been restoring it by hand, which worked until
something was built from `.md-state` plus utilities alone — the app switcher
and the clarify options both shipped with the wrong cursor, and neither looked
broken in a screenshot.

### Material 3

The UI is Material 3. Reuse what is in `frontend/app/md.tsx` (`Button`,
`Dialog`, `TextArea`, `Chip`, `Ripplable`, …) and the `--md-*` tokens in
`globals.css` rather than writing new one-off styles. Colour roles come in
pairs — use `--md-on-x` with `--md-x` so contrast holds in every theme.

### One stack, several apps

Apps are registered in `frontend/app/projects.ts`. Adding one is a single
entry there plus its pages; the drawer, the app switcher and active-item
highlighting all read from it. Do not hardcode nav items anywhere else.

Three today — **Research Desk** (chat, library, lab, atlas), **Model Lab**
(playground, transcribe) and **Earshot** (voice). Earshot is a different DOOR
onto the Research Desk's agent, not a second assistant: same corpus, same
graph, same tools, same web search. A document uploaded in the Library is
answerable by voice the moment it finishes indexing. If you find yourself
adding a second index or a parallel prompt stack for it, that is the mistake.

### Adding a frontend dependency needs an image rebuild

`docker-compose.yml` mounts only `frontend/app` and `frontend/lib`.
`package.json` and `node_modules` live **inside** the `web` image, so an
`npm install` on the host never reaches the running container — the dev server
keeps resolving against the image it was built from and reports
`Module not found`.

```powershell
npm install <pkg>          # updates package.json + lockfile on the host
docker compose build web   # the step that actually installs it
docker compose up -d web
```

Deployed builds are unaffected: they build the image from the committed
`package.json`, so the dependency is present.

## Backend

- Every DB query that touches user data is scoped by `owner_id`, passed
  explicitly rather than read from a context var. Aggregates especially: a
  count that quietly includes another tenant's rows still looks plausible.
- Secrets stay server-side. The frontend calls our API; our API calls the
  provider.
- Tool results that are not retrieved passages are not citable. See
  `corpus_facts` in `backend/app/agent/state.py`.

## Working style

- `docker compose restart` does **not** re-read `.env`. Use
  `docker compose up -d --force-recreate <service>`.
- Never commit or push unless asked in that message.
