"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type AnchorHTMLAttributes,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

/* ===========================================================================
   Material 3 primitives.

   Ripple is the reason these are components rather than CSS classes: it needs
   the pointer coordinates, so it needs an event handler and a DOM node.
   Everything visual still lives in globals.css.
   =========================================================================== */

/**
 * THE RIPPLE KILL SWITCH. Flip to `true` to bring press ripples back.
 *
 * OFF because no duration read right across the whole app: fast enough to suit
 * the buttons made it a flicker on the wide list rows, and slow enough to suit
 * the rows made every button feel laggy.
 *
 * A single flag rather than deleting the code or setting the duration to 0ms.
 * Zero duration would still measure the host, build a clip layer and append it
 * to <body> on every press -- paying the whole cost to render nothing -- and
 * deleting it would throw away the part that was genuinely hard: the layer
 * that survives an unmounting row while still clipping to the host's shape.
 * See `useRipple`.
 *
 * Nothing else changes when this is off. The state layer in globals.css still
 * provides hover, focus and press tinting, so every control keeps its press
 * feedback; it simply does not expand a circle.
 */
const RIPPLE_ENABLED = false;

/**
 * Attaches an M3 press ripple to a host element.
 *
 * The state layer in CSS covers hover/focus/press tinting; this covers the
 * expanding circle. It measures the host, spawns a span at the pointer, and
 * removes it on animation end. Respects prefers-reduced-motion via the global
 * transition kill-switch, which reduces the animation to ~0ms.
 *
 * DURATION IS NOT SET HERE. It is `--md-ripple-duration`, one token for the
 * whole app, and cleanup hangs off `animationend` so it follows whatever that
 * token says without this file knowing the number.
 *
 * WHY THE RIPPLE IS NOT A CHILD OF THE ELEMENT IT BELONGS TO
 *
 * As a child it died when that element unmounted, so the animation was not
 * shortened but CUT OFF wherever it had reached. Rows that navigate showed it
 * plainly: a session row in the drawer lives in the persistent shell and ran
 * full length, while the identical row on the chat LIST page unmounted with
 * the page and stopped part way -- same component, same duration, different
 * lifetime, and it read as "the chat list ripple is much faster".
 *
 * So the span goes into a clip layer on `document.body` instead, which
 * outlives any unmount. The layer exists because the host was silently doing
 * three jobs for the ripple, and all three have to be reproduced or something
 * breaks: its RECT (position), its BORDER-RADIUS (or circles spill out of
 * every pill and FAB), and its `color` (which `currentColor` on the span
 * reads -- miss it and the ripple on a filled button turns dark instead of
 * white).
 */
export function useRipple<T extends HTMLElement>() {
  const host = useRef<T | null>(null);

  const spawn = useCallback((e: React.PointerEvent<T>) => {
    // Before any measuring: the cheapest place to opt out is ahead of the
    // getBoundingClientRect and getComputedStyle below, both of which force
    // layout on every press.
    if (!RIPPLE_ENABLED) return;

    const el = host.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    // Radius must reach the furthest corner, or the ripple stops short on
    // wide elements like list rows.
    const dx = Math.max(e.clientX - rect.left, rect.right - e.clientX);
    const dy = Math.max(e.clientY - rect.top, rect.bottom - e.clientY);
    const radius = Math.hypot(dx, dy);

    // A CLIP LAYER ON <body>, NOT A CHILD OF THE HOST.
    //
    // As a child it died with the host, so any row that navigates had its
    // ripple cut off mid-animation -- visibly "faster" on the chat list, while
    // the identical row in the drawer (which survives the navigation) ran full
    // length. Same component, same duration, different lifetime.
    //
    // The reason this was not simply portalled before is clipping: the ripple
    // is kept inside the host's rounded shape by living inside it, and a bare
    // circle on <body> would spill outside every button and pill. So the layer
    // reproduces the only three things the host was providing -- its rect, its
    // border-radius, and its `color`, which `currentColor` on the span reads.
    // Miss the colour and a ripple on the filled "New chat" button turns dark
    // instead of white.
    const style = getComputedStyle(el);
    const layer = document.createElement("span");
    layer.className = "md-ripple-layer";
    layer.style.left = `${rect.left}px`;
    layer.style.top = `${rect.top}px`;
    layer.style.width = `${rect.width}px`;
    layer.style.height = `${rect.height}px`;
    layer.style.borderRadius = style.borderRadius;
    layer.style.color = style.color;

    const span = document.createElement("span");
    span.className = "md-ripple-span";
    span.style.width = span.style.height = `${radius * 2}px`;
    // Relative to the LAYER, which is positioned at the host's rect -- so
    // these stay the same numbers as when the span was a child of the host.
    span.style.left = `${e.clientX - rect.left - radius}px`;
    span.style.top = `${e.clientY - rect.top - radius}px`;
    // Removes the layer, not the span: the span is inside it, and leaving
    // empty layers on <body> would leak one element per press.
    span.addEventListener("animationend", () => layer.remove(), { once: true });

    layer.appendChild(span);
    document.body.appendChild(layer);
  }, []);

  return { ref: host, onPointerDown: spawn };
}

/**
 * Marks a scroll container as actively scrolling, so its scrollbar can fade in
 * and back out.
 *
 * CSS can do the hover half on its own; "while scrolling" needs JavaScript,
 * because there is no `:scrolling` selector. The attribute is written directly
 * to the DOM rather than held in state -- this fires on every scroll frame, and
 * re-rendering the whole page at 60fps to toggle a scrollbar colour would be an
 * absurd trade.
 *
 * Pair with `.md-scroll`, which owns the appearance.
 */
export function useAutoHideScroll<T extends HTMLElement>() {
  const ref = useRef<T>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let timer: number | undefined;
    const onScroll = () => {
      el.dataset.scrolling = "true";
      window.clearTimeout(timer);
      // Long enough to survive the gap between two flicks of a wheel, short
      // enough that the bar is gone by the time you have finished reading.
      timer = window.setTimeout(() => delete el.dataset.scrolling, 900);
    };

    // Passive: this never calls preventDefault, and saying so lets the browser
    // scroll without waiting on the handler.
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      window.clearTimeout(timer);
    };
  }, []);

  return ref;
}

type ButtonVariant =
  | "filled"
  | "tonal"
  | "outlined"
  | "text"
  | "error"
  | "error-text";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: "sm" | "md";
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    { variant = "filled", size = "md", className = "", children, ...rest },
    _forwarded,
  ) {
    const ripple = useRipple<HTMLButtonElement>();
    return (
      <button
        ref={ripple.ref}
        onPointerDown={ripple.onPointerDown}
        className={`md-btn md-btn-${variant} md-state ${
          size === "sm" ? "md-btn-sm" : ""
        } ${className}`}
        {...rest}
      >
        {children}
      </button>
    );
  },
);

type IconButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "standard" | "filled" | "error";
  size?: "sm" | "md";
};

export function IconButton({
  variant = "standard",
  size = "md",
  className = "",
  children,
  ...rest
}: IconButtonProps) {
  const ripple = useRipple<HTMLButtonElement>();
  return (
    <button
      ref={ripple.ref}
      onPointerDown={ripple.onPointerDown}
      className={`md-icon-btn md-state ${
        variant !== "standard" ? `md-icon-btn-${variant}` : ""
      } ${size === "sm" ? "md-icon-btn-sm" : ""} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Fab({
  size = "md",
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { size?: "sm" | "md" }) {
  const ripple = useRipple<HTMLButtonElement>();
  return (
    <button
      ref={ripple.ref}
      onPointerDown={ripple.onPointerDown}
      className={`md-fab md-state ${size === "sm" ? "md-fab-sm" : ""} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/**
 * A chip that navigates. Same visual role as `Chip`, but an anchor rather than
 * a button, because a web citation's job is to take the reader to a URL --
 * ctrl-click, middle-click and "copy link address" have to work, and a button
 * calling window.open gives up all three. No ripple: the page is leaving.
 */
export function LinkChip({
  size = "md",
  className = "",
  children,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement> & { size?: "sm" | "md" }) {
  return (
    <a
      className={`md-chip md-state no-underline ${
        size === "sm" ? "md-chip-sm" : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </a>
  );
}

export function Chip({
  selected = false,
  size = "md",
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  selected?: boolean;
  size?: "sm" | "md";
}) {
  const ripple = useRipple<HTMLButtonElement>();
  return (
    <button
      ref={ripple.ref}
      onPointerDown={ripple.onPointerDown}
      className={`md-chip md-state ${selected ? "md-chip-selected" : ""} ${
        size === "sm" ? "md-chip-sm" : ""
      } ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/** Generic rippling surface — list rows, nav items, tabs, cards. */
export function Ripplable({
  as: Tag = "button",
  type,
  className = "",
  children,
  ...rest
}: {
  /**
   * Element or component to render. A string tag, or a component such as
   * next/link's `Link` — which the navigation drawer uses so its items are
   * real anchors and get Next's route prefetching.
   *
   * `React.ElementType` rather than a string union because the union could not
   * accept a component, and wrapping a Ripplable inside a Link would nest two
   * interactive elements.
   */
  as?: React.ElementType;
  type?: "button" | "submit" | "reset";
  className?: string;
  children: React.ReactNode;
  // Extra props are forwarded to the rendered component, so `href`/`prefetch`
  // reach Link without Ripplable needing to know they exist.
} & React.HTMLAttributes<HTMLElement> &
  Record<string, unknown>) {
  const ripple = useRipple<HTMLElement>();
  const Component = Tag as React.ElementType;
  return (
    <Component
      ref={ripple.ref}
      onPointerDown={ripple.onPointerDown}
      // A <button> defaults to type="submit", so an unlabelled Ripplable
      // inside a form would submit it — which is how the Lab's top-k stepper
      // would have fired a query on every increment.
      type={Tag === "button" ? (type ?? "button") : type}
      className={`md-state ${className}`}
      {...rest}
    >
      {children}
    </Component>
  );
}

export function Switch({
  on,
  onChange,
  className = "",
  ...rest
}: {
  on: boolean;
  onChange: (v: boolean) => void;
  className?: string;
} & Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onChange">) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      data-on={on}
      onClick={() => onChange(!on)}
      className={`md-switch ${className}`}
      {...rest}
    />
  );
}

export function Checkbox({ on }: { on: boolean }) {
  return (
    <span className="md-checkbox" data-on={on} aria-hidden="true">
      {on && (
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={3}
          strokeLinecap="round"
          strokeLinejoin="round"
          className="h-3 w-3"
        >
          <path d="m5 13 4.5 4.5L19 7" />
        </svg>
      )}
    </span>
  );
}

type TextFieldProps = InputHTMLAttributes<HTMLInputElement> & {
  label: string;
  /**
   * The field's fill. Drives the input background AND the floating label's
   * background, which is what keeps the notch the label cuts in the outline
   * the same colour as the field it is cut into — set them separately and the
   * label reads as a floating swatch the moment the two diverge.
   *
   * Omit for a transparent field on whatever is behind it (M3's default).
   */
  surface?: string;
  /**
   * Corner radius. Defaults to M3's 4px outlined field; pass
   * `var(--md-shape-xl)` for a pill, as the chat composer does.
   *
   * Set on the WRAPPER, not the input: the floating label is a sibling of the
   * input, so a custom property on the input cannot reach it. The label needs
   * its inset moved in step with the radius or it floats onto the corner
   * curve, which is why these two travel together as one prop.
   */
  shape?: string;
};

/**
 * Outlined text field with a floating label.
 *
 * The label must come AFTER the input in the DOM so CSS can style it from the
 * input's :focus / :not(:placeholder-shown) state with a sibling selector.
 * A placeholder of " " is required — without one, :placeholder-shown never
 * matches and the label never floats for filled-but-unfocused fields.
 */
export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(
  function TextField({ label, surface, shape, className = "", ...rest }, ref) {
    const id = useId();
    return (
      <span
        className={`md-field ${className}`}
        style={
          {
            ...(surface ? { "--md-field-bg": surface } : {}),
            ...(shape
              ? {
                  "--md-field-radius": shape,
                  // Clear the corner curve. 1.25rem is enough for the 28px
                  // pill and harmless at smaller radii.
                  "--md-field-label-left": "1.25rem",
                }
              : {}),
          } as React.CSSProperties
        }
      >
        <input
          id={id}
          ref={ref}
          placeholder=" "
          className="md-field-input"
          {...rest}
        />
        <label htmlFor={id} className="md-field-label">
          {label}
        </label>
      </span>
    );
  },
);

type TextAreaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  label: string;
  surface?: string;
  shape?: string;
};

/**
 * The same M3 field, as a textarea that grows with its content.
 *
 * A separate component rather than a `multiline` flag on TextField: the two
 * take different element props (`rows` vs `type`), different refs, and
 * different change-event types, so one component serving both would need a
 * union everywhere it is used and a cast at every call site.
 *
 * Height is set from `scrollHeight` rather than by counting rows, because the
 * only thing that knows how many lines the text occupies after wrapping is the
 * browser. Counting "\n" would keep a long wrapped paragraph one line tall.
 */
export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(
  function TextArea(
    // `placeholder` is SWALLOWED, not forwarded.
    //
    // The floating label works by `:placeholder-shown`, which is why the
    // textarea below sets `placeholder=" "` — a single space that is always
    // "shown" while the field is empty. A caller passing a real placeholder
    // overrode it, and the label then sat still while the hint text rendered
    // underneath it, overlapping. Accepting and discarding it here makes that
    // impossible rather than something to remember; hint text belongs beneath
    // the field, where it survives being typed into.
    { label, surface, shape, className = "", placeholder: _ignored, ...rest },
    ref,
  ) {
    const id = useId();
    const inner = useRef<HTMLTextAreaElement | null>(null);

    const resize = useCallback((el: HTMLTextAreaElement | null) => {
      if (!el) return;
      // Reset FIRST. scrollHeight never reports less than the element's
      // current height, so measuring without this makes the box grow
      // monotonically and never come back down when text is deleted.
      //
      // "0px" rather than "auto": with `auto` the browser may still lay the
      // element out against its `rows` attribute before reporting, which reads
      // back a taller box than the content needs and leaves an empty composer
      // several lines deep.
      el.style.height = "0px";

      const styles = window.getComputedStyle(el);

      // ADD THE BORDERS BACK. `scrollHeight` measures padding + content and
      // EXCLUDES borders, but with `box-sizing: border-box` the height being
      // set INCLUDES them. Assigning scrollHeight directly therefore leaves the
      // content box two pixels short of its own content -- just enough to make
      // a scrollbar appear on a textarea that fits perfectly, permanently, at
      // every size. That was the scrollbar sitting against the pill's edge.
      const borders =
        styles.boxSizing === "border-box"
          ? parseFloat(styles.borderTopWidth) + parseFloat(styles.borderBottomWidth)
          : 0;
      const content = el.scrollHeight + (Number.isFinite(borders) ? borders : 0);

      // Clamp in JS as well as CSS: the inline height set here beats the
      // stylesheet's min-height/max-height on some engines, and an empty field
      // must stay exactly the size of the single-line input it replaced.
      const min = parseFloat(styles.minHeight) || 0;
      const max = parseFloat(styles.maxHeight) || Number.POSITIVE_INFINITY;
      const next = Math.min(Math.max(content, min), max);
      el.style.height = `${next}px`;

      // Scroll ONLY once it is actually clamped. Below the cap the box is
      // exactly its content, so `overflow: auto` would still reserve a gutter
      // in some engines and paint a track over the rounded corner.
      el.style.overflowY = next < content ? "auto" : "hidden";
    }, []);

    // Also on `value`, not just on input: the composer clears the field
    // programmatically after sending, and without this the box would stay
    // several lines tall around an empty textarea.
    useEffect(() => {
      resize(inner.current);
    }, [resize, rest.value]);

    return (
      <span
        className={`md-field ${className}`}
        style={
          {
            ...(surface ? { "--md-field-bg": surface } : {}),
            ...(shape
              ? {
                  "--md-field-radius": shape,
                  "--md-field-label-left": "1.25rem",
                }
              : {}),
          } as React.CSSProperties
        }
      >
        <textarea
          id={id}
          ref={(el) => {
            inner.current = el;
            if (typeof ref === "function") ref(el);
            else if (ref) ref.current = el;
            resize(el);
          }}
          rows={1}
          placeholder=" "
          className="md-field-input"
          {...rest}
        />
        <label htmlFor={id} className="md-field-label">
          {label}
        </label>
      </span>
    );
  },
);

/** M3 dialog with scrim. Escape and scrim-click both dismiss. */
export function Dialog({
  open,
  onClose,
  title,
  body,
  children,
  wide = false,
  contentClassName = "mt-6 flex justify-end gap-2",
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  body?: string;
  children: React.ReactNode;
  /**
   * Widens the panel for dialogs whose content is a list rather than a
   * question. 22rem fits "Delete 3 conversations?" and two buttons; it
   * squeezes anything with rows in it.
   *
   * `"xl"` is for content that is a VIEW rather than a form -- a plot, a
   * canvas -- which needs room to be read at all rather than merely room to
   * avoid wrapping.
   */
  wide?: boolean | "xl";
  /**
   * Classes on the children wrapper.
   *
   * The default is an ACTION ROW -- right-aligned, horizontal -- because that
   * is what every confirm dialog puts here. It is overridable because it is
   * wrong for anything else: a column of cards passed as children was laid out
   * as one right-aligned flex item and shrank to its content, which is what
   * made the app switcher render as a narrow strip.
   */
  contentClassName?: string;
}) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center p-4"
      onClick={onClose}
    >
      <div className="md-scrim" />
      <div
        className={`md-dialog relative ${
          wide === "xl" ? "md-dialog-xl" : wide ? "md-dialog-wide" : ""
        }`}
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="md-headline-small">{title}</h2>
        {body && (
          <p
            className="md-body-medium mt-3"
            style={{ color: "var(--md-on-surface-variant)" }}
          >
            {body}
          </p>
        )}
        <div className={contentClassName}>{children}</div>
      </div>
    </div>
  );
}

/**
 * M3 linear progress, determinate or indeterminate.
 *
 * Omit `value` for the indeterminate variant — the correct choice when the
 * work has no measurable progress, like a benchmark run whose duration depends
 * on rate limits. A determinate bar stuck at one value reads as frozen; an
 * indeterminate one reads as working.
 *
 * `aria-valuenow` is set only in the determinate case: on an indeterminate
 * progressbar its absence is what tells assistive tech the value is unknown.
 */
export function LinearProgress({ value }: { value?: number }) {
  const indeterminate = value === undefined;
  return (
    <div
      className="md-linear-progress"
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : Math.round(value)}
    >
      {indeterminate ? (
        <div className="md-linear-progress-indeterminate" />
      ) : (
        <div style={{ width: `${Math.max(2, Math.min(100, value))}%` }} />
      )}
    </div>
  );
}

/** Confirm-in-place helper: a text button that becomes a dialog. */
export function ConfirmButton({
  label,
  title,
  body,
  confirmLabel = "Delete",
  onConfirm,
  icon,
  className = "",
}: {
  label: string;
  title: string;
  body: string;
  confirmLabel?: string;
  onConfirm: () => void | Promise<void>;
  icon?: React.ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        variant="error-text"
        size="sm"
        className={className}
        onClick={() => setOpen(true)}
      >
        {icon}
        {label}
      </Button>
      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        title={title}
        body={body}
      >
        <Button variant="text" onClick={() => setOpen(false)}>
          Cancel
        </Button>
        <Button
          variant="error"
          onClick={() => {
            setOpen(false);
            void onConfirm();
          }}
        >
          {confirmLabel}
        </Button>
      </Dialog>
    </>
  );
}
