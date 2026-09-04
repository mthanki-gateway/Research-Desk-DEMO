"use client";

import {
  forwardRef,
  useCallback,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
} from "react";

/* ===========================================================================
   Material 3 primitives.

   Ripple is the reason these are components rather than CSS classes: it needs
   the pointer coordinates, so it needs an event handler and a DOM node.
   Everything visual still lives in globals.css.
   =========================================================================== */

/**
 * Attaches an M3 press ripple to a host element.
 *
 * The state layer in CSS covers hover/focus/press tinting; this covers the
 * expanding circle. It measures the host, spawns a span at the pointer, and
 * removes it on animation end. Respects prefers-reduced-motion via the global
 * transition kill-switch, which reduces the animation to ~0ms.
 */
export function useRipple<T extends HTMLElement>() {
  const host = useRef<T | null>(null);

  const spawn = useCallback((e: React.PointerEvent<T>) => {
    const el = host.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    // Radius must reach the furthest corner, or the ripple stops short on
    // wide elements like list rows.
    const dx = Math.max(e.clientX - rect.left, rect.right - e.clientX);
    const dy = Math.max(e.clientY - rect.top, rect.bottom - e.clientY);
    const radius = Math.hypot(dx, dy);

    const span = document.createElement("span");
    span.className = "md-ripple-span";
    span.style.width = span.style.height = `${radius * 2}px`;
    span.style.left = `${e.clientX - rect.left - radius}px`;
    span.style.top = `${e.clientY - rect.top - radius}px`;
    span.addEventListener("animationend", () => span.remove(), { once: true });
    el.appendChild(span);
  }, []);

  return { ref: host, onPointerDown: spawn };
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
  /** Surface the field sits on, so the floating label's notch matches it. */
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

/** M3 dialog with scrim. Escape and scrim-click both dismiss. */
export function Dialog({
  open,
  onClose,
  title,
  body,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  body?: string;
  children: React.ReactNode;
}) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center p-4"
      onClick={onClose}
    >
      <div className="md-scrim" />
      <div
        className="md-dialog relative"
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
        <div className="mt-6 flex justify-end gap-2">{children}</div>
      </div>
    </div>
  );
}

export function LinearProgress({ value }: { value: number }) {
  return (
    <div
      className="md-linear-progress"
      role="progressbar"
      aria-valuenow={Math.round(value)}
    >
      <div style={{ width: `${Math.max(2, Math.min(100, value))}%` }} />
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
