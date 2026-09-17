/**
 * Inline SVG icons, stroked, 1.6px, currentColor.
 *
 * Hand-rolled rather than an icon package: we need eleven glyphs, and a
 * dependency would ship hundreds. Deliberately no emoji — they render
 * differently per platform and read as informal.
 */

type Props = { className?: string };

function Svg({
  children,
  className = "h-4 w-4",
}: Props & { children: React.ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export const IconChat = (p: Props) => (
  <Svg {...p}>
    <path d="M21 12a8 8 0 0 1-8 8H8l-4 3v-5.5A8 8 0 0 1 12 4h1a8 8 0 0 1 8 8Z" />
  </Svg>
);

export const IconLibrary = (p: Props) => (
  <Svg {...p}>
    <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H9v16H5.5A1.5 1.5 0 0 1 4 18.5Z" />
    <path d="M9 4h4.5A1.5 1.5 0 0 1 15 5.5v13A1.5 1.5 0 0 1 13.5 20H9" />
    <path d="m17 5.6 2.6 12.1" />
  </Svg>
);

export const IconLab = (p: Props) => (
  <Svg {...p}>
    <path d="M9 3h6M10 3v5.5L5.5 17A3 3 0 0 0 8 21h8a3 3 0 0 0 2.5-4L14 8.5V3" />
    <path d="M7.5 14h9" />
  </Svg>
);

export const IconMic = (p: Props) => (
  <Svg {...p}>
    <rect x="9" y="3" width="6" height="11" rx="3" />
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" />
  </Svg>
);

/** Parley's mark: sound arriving at an ear. */
export const IconParley = (p: Props) => (
  <Svg {...p}>
    <path d="M8.5 20c0-2.5-1-3-2.2-4.2A6.5 6.5 0 0 1 4.5 11a6 6 0 0 1 12 0c0 2-1 3-2.3 3.6-1 .5-1.2 1.2-1.2 2.1a2.5 2.5 0 0 1-4.5 1.5" />
    <path d="M9 10.5a1.8 1.8 0 0 1 3.5.6" />
    <path d="M19 8.5a6 6 0 0 1 0 7M21.5 6a10 10 0 0 1 0 12" />
  </Svg>
);

/** Interview: two people, one asking. */
export const IconInterview = (p: Props) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="3" />
    <path d="M2.5 20a5.5 5.5 0 0 1 11 0" />
    <path d="M16 5h5.5v6H19l-1.5 2.5V11H16z" />
  </Svg>
);

/** A waveform, for the speaking state. */
export const IconWave = (p: Props) => (
  <Svg {...p}>
    <path d="M3 12h2M8 7v10M12 4v16M16 8v8M20 11h1" />
  </Svg>
);

/** Howler: a brief going in, a conversation coming out. */
export const IconHowler = (p: Props) => (
  <Svg {...p}>
    <path d="M4 5h9M4 9h7M4 13h9" />
    <path d="M14 17.5a4.5 4.5 0 1 0 4.5-4.5" />
    <path d="M18.5 13v4.5h4.5" />
  </Svg>
);

/** Rename. */
export const IconEdit = (p: Props) => (
  <Svg {...p}>
    <path d="M4 20h4l10-10-4-4L4 16Z" />
    <path d="M14.5 5.5 18.5 9.5" />
  </Svg>
);

/** Overflow menu. Three dots, drawn rather than typed as "⋯" — the character
 *  renders at a different size and baseline in every font. */
export const IconMore = (p: Props) => (
  <Svg {...p}>
    <circle cx="12" cy="5" r="1.4" fill="currentColor" />
    <circle cx="12" cy="12" r="1.4" fill="currentColor" />
    <circle cx="12" cy="19" r="1.4" fill="currentColor" />
  </Svg>
);

/** Playback. */
export const IconPlay = (p: Props) => (
  <Svg {...p}>
    <path d="M8 5.5 18 12 8 18.5Z" />
  </Svg>
);

export const IconStop = (p: Props) => (
  <Svg {...p}>
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </Svg>
);

export const IconAtlas = (p: Props) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="2" />
    <circle cx="5" cy="7" r="1.6" />
    <circle cx="19" cy="8" r="1.6" />
    <circle cx="7" cy="18" r="1.6" />
    <circle cx="18" cy="17" r="1.6" />
    <path d="M6.4 8.2 10 10.8M17.6 9 14 10.8M8.3 16.8 10.7 13.8M16.7 15.8 13.5 13.4" />
  </Svg>
);

export const IconChip = (p: Props) => (
  <Svg {...p}>
    <rect x="7" y="7" width="10" height="10" rx="2" />
    <path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4" />
  </Svg>
);

export const IconGrid = (p: Props) => (
  <Svg {...p}>
    <rect x="4" y="4" width="7" height="7" rx="1.5" />
    <rect x="13" y="4" width="7" height="7" rx="1.5" />
    <rect x="4" y="13" width="7" height="7" rx="1.5" />
    <rect x="13" y="13" width="7" height="7" rx="1.5" />
  </Svg>
);

export const IconProfile = (p: Props) => (
  <Svg {...p}>
    <circle cx="12" cy="8" r="3.5" />
    <path d="M5 20a7 7 0 0 1 14 0" />
  </Svg>
);

export const IconPlus = (p: Props) => (
  <Svg {...p}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
);

export const IconClose = (p: Props) => (
  <Svg {...p}>
    <path d="M6 6l12 12M18 6 6 18" />
  </Svg>
);

export const IconMenu = (p: Props) => (
  <Svg {...p}>
    <path d="M4 7h16M4 12h16M4 17h16" />
  </Svg>
);

export const IconChevron = ({
  className = "h-4 w-4",
  open = false,
}: Props & { open?: boolean }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    className={`${className} transition-transform ${open ? "rotate-90" : ""}`}
    aria-hidden="true"
  >
    <path d="m9 6 6 6-6 6" />
  </svg>
);

export const IconSearch = (p: Props) => (
  <Svg {...p}>
    <circle cx="11" cy="11" r="6.5" />
    <path d="m16 16 4 4" />
  </Svg>
);

export const IconTrash = (p: Props) => (
  <Svg {...p}>
    <path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13" />
  </Svg>
);

export const IconCheck = (p: Props) => (
  <Svg {...p}>
    <path d="m5 13 4.5 4.5L19 7" />
  </Svg>
);

export const IconUpload = (p: Props) => (
  <Svg {...p}>
    <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
    <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
  </Svg>
);

export const IconDocument = (p: Props) => (
  <Svg {...p}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
    <path d="M14 3v5h5" />
  </Svg>
);

export const IconQuote = (p: Props) => (
  <Svg {...p}>
    <path d="M6 7h5v5a4 4 0 0 1-4 4H6M14 7h5v5a4 4 0 0 1-4 4h-1" />
  </Svg>
);

/** Marks a citation that leaves the app — a web source, not a document chunk. */
export const IconExternal = (p: Props) => (
  <Svg {...p}>
    <path d="M14 4h6v6M20 4l-8 8M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />
  </Svg>
);

export const IconThumbUp = (p: Props) => (
  <Svg {...p}>
    <path d="M7 20V9m0 0 3.5-6a2 2 0 0 1 2.9 2.4L12.5 9H18a2 2 0 0 1 2 2.3l-1.1 6.4A2 2 0 0 1 17 19.4H7Z" />
  </Svg>
);

export const IconThumbDown = (p: Props) => (
  <Svg {...p}>
    <path d="M17 4v11m0 0-3.5 6a2 2 0 0 1-2.9-2.4l.9-3.6H6a2 2 0 0 1-2-2.3l1.1-6.4A2 2 0 0 1 7 4.6h10Z" />
  </Svg>
);

export const IconSignOut = (p: Props) => (
  <Svg {...p}>
    <path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3" />
    <path d="M10 8l-4 4 4 4M6 12h9" />
  </Svg>
);

export const IconSpinner = ({ className = "h-4 w-4" }: Props) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    className={`${className} animate-spin`}
    aria-hidden="true"
  >
    <path d="M12 3a9 9 0 1 0 9 9" />
  </svg>
);

/** A magic link: the thing you send to whoever is being interviewed. */
export const IconLink = (p: Props) => (
  <Svg {...p}>
    <path d="M10 13.5a4 4 0 0 0 5.7.3l2.8-2.8a4 4 0 0 0-5.7-5.7l-1.6 1.6" />
    <path d="M14 10.5a4 4 0 0 0-5.7-.3l-2.8 2.8a4 4 0 0 0 5.7 5.7l1.6-1.6" />
  </Svg>
);

export const IconCopy = (p: Props) => (
  <Svg {...p}>
    <rect x="9" y="9" width="11" height="11" rx="2" />
    <path d="M6 15H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v1" />
  </Svg>
);

/** Fetch again — for state that changes in somebody else's browser. */
export const IconRefresh = (p: Props) => (
  <Svg {...p}>
    <path d="M20 11a8 8 0 1 0-.7 4.3" />
    <path d="M20 5v6h-6" />
  </Svg>
);
