import type { Accent } from "@/lib/accents";
import {
  IconChat,
  IconAtlas,
  IconChip,
  IconHowler,
  IconInterview,
  IconParley,
  IconMic,
  IconLab,
  IconLibrary,
  IconProfile,
} from "./icons";

/**
 * The apps living in this one deployment.
 *
 * ONE STACK, SEVERAL APPS. Each of these could be its own service with its own
 * database, auth, CI and hosting bill. None of them is big enough to deserve
 * that, and standing up a second stack is work that teaches nothing about the
 * thing being learned -- so they share a Postgres, a FastAPI process, a Next
 * app and a login, and differ only in which routes they own.
 *
 * Adding an app is: one entry here, plus its pages. The drawer, the switcher
 * and the active-item highlighting all read from this, so there is no second
 * place to update and no way for the nav to disagree with reality.
 */
export type Project = {
  id: string;
  /**
   * The accent this app wears, so which one you are in is legible before you
   * read a word of it. A DEFAULT rather than a lock: somebody who picks an
   * accent explicitly gets it everywhere, because that is a preference about
   * their eyes rather than about the app.
   */
  accent: Accent;
  name: string;
  /** One line, shown in the switcher. Say what it is FOR, not what it is. */
  blurb: string;
  /** The app's own mark. Shown in the drawer header and the switcher, so
   *  the two always agree about which app you are looking at. */
  Icon: typeof IconChat;
  /** Where the switcher navigates. Must be one of `nav`'s hrefs. */
  home: string;
  /** Every route this app owns, used to match the current project. */
  owns: string[];
  nav: { href: string; label: string; Icon: typeof IconChat }[];
};

export const PROJECTS: Project[] = [
  {
    id: "research-desk",
    accent: "purple",
    name: "Research Desk",
    blurb: "Ask questions across your documents and the web, with citations.",
    Icon: IconLibrary,
    home: "/chat",
    owns: ["/chat", "/library", "/lab", "/atlas"],
    nav: [
      { href: "/chat", label: "Chat", Icon: IconChat },
      { href: "/library", label: "Library", Icon: IconLibrary },
      { href: "/lab", label: "Lab", Icon: IconLab },
      { href: "/atlas", label: "Atlas", Icon: IconAtlas },
    ],
  },
  {
    id: "model-lab",
    accent: "blue",
    name: "Model Lab",
    blurb: "Call open models directly — text and speech — and watch what they cost.",
    Icon: IconChip,
    home: "/playground",
    owns: ["/playground", "/transcribe"],
    nav: [
      { href: "/playground", label: "Playground", Icon: IconChip },
      { href: "/transcribe", label: "Transcribe", Icon: IconMic },
    ],
  },
  {
    id: "parley",
    accent: "brown",
    name: "Parley",
    blurb: "Talk it through out loud, or be interviewed. Speech in, speech out.",
    Icon: IconParley,
    home: "/parley",
    owns: ["/parley"],
    nav: [
      { href: "/parley", label: "Speak", Icon: IconParley },
      { href: "/parley/interview", label: "Interview", Icon: IconInterview },
      { href: "/parley/howler", label: "Howler", Icon: IconHowler },
    ],
  },
];

/** Routes every project shares. Not owned by any one of them. */
export const SHARED_NAV = [
  { href: "/profile", label: "Profile", Icon: IconProfile },
];

/**
 * Which project a path belongs to.
 *
 * Longest prefix wins, so a future "/lab/runs" cannot be claimed by a project
 * that merely owns "/l". Falls back to the first project rather than to
 * undefined: a path with no owner (`/profile`, or a 404) still has to render a
 * drawer, and an empty one looks broken.
 */
export function projectFor(pathname: string): Project {
  let best: Project | null = null;
  let bestLength = -1;
  for (const project of PROJECTS) {
    for (const route of project.owns) {
      if (
        (pathname === route || pathname.startsWith(`${route}/`)) &&
        route.length > bestLength
      ) {
        best = project;
        bestLength = route.length;
      }
    }
  }
  return best ?? PROJECTS[0];
}
