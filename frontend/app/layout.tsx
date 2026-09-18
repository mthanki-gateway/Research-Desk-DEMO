import type { Metadata } from "next";
// Roboto is Material's type family. Self-hosted via Fontsource rather than
// next/font/google, so the Docker build needs no network access.
import "@fontsource/roboto/400.css";
import "@fontsource/roboto/500.css";
import "@fontsource/roboto/700.css";
import "./globals.css";
import { Providers } from "./providers";
import Shell from "./shell";

export const metadata: Metadata = {
  title: "Research Desk",
  description: "Document-grounded research agent built on LangGraph",
};

/**
 * Apply the app's accent BEFORE first paint.
 *
 * Doing this in a `useEffect` instead would render one palette and repaint in
 * another -- a visible colour flash on every page load, the same class of
 * problem as a dark-mode flash. A blocking inline script in <head> is the
 * standard fix and the only thing that runs early enough.
 *
 * FROM THE PATH, not from storage. It read `rd.accent` until the accent became
 * a property of the app rather than a personal setting -- and since nothing in
 * the UI ever wrote that key, a value left over from an older build was enough
 * to pin every app to one colour with no way to clear it. Reading the path has
 * no such state to go stale.
 *
 * The mapping is duplicated from `projects.ts` because this runs before any
 * module does. Kept to prefixes so it cannot drift far, and wrong only in the
 * direction of the default palette, which is a complete working theme.
 */
const ACCENT_INIT = `try{var p=location.pathname,a=p.indexOf("/parley")===0?"brown":(p.indexOf("/playground")===0||p.indexOf("/transcribe")===0)?"blue":"purple";document.documentElement.setAttribute("data-accent",a)}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <script dangerouslySetInnerHTML={{ __html: ACCENT_INIT }} />
      </head>
      <body className="min-h-screen antialiased">
        <Providers>
          <Shell>{children}</Shell>
        </Providers>
      </body>
    </html>
  );
}
