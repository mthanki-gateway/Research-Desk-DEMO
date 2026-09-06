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
 * Apply the saved accent BEFORE first paint.
 *
 * Doing this in a `useEffect` instead would render the default palette, then
 * repaint in the chosen one -- a visible colour flash on every page load, the
 * same class of problem as a dark-mode flash. A blocking inline script in
 * <head> is the standard fix and the only thing that runs early enough.
 *
 * Deliberately tiny and defensive: any failure (blocked storage, an accent
 * that no longer exists) leaves the default `:root` palette in place, which is
 * a complete working theme rather than a broken one.
 */
const ACCENT_INIT = `try{var a=localStorage.getItem("rd.accent");if(a)document.documentElement.setAttribute("data-accent",a)}catch(e){}`;

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
