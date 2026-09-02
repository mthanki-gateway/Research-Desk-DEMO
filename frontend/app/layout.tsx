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

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Providers>
          <Shell>{children}</Shell>
        </Providers>
      </body>
    </html>
  );
}
