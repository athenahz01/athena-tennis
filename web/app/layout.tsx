import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Athena Tennis · US Open 2026",
  description: "Every US Open match, simulated point by point and compared with Kalshi.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="light-field" aria-hidden="true" />
        <header className="topbar">
          <div className="nav-shell">
            <Link className="brand" href="/" aria-label="Athena Tennis home">
              <span className="brand-ball">A</span>
              <span>Athena <strong>Tennis</strong></span>
            </Link>
            <nav aria-label="Primary navigation">
              <Link href="/#matches">Matches</Link>
              <Link href="/agent">Agent</Link>
              <Link href="/#method">Method</Link>
              <a href="https://athena-soccer.vercel.app/">Soccer</a>
            </nav>
          </div>
        </header>
        {children}
        <footer>
          Model probabilities are research outputs. The agent trades on paper. Athena 2026.
        </footer>
      </body>
    </html>
  );
}
