import type { Metadata } from "next";
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
            <a className="brand" href="#top" aria-label="Athena Tennis home">
              <span className="brand-ball">A</span>
              <span>Athena <strong>Tennis</strong></span>
            </a>
            <nav aria-label="Primary navigation">
              <a href="#matches">Matches</a>
              <a href="#method">Method</a>
              <a href="https://athena-soccer.vercel.app/">Soccer ↗</a>
            </nav>
          </div>
        </header>
        {children}
        <footer>
          Model probabilities, not betting advice. Markets are timestamped snapshots. Athena 2026.
        </footer>
      </body>
    </html>
  );
}
