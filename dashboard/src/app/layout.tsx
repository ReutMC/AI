import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ARION ALPHA 1 — Persian phase dashboard",
  description: "Training pipeline status for the ARION Persian-specialized model",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen flex flex-col bg-zinc-950 antialiased">{children}</body>
    </html>
  );
}
