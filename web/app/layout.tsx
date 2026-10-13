import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "AI Tennis Coach", description: "AI-assisted forehand feedback and adaptive practice planning for beginners (GP8001 Group K5)" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
