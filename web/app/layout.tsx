import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = { title: "AI Tennis Coach", description: "AI-assisted forehand feedback and adaptive practice planning for beginners" };

// apply the viewer's saved theme before first paint (no flash); "system" = no attribute
const THEME = `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: THEME }} /></head>
      <body>{children}</body>
    </html>
  );
}
