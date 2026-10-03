import type { Metadata } from "next";
import { Geist, Geist_Mono, Source_Serif_4 } from "next/font/google";
import NavLinks from "@/components/NavLinks";
import "katex/dist/katex.min.css";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Paper Agent",
  description: "A research-assistant agent for professors.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${sourceSerif.variable} h-full antialiased`}
    >
      <body className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
        <nav className="flex shrink-0 items-center gap-6 border-b border-border px-6 py-4">
          <span className="font-serif text-[15px] font-semibold tracking-tight text-foreground">
            Paper Agent
          </span>
          <NavLinks />
        </nav>
        <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      </body>
    </html>
  );
}
