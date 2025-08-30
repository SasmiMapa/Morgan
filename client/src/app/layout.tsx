import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Morgan',
  description: 'Identify Counterfeit',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`antialiased`}>
        <div className="flex flex-col min-h-screen">
          {/* <Navbar /> */}
          <main className="flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
