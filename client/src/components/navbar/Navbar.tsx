'use client';

import Image from 'next/image';
import Link from 'next/link';

const logo = '/images/MICON.png';

const navLinks = [{ label: 'About Morgan', path: '/about' }];

const Navbar = ({ scrollRef }: { scrollRef: React.RefObject<any> }) => {
  return (
    <div className="md:max-w-2xl sm:max-w-sm mx-auto w-fit fixed top-0 left-0 right-0 my-5 z-[50]">
      <div className="flex justify-center items-center bg-white/5 backdrop-blur-2xl text-center rounded-xl p-1">
        <div className="flex items-center justify-center gap-1 bg-white/5 rounded-lg p-1">
          <Link
            href="/"
            className="rounded-md px-4 py-2 relative h-9 w-9 overflow-hidden border border-white/5 bg-white/10 hover:scale-95 active:scale-90 transition-all"
          >
            <Image src={logo} alt="Logo" fill className="object-contain" />
          </Link>

          <div className="rounded-md px-4 py-2 text-sm text-white/60 border border-white/5 hover:scale-[98%] active:scale-95 transition-all">
            {navLinks.map((link, idx) => (
              <Link key={idx} href={link.path}>
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
