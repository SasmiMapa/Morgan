'use client';

import { useEffect, useRef } from 'react';
import { Navbar } from '@/components';
import { About } from '@/components/pages';

export default function Home() {
  const scrollRef = useRef<any>(null);

  useEffect(() => {
    (async () => {
      const LocomotiveScroll = (await import('locomotive-scroll')).default;

      scrollRef.current = new LocomotiveScroll({
        el: document.querySelector('main')!,
        smooth: true,
      });
    })();
  }, []);

  return (
    <main>
      <Navbar scrollRef={scrollRef} />
      <About />
    </main>
  );
}
