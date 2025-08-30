'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';

const logo = '/images/MLOGO2.png';

function Hero() {
  return (
    <div className="md:max-w-4xl sm:max-w-sm sm:mx-auto top-0 left-0 right-0 my-48 mx-4">
      <div className="relative h-96 overflow-hidden">
        <Image src={logo} alt="Logo" fill className="object-contain" />
      </div>
      <span className="flex items-center justify-center text-3xl bg-gradient-to-r from-transparent via-white/30 to-transparent p-1">
        you cannot hide forever
      </span>
    </div>
  );
}

export default Hero;
