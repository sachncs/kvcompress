import { motion, useReducedMotion } from 'framer-motion';
import type { ReactNode } from 'react';
import { useInView } from './useInView';

type RevealProps = {
  children: ReactNode;
  delay?: number;
  y?: number;
  className?: string;
  as?: 'div' | 'section' | 'header' | 'footer' | 'span' | 'li' | 'p' | 'h1' | 'h2' | 'h3';
};

export function Reveal({
  children,
  delay = 0,
  y = 18,
  className,
  as = 'div',
}: RevealProps) {
  const { ref, inView } = useInView<HTMLDivElement>();
  const reduced = useReducedMotion();

  const MotionTag = motion[as] as typeof motion.div;

  return (
    <MotionTag
      ref={ref}
      className={className}
      initial={reduced ? false : { opacity: 0, y }}
      animate={inView ? { opacity: 1, y: 0 } : reduced ? { opacity: 1 } : undefined}
      transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay }}
    >
      {children}
    </MotionTag>
  );
}