import { useEffect, useRef, useState } from 'react';

type Options = {
  /** Trigger only once. Defaults to false (re-trigger on every entry). */
  once?: boolean;
  /** Root margin passed to IntersectionObserver. */
  rootMargin?: string;
  /** Threshold for the IntersectionObserver. */
  threshold?: number;
};

export function useInView<T extends HTMLElement = HTMLElement>({
  once = true,
  rootMargin = '0px 0px -10% 0px',
  threshold = 0.1,
}: Options = {}) {
  const ref = useRef<T | null>(null);
  // Default to true so SSR / pre-hydration / IO-unavailable environments show content.
  const [inView, setInView] = useState(true);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    // If the element is already visible on mount, mark it in view immediately.
    const rect = node.getBoundingClientRect();
    const vh = typeof window !== 'undefined' ? window.innerHeight : 0;
    if (rect.top < vh && rect.bottom > 0) {
      setInView(true);
      if (once) return;
    }

    if (typeof IntersectionObserver === 'undefined') return;

    const obs = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setInView(true);
            if (once) obs.disconnect();
          } else if (!once) {
            setInView(false);
          }
        }
      },
      { rootMargin, threshold },
    );
    obs.observe(node);
    return () => obs.disconnect();
  }, [once, rootMargin, threshold]);

  return { ref, inView };
}