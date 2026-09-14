import { useTheme } from '../lib/theme';

export function Logo({ size = 28 }: { size?: number }) {
  const { theme } = useTheme();
  const fill = theme === 'dark' ? '#ffffff' : '#08090d';
  return (
    <div className="flex items-center gap-2.5">
      <span
        className="relative inline-flex items-center justify-center rounded-[10px] bg-accent-500 text-white shadow-[inset_0_1px_0_0_rgba(255,255,255,0.25),0_8px_24px_-12px_rgba(58,92,245,0.7)]"
        style={{ width: size, height: size }}
        aria-hidden
      >
        <span
          className="font-display text-[18px] font-bold"
          style={{ fontFamily: 'Inter, sans-serif', letterSpacing: '-0.04em' }}
        >
          k
        </span>
      </span>
      <span
        className="text-[17px] font-semibold tracking-tight"
        style={{ color: fill, letterSpacing: '-0.02em' }}
      >
        kvfold
      </span>
    </div>
  );
}