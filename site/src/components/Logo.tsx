import { useTheme } from '../lib/theme';

export function Logo({ size = 28 }: { size?: number }) {
  const { theme } = useTheme();
  const fill = theme === 'dark' ? '#ffffff' : '#08090d';
  return (
    <div className="flex items-center gap-2.5">
      <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="shrink-0">
        <defs>
          <linearGradient id="kvfold-mark" x1="4" y1="4" x2="28" y2="28">
            <stop stopColor="#6f8cff" />
            <stop offset="1" stopColor="#3856e8" />
          </linearGradient>
        </defs>
        <rect x="1" y="1" width="30" height="30" rx="10" fill="url(#kvfold-mark)" />
        <path d="M10 8v16M10 16l11-8M10 16l11 8" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M15 16h8" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" opacity=".75" />
      </svg>
      <span
        className="text-[17px] font-semibold tracking-tight"
        style={{ color: fill, letterSpacing: '-0.02em' }}
      >
        kvfold
      </span>
    </div>
  );
}
