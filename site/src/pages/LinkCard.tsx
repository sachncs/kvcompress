export function LinkCard({ number, title, body }: { number: string; title: string; body: string }) {
  return <div className="surface p-4 light:surface-light"><span className="font-mono text-xs text-blue-300 light:text-blue-700">{number}</span><h3 className="mt-3 text-sm font-semibold">{title}</h3><p className="mt-1 text-xs leading-5 text-ink-400 light:text-ink-600">{body}</p></div>;
}
