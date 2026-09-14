# kvfold — product site

Premium product page for the `kvfold` library. Built with Vite, React 18,
TypeScript, Tailwind CSS, and Framer Motion. Deployed to GitHub Pages.

This folder is fully self-contained — no files outside `site/` are required to
build or run it.

## Develop

```bash
cd site
npm install
npm run dev      # http://127.0.0.1:5173
```

## Build

```bash
cd site
npm run build    # → site/dist
npm run preview  # serve the built site locally at the configured base path
```

`npm run typecheck` runs the TypeScript project references without emitting.

## Deploy

Pushing to `master` triggers `.github/workflows/pages.yml`, which builds the
site with `VITE_BASE=/<repo>/` and publishes the contents of `site/dist/` to
GitHub Pages.

The default base path is `/kvcompress/`. Override at build time:

```bash
VITE_BASE=/my-fork/ npm run build
```

## Layout

```
site/
├── index.html               # Vite entry, sets <html class="dark">
├── public/                   # Copied verbatim into dist/ (favicon, og, 404)
│   ├── favicon.svg
│   ├── og.svg
│   └── 404.html
├── src/
│   ├── main.tsx             # React bootstrap
│   ├── App.tsx              # Section composition + ThemeProvider
│   ├── index.css            # Tailwind layers + design-system utilities
│   ├── components/          # Navbar, Logo, Footer, Icons
│   ├── lib/                 # theme.tsx, content.ts, Reveal, useInView
│   └── sections/            # Hero, ValueProps, Methods, HowItWorks,
│                            # Benchmarks, Integrations, Install, FAQ, CTA
└── tailwind.config.ts       # Design tokens (ink, accent, display sizes, keyframes)
```

## Design system

- **Type** — Inter (UI), Instrument Serif (display accents), JetBrains Mono (code).
- **Color** — `ink` (9-step neutral scale), single `accent` (electric indigo).
- **Motion** — Framer Motion for in-view reveals, IntersectionObserver for
  reduced-motion-safe triggers.
- **Theme** — class-based (`dark` / `light`), persisted in localStorage, with
  one-tap toggle in the navbar.