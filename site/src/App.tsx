import { ThemeProvider } from './lib/theme';
import type { ReactNode } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Hero } from './sections/Hero';
import { ValueProps } from './sections/ValueProps';
import { Methods } from './sections/Methods';
import { HowItWorks } from './sections/HowItWorks';
import { Benchmarks } from './sections/Benchmarks';
import { Integrations } from './sections/Integrations';
import { Install } from './sections/Install';
import { FAQ } from './sections/FAQ';
import { CTA } from './sections/CTA';
import { DocsPage } from './pages/DocsPage';
import { OnboardingPage } from './pages/OnboardingPage';
import { ExamplesPage } from './pages/ExamplesPage';

function Shell({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <div className="relative min-h-screen overflow-x-clip bg-ink-950 text-ink-50 light:bg-ink-50 light:text-ink-900">
        <Navbar />
        {children}
        <Footer />
      </div>
    </ThemeProvider>
  );
}

export default function App() {
  const page = new URLSearchParams(window.location.search).get('page');

  if (page === 'docs') {
    return <Shell><DocsPage /></Shell>;
  }
  if (page === 'onboarding') {
    return <Shell><OnboardingPage /></Shell>;
  }
  if (page === 'examples') {
    return <Shell><ExamplesPage /></Shell>;
  }

  return <Shell>
        <main>
          <Hero />
          <ValueProps />
          <Methods />
          <HowItWorks />
          <Benchmarks />
          <Integrations />
          <Install />
          <FAQ />
          <CTA />
        </main>
      </Shell>;
}
