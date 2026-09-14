import { ThemeProvider } from './lib/theme';
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

export default function App() {
  return (
    <ThemeProvider>
      <div className="relative min-h-screen overflow-x-clip bg-ink-950 text-ink-50 light:bg-ink-50 light:text-ink-900">
        <Navbar />
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
        <Footer />
      </div>
    </ThemeProvider>
  );
}