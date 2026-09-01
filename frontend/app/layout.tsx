import './globals.css';
import Nav from './Nav';

export const metadata = {
  title: 'SatQuery AI',
  description: 'Interactive Vision-Language Assistant for Remote Sensing',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {/* One navigation for every route. Without it /models and
            /benchmarks - the two pages that carry the measured evidence -
            were reachable only by typing their URLs. */}
        <Nav />
        {children}
      </body>
    </html>
  );
}
