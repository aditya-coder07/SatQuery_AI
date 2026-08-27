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
      <body>{children}</body>
    </html>
  );
}
