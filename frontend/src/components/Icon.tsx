export type IconName = 'overview' | 'skills' | 'salary' | 'clusters' | 'live' | 'model' | 'menu' | 'close' | 'info' | 'arrow' | 'database';
const paths: Record<IconName, string> = {
  overview: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
  skills: 'M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM4 3a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM20 3a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM4 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM20 18a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM6 6l4 4m4 0 4-4M6 18l4-4m4 0 4 4',
  salary: 'M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Zm8 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM6 12h.01M18 12h.01',
  clusters: 'M8 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm9 1a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM2 21v-2a6 6 0 0 1 12 0v2m2-7a5 5 0 0 1 6 5v2',
  live: 'M2 12h4l3-8 5 16 3-8h5',
  model: 'M4 4v16h17M8 16l4-5 4 2 5-8',
  menu: 'M4 6h16M4 12h16M4 18h16', close: 'm6 6 12 12M6 18 18 6',
  info: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM12 11v6m0-10v1',
  arrow: 'M5 12h14m-6-6 6 6-6 6',
  database: 'M3 6c0-4 18-4 18 0s-18 4-18 0Zm0 0v12c0 4 18 4 18 0V6M3 12c0 4 18 4 18 0',
};
export default function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
