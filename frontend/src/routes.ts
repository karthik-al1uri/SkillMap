import type { IconName } from './components/Icon';
export type Mode = 'user' | 'admin';
export type PageId = 'explore' | 'skills' | 'salary' | 'profiles' | 'market' | 'admin-overview' | 'model' | 'mining' | 'clustering' | 'quality' | 'validation';
export interface Route { path: string; label: string; mode: Mode; page: PageId; icon: IconName }
export const routes: Route[] = [
  {path:'/app',label:'Explore',mode:'user',page:'explore',icon:'overview'},
  {path:'/app/skills',label:'Skills',mode:'user',page:'skills',icon:'skills'},
  {path:'/app/salary',label:'Salary Intelligence',mode:'user',page:'salary',icon:'salary'},
  {path:'/app/profiles',label:'Career Profiles',mode:'user',page:'profiles',icon:'clusters'},
  {path:'/app/live',label:'Live Market',mode:'user',page:'market',icon:'live'},
  {path:'/admin',label:'Overview',mode:'admin',page:'admin-overview',icon:'overview'},
  {path:'/admin/model',label:'Model Evaluation',mode:'admin',page:'model',icon:'model'},
  {path:'/admin/mining',label:'Association Mining',mode:'admin',page:'mining',icon:'skills'},
  {path:'/admin/clusters',label:'Clustering Analysis',mode:'admin',page:'clustering',icon:'clusters'},
  {path:'/admin/quality',label:'Data Quality',mode:'admin',page:'quality',icon:'database'},
  {path:'/admin/live',label:'Live Validation',mode:'admin',page:'validation',icon:'live'},
];
export const legacyRoutes: Record<string,string> = {overview:'/app',skills:'/app/skills',salary:'/app/salary',clusters:'/admin/clusters',live:'/admin/live',model:'/admin/model'};
export function resolveRoute(path: string): Route | null {
  const canonical = path.replace(/\/+$/,'') || '/app';
  const alias = canonical === '/app/explore' ? '/app' : canonical === '/admin/overview' ? '/admin' : canonical;
  return routes.find(route=>route.path===alias) ?? null;
}
export function navigate(path: string, replace = false) {
  if (replace) window.history.replaceState(null,'',path);
  else window.history.pushState(null,'',path);
  window.dispatchEvent(new PopStateEvent('popstate'));
}
export function initialRoute(): Route | null {
  const legacy = legacyRoutes[location.hash.slice(1)];
  if (legacy) { window.history.replaceState(null,'',legacy); return resolveRoute(legacy); }
  return resolveRoute(location.pathname);
}
