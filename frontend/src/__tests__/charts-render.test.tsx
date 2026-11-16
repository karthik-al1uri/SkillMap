// @vitest-environment node
/** Real ECharts SVG rendering verifies configuration, not browser layout or canvas interaction. */
import { readFileSync } from 'node:fs';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { SVGRenderer } from 'echarts/renderers';
import * as echarts from 'echarts/core';
import type { ComponentProps } from 'react';
import Overview from '../pages/Overview';
import Skills from '../pages/Skills';
import Salary from '../pages/Salary';
import Clusters from '../pages/Clusters';
import Live from '../pages/Live';
import Model from '../pages/Model';
import LiveMarket from '../pages/LiveMarket';
import { AdminOverview, AssociationMining, DataQuality } from '../pages/Admin';
import { validateMarketData } from '../data/market';
import type { ProductData } from '../data/types';
import { validateData } from '../data/logic';
import { emptyProductFixture } from './fixtures';

const svgSizes = vi.hoisted(() => [] as number[]);
vi.mock('../components/Chart', async importOriginal => {
  const actual = await importOriginal<typeof import('../components/Chart')>();
  echarts.use([SVGRenderer]);
  return { ...actual, default: ({ option, ariaLabel }: ComponentProps<typeof actual.default>) => {
    const chart = echarts.init(null, undefined, { renderer: 'svg', ssr: true, width: 900, height: 550 });
    try {
      chart.setOption({ ...option, animation: false });
      const svg = chart.renderToSVGString();
      svgSizes.push(svg.length);
      expect(svg).toContain('<svg');
      expect(svg).not.toMatch(/(?:NaN|Infinity)/);
      return <div role="img" aria-label={ariaLabel}/>;
    } finally { chart.dispose(); }
  }};
});

const MarketPage = ({data}:{data:ProductData}) => <LiveMarket data={data} market={validateMarketData(JSON.parse(readFileSync('public/data/market.json','utf8')))}/>;
const ResearchClusters = ({data}:{data:ProductData}) => <Clusters data={data} technical/>;
const pages = { Overview, Skills, Salary, Clusters, Live, Model, MarketPage, ResearchClusters, AdminOverview, AssociationMining, DataQuality };
describe('production chart configuration', () => {
  it.each(Object.entries(pages))('%s renders with real saved data and real ECharts', (_name, Page) => {
    // Skills uses a guarded media-query initializer; no browser layout is simulated here.
    vi.stubGlobal('window', {});
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      svgSizes.length = 0;
      const data = validateData(JSON.parse(readFileSync('public/data/skillmap.json', 'utf8')));
      expect(renderToStaticMarkup(<Page data={data}/>)).toContain('<h1>');
      expect(svgSizes.length).toBeGreaterThanOrEqual(['AdminOverview','AssociationMining'].includes(_name)?0:_name==='DataQuality'?1:2);
      expect(svgSizes.every(size => size > 500)).toBe(true);
      expect(warn).not.toHaveBeenCalled();
      expect(error).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); warn.mockRestore(); error.mockRestore(); }
  });
  it.each(Object.entries(pages))('%s handles unavailable optional evidence', (_name, Page) => {
    vi.stubGlobal('window', {});
    try { expect(renderToStaticMarkup(<Page data={emptyProductFixture()}/>)).toContain('<h1>'); }
    finally { vi.unstubAllGlobals(); }
  });
});
