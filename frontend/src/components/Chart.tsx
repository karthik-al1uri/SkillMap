import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, GraphChart, HeatmapChart, LineChart, PieChart, ScatterChart } from 'echarts/charts';
import { AriaComponent, GraphicComponent, GridComponent, LegendComponent, MarkLineComponent, TitleComponent, TooltipComponent, VisualMapComponent } from 'echarts/components';
import { LabelLayout } from 'echarts/features';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsOption } from 'echarts';

echarts.use([BarChart, GraphChart, HeatmapChart, LineChart, PieChart, ScatterChart,
  AriaComponent, GraphicComponent, GridComponent, LegendComponent, MarkLineComponent, TitleComponent, TooltipComponent, VisualMapComponent, LabelLayout, CanvasRenderer]);

export const colors = {
  accent: '#b9a3ff', text: '#f0f1f6', muted: '#afb5c5', grid: '#2b3040', reference: '#8893aa',
  tiers: ['#65c5d2', '#e7b56a', '#b9a3ff'],
  palette: ['#b9a3ff', '#65c5d2', '#e7b56a', '#87a5ee', '#df92bb', '#9acaa5'],
};

export interface ChartClickEvent {
  componentType?: string;
  seriesType?: string;
  dataType?: string;
  data?: unknown;
  name?: string;
  value?: unknown;
  dataIndex?: number;
}

export function chartBase(extra: EChartsOption = {}): EChartsOption {
  return {
    backgroundColor: 'transparent', color: colors.palette,
    textStyle: { fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif', color: colors.muted, fontSize: 13 },
    grid: { left: 12, right: 24, top: 22, bottom: 12, containLabel: true },
    tooltip: { trigger: 'axis', backgroundColor: '#242731', borderColor: '#3a3d47',
      textStyle: { color: colors.text, fontSize: 13 }, padding: [10, 13], confine: true },
    xAxis: { type: 'value', axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    yAxis: { type: 'category', axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: colors.muted, fontSize: 13 }, splitLine: { show: false } },
    legend: { textStyle: { color: colors.muted }, icon: 'roundRect', itemWidth: 12, itemHeight: 10 },
    aria: { enabled: true, decal: { show: false } }, animationDuration: 260, ...extra,
  };
}

interface Props {
  option: EChartsOption;
  height?: CSSProperties['height'];
  onClick?: (event: ChartClickEvent) => void;
  className?: string;
  ariaLabel?: string;
}

/** One canvas instance per mounted chart; lifecycle is safe under React StrictMode. */
export default function Chart({ option, height = 320, onClick, className, ariaLabel = 'Interactive evidence chart' }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<ReturnType<typeof echarts.init> | null>(null);

  useEffect(() => {
    if (!container.current) return;
    const chart = echarts.init(container.current, undefined, { renderer: 'canvas' });
    instance.current = chart;
    let frame = 0;
    const resize = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(() => chart.resize()); };
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize);
    observer?.observe(container.current);
    if (!observer) window.addEventListener('resize', resize);
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(frame);
      chart.dispose();
      instance.current = null;
    };
  }, []);

  useEffect(() => {
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    instance.current?.setOption({ ...option, animation: !reduceMotion }, { notMerge: true, lazyUpdate: false });
  }, [option]);

  useEffect(() => {
    const chart = instance.current;
    if (!chart || !onClick) return;
    const handle = (event: unknown) => onClick(event as ChartClickEvent);
    chart.on('click', handle);
    return () => { chart.off('click', handle); };
  }, [onClick]);

  return <div ref={container} role="img" aria-label={ariaLabel} className={className} style={{ width: '100%', height, minWidth: 0 }} />;
}
