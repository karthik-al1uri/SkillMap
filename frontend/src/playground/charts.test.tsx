// @vitest-environment node
/** SVG rendering catches chart errors; it is not a browser visual acceptance test. */
import {readFileSync} from 'node:fs';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe,expect,it,vi} from 'vitest';
import type {ComponentProps} from 'react';
import * as echarts from 'echarts/core';
import {SVGRenderer} from 'echarts/renderers';
import Home from './Home';
import SkillsSalary from './SkillsSalary';
import Profiles from './Profiles';
import Market from './Market';
import {defaults,validatePlaygroundData} from './data';
import {validateData} from '../data/logic';
import {validateMarketData} from '../data/market';
const drawing=vi.hoisted(()=>({width:1100,count:0}));
vi.mock('../components/Chart',async importOriginal=>{const actual=await importOriginal<typeof import('../components/Chart')>();echarts.use([SVGRenderer]);return{...actual,default:({option,ariaLabel,height}:ComponentProps<typeof actual.default>)=>{const chart=echarts.init(null,undefined,{renderer:'svg',ssr:true,width:drawing.width,height:typeof height==='number'?height:450});try{chart.setOption({...option,animation:false});const svg=chart.renderToSVGString();expect(svg.length).toBeGreaterThan(500);expect(svg).not.toMatch(/NaN|Infinity/);drawing.count++;return <div role="img" aria-label={ariaLabel}/>;}finally{chart.dispose();}}};});
const data=validateData(JSON.parse(readFileSync('public/data/skillmap.json','utf8'))),metadata=validatePlaygroundData(JSON.parse(readFileSync('public/data/playground.json','utf8')),data),market=validateMarketData(JSON.parse(readFileSync('public/data/market.json','utf8')));
const pages={Home,SkillsSalary,Profiles,Market};
describe('real playground chart rendering',()=>{it.each([1440,390])('renders all sections with verified data at %ipx chart width',width=>{const warnings=vi.spyOn(console,'warn').mockImplementation(()=>{}),errors=vi.spyOn(console,'error').mockImplementation(()=>{});drawing.width=width;drawing.count=0;try{for(const Page of Object.values(pages)){const markup=renderToStaticMarkup(<Page data={data} metadata={metadata} market={market} state={defaults} update={()=>{}}/>);expect(markup).toContain('<h1>');expect(markup).not.toContain('No matched skills');}expect(drawing.count).toBeGreaterThanOrEqual(9);expect(warnings).not.toHaveBeenCalled();expect(errors).not.toHaveBeenCalled();}finally{warnings.mockRestore();errors.mockRestore();}});});
