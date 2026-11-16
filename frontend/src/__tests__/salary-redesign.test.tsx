/** Interaction tests inspect generated chart options; canvas rendering needs browser review. */
import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import Salary from '../pages/Salary';
import { productFixture } from './fixtures';
vi.mock('../components/Chart', async importOriginal => {
  const actual = await importOriginal<typeof import('../components/Chart')>();
  return { ...actual, default: ({ option, ariaLabel }: ComponentProps<typeof actual.default>) => <div role="img" aria-label={ariaLabel}><output>{JSON.stringify(option)}</output></div> };
});
function plottedJobs() {
  const image = screen.getByRole('img', { name: 'Salary distribution of the filtered unique-job cohort' });
  const option = JSON.parse(within(image).getByRole('status').textContent ?? '{}') as { series: {data:number[]}[] };
  return option.series[0].data.reduce((sum,value) => sum+value, 0);
}
function stat(label: string) {
  return screen.getAllByRole('article').find(item => within(item).queryByText(label, {exact:true}))!;
}
describe('salary explorer redesign', () => {
  it('makes excluded tails explicit, retains full-cohort statistics and can show every outlier', async () => {
    const user = userEvent.setup();
    render(<Salary data={productFixture()}/>);
    expect(plottedJobs()).toBe(4);
    expect(screen.getByText(/4 jobs shown · 1 below P5 · 1 above P95/)).toBeInTheDocument();
    expect(stat('Median annual salary')).toHaveTextContent('$90,000');
    await user.click(screen.getByRole('button', {name:'Full range'}));
    expect(plottedJobs()).toBe(6);
    expect(stat('Median annual salary')).toHaveTextContent('$90,000');
  });
  it('searches filters, handles no results and resets the full cohort', async () => {
    const user = userEvent.setup();
    render(<Salary data={productFixture()}/>);
    const search = screen.getByRole('combobox', {name:'Skill mentioned'});
    await user.click(search);
    await user.type(search, 'secur');
    await user.click(screen.getByRole('option', {name:'security'}));
    expect(stat('Jobs in your selection')).toHaveTextContent('1');
    expect(stat('Median annual salary')).toHaveTextContent('$120,000');
    await user.click(search);
    await user.type(search, 'zz-no-such-skill');
    expect(screen.getByText('No matching options')).toBeInTheDocument();
    await user.keyboard('{Escape}');
    await user.click(screen.getByRole('button', {name:'Reset filters'}));
    expect(stat('Jobs in your selection')).toHaveTextContent('6');
    expect(search).toHaveValue('Any skill');
  });
  it('supports keyboard-equivalent group focus, clearing and cohort drilldown', async () => {
    const data = productFixture();
    data.rows = data.rows.flatMap(row => Array.from({length:5}, () => structuredClone(row)));
    const user = userEvent.setup();
    render(<Salary data={data}/>);
    await user.selectOptions(screen.getByRole('combobox', {name:'Minimum group size'}), '10');
    await user.click(screen.getByText('Accessible values & keyboard drilldown'));
    await user.click(screen.getByRole('button', {name:'Inspect python'}));
    expect(screen.getByRole('button', {name:'Inspect python'})).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', {name:'Clear selection'}));
    expect(screen.getByRole('button', {name:'Inspect python'})).toHaveAttribute('aria-pressed', 'false');
    await user.click(screen.getByRole('button', {name:'Inspect python'}));
    await user.click(screen.getByRole('button', {name:'Filter to this group →'}));
    expect(stat('Jobs in your selection')).toHaveTextContent('20');
    expect(stat('Median annual salary')).toHaveTextContent('$100,000');
  });
});
