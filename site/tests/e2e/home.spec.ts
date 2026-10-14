import { expect, test } from '@playwright/test';

test('desktop navigation and FAQ are accessible', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/kvfold/i);
  await expect(page.getByRole('link', { name: /kvfold home/i })).toBeVisible();

  const faqButton = page.getByRole('button', { name: /will my generation quality drop/i });
  await expect(faqButton).toHaveAttribute('aria-controls', /faq-answer-/);
  await faqButton.click();
  await expect(faqButton).toHaveAttribute('aria-expanded', 'true');
});

test('mobile navigation opens and closes', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  const menu = page.getByRole('button', { name: /open navigation menu/i });
  await menu.click();
  await expect(page.getByRole('button', { name: /close navigation menu/i })).toBeVisible();
  await page.getByRole('navigation').getByRole('link', { name: 'Methods' }).click();
  await expect(page.getByRole('button', { name: /open navigation menu/i })).toBeVisible();
});

test('theme toggle updates the document theme', async ({ page }) => {
  await page.goto('/');
  const toggle = page.getByRole('button', { name: /switch to (light|dark) theme/i });
  const before = await page.locator('html').getAttribute('class');
  await toggle.click();
  await expect.poll(() => page.locator('html').getAttribute('class')).not.toBe(before);
});

test('documentation and benchmark evidence stay on the product site', async ({ page }) => {
  await page.goto('/');
  const docs = page.getByRole('navigation').getByRole('link', { name: 'Docs' });
  await expect(docs).not.toHaveAttribute('target', '_blank');
  await expect(docs).toHaveAttribute('href', /page=docs/);
  await page.getByText('View values and reproduction commands').click();
  await expect(page.getByRole('link', { name: /full benchmark protocol/i })).toHaveAttribute(
    'href',
    /page=docs/,
  );
});

test('standalone documentation and onboarding pages render', async ({ page }) => {
  await page.goto('?page=docs');
  await expect(page.getByRole('heading', { name: /compress the cache/i })).toBeVisible();
  await page.goto('?page=onboarding');
  await expect(page.getByRole('heading', { name: /clean path from clone/i })).toBeVisible();
});

test('reduced motion renders content without an entrance fade', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.locator('h1')).toHaveCSS('opacity', '1');
});
