import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { target: 'es2022', sourcemap: true },
  test: { environment: 'jsdom', setupFiles: ['./src/test-setup.ts'], css: false, restoreMocks: true },
});
