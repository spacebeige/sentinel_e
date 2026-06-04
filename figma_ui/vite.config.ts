import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig({

  plugins: [
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
      '@services': path.resolve(__dirname, '../frontend/src/services'),
      '@stores': path.resolve(__dirname, '../frontend/src/stores'),
      '@hooks': path.resolve(__dirname, '../frontend/src/hooks'),
      '@utils': path.resolve(__dirname, '../frontend/src/utils'),
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom', 'react-router'],
          'three-vendor': ['three', '@react-three/fiber', '@react-three/drei'],
          'supabase-vendor': ['@supabase/supabase-js'],
        }
      }
    }
  }
})
