const THEME_KEY = 'sentinel-theme';

function getSystemTheme() {
  if (typeof window === 'undefined' || !window.matchMedia) return 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function getSavedTheme() {
  if (typeof window === 'undefined') return null;
  try {
    const stored = window.localStorage.getItem(THEME_KEY);
    return stored === 'light' || stored === 'dark' ? stored : null;
  } catch {
    return null;
  }
}

export function getInitialTheme() {
  return getSavedTheme() || getSystemTheme();
}

export function applyTheme(theme) {
  if (typeof document === 'undefined') return theme;
  const resolved = theme === 'light' ? 'light' : 'dark';
  const root = document.documentElement;
  root.classList.toggle('dark', resolved === 'dark');
  root.setAttribute('data-theme', resolved);
  root.style.colorScheme = resolved;
  return resolved;
}

export function persistTheme(theme) {
  const resolved = applyTheme(theme);
  if (typeof window !== 'undefined') {
    try {
      window.localStorage.setItem(THEME_KEY, resolved);
    } catch {
      /* localStorage is best-effort */
    }
  }
  return resolved;
}

export function initializeTheme() {
  return applyTheme(getInitialTheme());
}

