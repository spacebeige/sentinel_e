const THEME_KEY = 'sentinel-theme';
const THEME_EVENT = 'sentinel-theme-change';

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

export function getCurrentTheme() {
  if (typeof document === 'undefined') return getInitialTheme();
  const root = document.documentElement;
  const explicit = root.getAttribute('data-theme');
  if (explicit === 'light' || explicit === 'dark') return explicit;
  return root.classList.contains('dark') ? 'dark' : getInitialTheme();
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
    window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: { theme: resolved } }));
  }
  return resolved;
}

export function initializeTheme() {
  return applyTheme(getInitialTheme());
}

export function subscribeThemeChanges(onThemeChange) {
  if (typeof window === 'undefined' || typeof onThemeChange !== 'function') {
    return () => {};
  }

  const handleThemeEvent = (event) => {
    const explicitTheme = event?.detail?.theme;
    onThemeChange(explicitTheme === 'light' || explicitTheme === 'dark' ? explicitTheme : getCurrentTheme());
  };

  const handleStorage = (event) => {
    if (event.key !== THEME_KEY) return;
    const theme = event.newValue === 'light' || event.newValue === 'dark'
      ? event.newValue
      : getInitialTheme();
    onThemeChange(theme);
  };

  const mediaQuery = window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)')
    : null;

  const handleSystemThemeChange = () => {
    if (getSavedTheme()) return;
    onThemeChange(getSystemTheme());
  };

  window.addEventListener(THEME_EVENT, handleThemeEvent);
  window.addEventListener('storage', handleStorage);
  if (mediaQuery) {
    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', handleSystemThemeChange);
    } else if (typeof mediaQuery.addListener === 'function') {
      mediaQuery.addListener(handleSystemThemeChange);
    }
  }

  return () => {
    window.removeEventListener(THEME_EVENT, handleThemeEvent);
    window.removeEventListener('storage', handleStorage);
    if (mediaQuery) {
      if (typeof mediaQuery.removeEventListener === 'function') {
        mediaQuery.removeEventListener('change', handleSystemThemeChange);
      } else if (typeof mediaQuery.removeListener === 'function') {
        mediaQuery.removeListener(handleSystemThemeChange);
      }
    }
  };
}
