// Inline SVG, so the panel needs no icon font and no second request.

export const ARROW = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="1.75" stroke-linecap="round"
  stroke-linejoin="round"><path d="M5 12h13"></path><path d="M13 6l6 6-6 6"></path></svg>`;

export const THEME = {
  system: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <rect x="3" y="4" width="18" height="13" rx="2"></rect>
    <path d="M9 21h6"></path><path d="M12 17v4"></path></svg>`,
  light: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="4.2"></circle>
    <path d="M12 2.6v2.2"></path><path d="M12 19.2v2.2"></path>
    <path d="M4.6 4.6l1.6 1.6"></path><path d="M17.8 17.8l1.6 1.6"></path>
    <path d="M2.6 12h2.2"></path><path d="M19.2 12h2.2"></path>
    <path d="M4.6 19.4l1.6-1.6"></path><path d="M17.8 6.2l1.6-1.6"></path></svg>`,
  dark: `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20 14.5A8.2 8.2 0 0 1 9.5 4 8.2 8.2 0 1 0 20 14.5z"></path></svg>`,
};
