(() => {
  const button = document.getElementById('themeToggle');
  function sync() {
    const dark = document.documentElement.dataset.theme === 'dark';
    button.setAttribute('aria-pressed', String(dark));
    button.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
    button.querySelector('span').textContent = dark ? 'Light mode' : 'Dark mode';
    button.firstChild.textContent = dark ? '☀ ' : '☾ ';
  }
  button.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('hexadepth-theme', theme); } catch (_) {}
    sync();
  });
  sync();
})();
