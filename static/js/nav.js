// ============================================================
// Navigation — Hamburger menu + theme toggle (shared)
// ============================================================

(function () {
    // === Theme (runs immediately to prevent FOUC) ===
    const stored = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (stored === 'dark' || (!stored && prefersDark)) {
        document.documentElement.classList.add('dark');
    }

    document.addEventListener('DOMContentLoaded', () => {
        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        themeToggle?.addEventListener('click', () => {
            document.documentElement.classList.toggle('dark');
            localStorage.setItem('theme',
                document.documentElement.classList.contains('dark') ? 'dark' : 'light');
            // Refresh chart themes if available
            if (typeof refreshChartTheme === 'function') refreshChartTheme();
        });

        // Hamburger menu
        const menuBtn = document.getElementById('mobileMenuBtn');
        const mobileMenu = document.getElementById('mobileMenu');
        if (menuBtn && mobileMenu) {
            menuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const isOpen = mobileMenu.classList.toggle('open');
                menuBtn.setAttribute('aria-expanded', isOpen);
            });

            // Close on outside click
            document.addEventListener('click', (e) => {
                if (!mobileMenu.contains(e.target) && !menuBtn.contains(e.target)) {
                    mobileMenu.classList.remove('open');
                    menuBtn.setAttribute('aria-expanded', 'false');
                }
            });

            // Close on link click
            mobileMenu.querySelectorAll('a').forEach(link => {
                link.addEventListener('click', () => {
                    mobileMenu.classList.remove('open');
                    menuBtn.setAttribute('aria-expanded', 'false');
                });
            });
        }

        // Init Lucide icons
        if (typeof lucide !== 'undefined') lucide.createIcons();
    });
})();
