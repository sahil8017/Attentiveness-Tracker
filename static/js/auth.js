// ============================================================
// Auth utility — JWT storage, auth headers, redirects
// ============================================================

const AUTH = {
    TOKEN_KEY: 'auth_token',
    USER_KEY: 'auth_user',

    // --- Token management ---

    getToken() {
        return localStorage.getItem(this.TOKEN_KEY);
    },

    setToken(token) {
        localStorage.setItem(this.TOKEN_KEY, token);
    },

    getUser() {
        try {
            return JSON.parse(localStorage.getItem(this.USER_KEY));
        } catch { return null; }
    },

    setUser(user) {
        localStorage.setItem(this.USER_KEY, JSON.stringify(user));
    },

    clearAuth() {
        localStorage.removeItem(this.TOKEN_KEY);
        localStorage.removeItem(this.USER_KEY);
    },

    isAuthenticated() {
        const token = this.getToken();
        if (!token) return false;

        // Basic JWT expiry check (decode payload without verification)
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            if (payload.exp && payload.exp * 1000 < Date.now()) {
                console.warn('AUTH: Token expired, clearing');
                this.clearAuth();
                return false;
            }
        } catch {
            // If we can't decode, treat as invalid
            this.clearAuth();
            return false;
        }
        return true;
    },

    logout() {
        this.clearAuth();
        window.location.href = '/login';
    },

    /**
     * Centralized fetch wrapper — attaches JWT Bearer token to every request.
     * On 401: clears auth and redirects to /login.
     * On abort (AbortController): rethrows so callers can handle gracefully.
     */
    async apiFetch(url, options = {}) {
        const token = this.getToken();

        // Build headers: merge caller's headers + Authorization
        const headers = { ...(options.headers || {}) };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        // Default Content-Type for non-FormData bodies
        if (!headers['Content-Type'] && options.body && !(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }

        let response;
        try {
            response = await fetch(url, { ...options, headers });
        } catch (err) {
            // AbortError or network failure — rethrow as-is, do NOT redirect
            throw err;
        }

        // Only redirect on genuine 401 (not other errors)
        if (response.status === 401) {
            console.warn('AUTH: 401 received from', url, '— redirecting to login');
            this.clearAuth();
            window.location.href = '/login';
            // Return a never-resolving promise so callers don't continue processing
            return new Promise(() => { });
        }

        return response;
    },

    /**
     * Page-level auth guard: redirect unauthenticated users to login.
     * Returns true if authenticated, false + redirect if not.
     */
    requireAuth() {
        if (!this.isAuthenticated()) {
            window.location.href = '/login';
            return false;
        }
        return true;
    },

    /**
     * Update nav UI: show user email + logout button, or login link.
     */
    updateNavUI() {
        const user = this.getUser();
        const isLoggedIn = this.isAuthenticated() && user;

        document.querySelectorAll('.auth-user-btn').forEach(el => {
            el.style.display = isLoggedIn ? 'flex' : 'none';
            el.classList.toggle('hidden', !isLoggedIn);
        });
        document.querySelectorAll('.auth-login-link').forEach(el => {
            el.style.display = isLoggedIn ? 'none' : '';
            el.classList.toggle('hidden', isLoggedIn);
        });

        if (isLoggedIn) {
            document.querySelectorAll('.auth-email').forEach(el => {
                el.textContent = user.email;
            });
            document.querySelectorAll('.auth-logout-btn').forEach(btn => {
                // Avoid duplicate listeners by cloning
                const newBtn = btn.cloneNode(true);
                btn.parentNode.replaceChild(newBtn, btn);
                newBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    AUTH.logout();
                });
            });
        }
    }
};

// Auto-update nav on page load
document.addEventListener('DOMContentLoaded', () => { AUTH.updateNavUI(); });
