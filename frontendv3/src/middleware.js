// middleware.js
import { NextResponse } from "next/server";

const loginPath = "/auth/login";
const defaultAdminPath = "/admin/dashboard";
const defaultUserPath = "/user/map";

// Base paths that have moved under /user/
// These are the *old* base paths.
const movedUserBasePaths = [
  "/map",
  "/settings",
  "/camera",
  "/traffic-analysis",
  "/reports",
  "/support",
  "/news-weather",
  // Add any other base paths that were at root and moved under /user/
  // For example, if you had /favorites, it would be listed here if now /user/favorites
];

export function middleware(request) {
  const { pathname } = request.nextUrl;
  const authCookie = request.cookies.get('access_token');
  const hasToken = !!authCookie;
  const roleCookie = request.cookies.get('user_role');
  const userRole = roleCookie?.value;

  // 1. Bypass API routes
  if (pathname.startsWith('/api')) {
    return NextResponse.next();
  }

  // 2. Handle root path ("/")
  if (pathname === "/") {
    if (!hasToken) {
      return NextResponse.redirect(new URL(loginPath, request.url));
    }
    return NextResponse.redirect(new URL(userRole?.toLowerCase() === "admin" ? defaultAdminPath : defaultUserPath, request.url));
  }

  // 3. Handle login page access when already authenticated
  if (pathname === loginPath) {
    if (hasToken) {
      return NextResponse.redirect(new URL(userRole?.toLowerCase() === "admin" ? defaultAdminPath : defaultUserPath, request.url));
    }
    return NextResponse.next(); // Allow access to login page if not authenticated
  }

  // 4. Redirect old root paths (and their subpaths) to new /user/ prefixed paths
  for (const oldBasePath of movedUserBasePaths) {
    // Check if the current pathname is exactly the old base path or starts with the old base path followed by a slash
    if (pathname === oldBasePath || pathname.startsWith(oldBasePath + "/")) {
      const newPath = `/user${pathname}`; // Prepend /user to the entire old path
      return NextResponse.redirect(new URL(newPath, request.url));
    }
  }

  // 5. Protect /admin/* routes
  if (pathname.startsWith("/admin")) {
    if (!hasToken) {
      const callbackUrl = new URL(loginPath, request.url);
      callbackUrl.searchParams.set('callbackUrl', encodeURI(request.url.toString()));
      return NextResponse.redirect(callbackUrl);
    }
    if (userRole?.toLowerCase() !== "admin") {
      // Non-admin trying to access admin page, redirect to their default user page
      return NextResponse.redirect(new URL(defaultUserPath, request.url));
    }
    // Admin is accessing admin page, allow
    return NextResponse.next();
  }

  // 6. Protect /user/* routes
  if (pathname.startsWith("/user")) {
    if (!hasToken) {
      const callbackUrl = new URL(loginPath, request.url);
      callbackUrl.searchParams.set('callbackUrl', encodeURI(request.url.toString()));
      return NextResponse.redirect(callbackUrl);
    }
    // Logged-in users (admin or regular) can access /user/* pages.
    // This fulfills "Admin can use user page".
    // "but not the other way around" is handled by rule 5 for /admin paths.
    return NextResponse.next();
  }

  // For any other paths not covered, let them pass.
  // This might include public static assets if not handled by the matcher, or other public pages.
  return NextResponse.next();
}

export const config = {
  matcher: [
    "/", // Root path
    "/auth/login", // Login page
    "/admin/:path*", // All admin routes
    "/user/:path*", // All new user routes

    // Old root paths (and their sub-paths) that need to be caught by the middleware for redirection.
    // This ensures the middleware runs for these paths to apply the redirection logic.
    "/map", "/map/:path*",
    "/settings", "/settings/:path*",
    "/camera", "/camera/:path*",
    "/traffic-analysis", "/traffic-analysis/:path*",
    "/reports", "/reports/:path*", 
    "/support", "/support/:path*",
    "/news-weather", "/news-weather/:path*",
    // Add other old root-level path patterns here if they existed and moved
  ],
};
