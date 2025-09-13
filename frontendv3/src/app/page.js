'use client';

import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import { useAuth } from '@/context/authContext'; // Import useAuth to potentially check auth status if needed

const HomePage = () => {
  const router = useRouter();
  const { isAuthenticated, loading, user } = useAuth(); // Get auth state from context

  useEffect(() => {
    // The middleware should handle initial redirection from "/".
    // This useEffect can act as a fallback or secondary check if the user somehow lands here
    // after initial load, or if we want to react to auth state changes while on this page.
    if (!loading) { // Only act once auth context is loaded
      if (isAuthenticated && user) {
        if (user.role?.toLowerCase() === 'admin') {
          router.replace('/admin/dashboard');
        } else {
          router.replace('/user/map');
        }
      } else if (!isAuthenticated) {
        router.replace('/auth/login');
      }
    }
  }, [router, isAuthenticated, loading, user]);

  // It's generally better to rely on the middleware for the initial redirect from "/".
  // This component can return null or a loading indicator.
  // If middleware is fully effective, this useEffect might become redundant for the initial load case.
  return null; // Render nothing, or a global loading spinner
};

export default HomePage;
