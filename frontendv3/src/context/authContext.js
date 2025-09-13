"use client";

import React, {
  createContext,
  useState,
  useContext,
  useEffect,
  useCallback,
} from "react";
import { useRouter } from "next/navigation";
import axiosInstance from "../utils/axiosInstance"; // Import the centralized axiosInstance
import Cookies from 'js-cookie';

const AuthContext = createContext(null);
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL_BE || "http://localhost:8000"; // This can remain if used for non-axiosInstance calls, or be removed if all calls go via instance

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState(null);
  const [permissions, setPermissions] = useState([]);
  const router = useRouter();

  // Removed the useEffect that set up global axios defaults and interceptors
  // All API calls will now use axiosInstance which has this configuration.

  // Function to set user role in a cookie for middleware access
  const setRoleCookie = (role) => {
    try {
      // Set a cookie with the user's role that expires in 1 day
      Cookies.set('user_role', role, { expires: 1, path: '/' });
    } catch (error) {
      console.error("Error setting role cookie:", error);
    }
  };

  // Function to clear role cookie
  const clearRoleCookie = () => {
    try {
      Cookies.remove('user_role', { path: '/' });
    } catch (error) {
      console.error("Error clearing role cookie:", error);
    }
  };
  
  // Clear all user data (for logout or auth errors)
  const clearUserData = useCallback(() => { // Wrapped in useCallback as it's used in refresh error handling potentially
    setUser(null);
    setIsAuthenticated(false);
    setPermissions([]);
    // Cookies.remove('access_token'); // Ensure access_token is also cleared
    clearRoleCookie();
  }, []);

  // Function to fetch current user data
  const fetchUser = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Use axiosInstance for the API call
      const response = await axiosInstance.get("/api/auth/me"); // Path is relative to baseURL in axiosInstance
      
      if (response.data) {
        setUser(response.data);
        setIsAuthenticated(true);
        
        if (response.data.permissions) {
          setPermissions(response.data.permissions);
        } else {
          const rolePermissions = response.data.role === 'Admin' 
            ? ['view_traffic', 'find_route', 'manage_cameras', 'manage_models', 'manage_reports', 'manage_users'] 
            : ['view_traffic', 'find_route', 'view_cameras', 'submit_report'];
          setPermissions(rolePermissions);
        }
        
        setRoleCookie(response.data.role);
        return true;
      }
      return false;
    } catch (error) {
      console.log("Not authenticated or error fetching user data", error);
      clearUserData();
      // If the error came from a failed token refresh, redirect to login
      if (error.message && error.message.includes("Session refresh failed")) {
        router.push('/auth/login');
      }
      
      if (error.response && error.response.status !== 401) {
        setError(error.response?.data?.detail || "Error fetching user data");
      }
      return false;
    } finally {
      setLoading(false);
    }
  }, [clearUserData, router]); // Added clearUserData and router to dependency array

  // Login function with error handling
  const login = async (username, password) => {
    try {
      setLoading(true);
      setError(null);
      
      const formData = new URLSearchParams();
      formData.append("username", username);
      formData.append("password", password);
      
      // Use axiosInstance for the API call
      const response = await axiosInstance.post(
        "/api/auth/login", // Path is relative to baseURL
        formData,
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          // withCredentials: true, // This is now part of axiosInstance default config
        }
      );
      
      // if (response.data && response.data.access_token) {
      //   Cookies.set('access_token', response.data.access_token, { expires: 1 }); // Store the token
      // } 

      if (response.data && response.data.user) {
        setUser(response.data.user);
        setIsAuthenticated(true);
        
        if (response.data.user.permissions) {
          setPermissions(response.data.user.permissions);
        } else {
          const rolePermissions = response.data.user.role === 'Admin' 
            ? ['view_traffic', 'find_route', 'manage_cameras', 'manage_models', 'manage_reports', 'manage_users'] 
            : ['view_traffic', 'find_route', 'view_cameras', 'submit_report'];
          setPermissions(rolePermissions);
        }
        
        setRoleCookie(response.data.user.role);
        return true;
      } else if (response.data && response.data.access_token) {
        // If user data is not in login response, but token is, fetch user data
        return await fetchUser(); 
      }
      
      return false;
    } catch (error) {
      console.error("Login error:", error);
      if (error.response && error.response.data) {
        setError(error.response.data.detail || "Login failed");
      } else if (error.request) {
        setError("No response from server. Check network or if server is running.");
      } else {
        setError("Login request failed. Please try again.");
      }
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Use axiosInstance for the API call
      await axiosInstance.post("/api/auth/logout"); // Path is relative to baseURL
      
    } catch (error) {
      console.error("Logout API call error:", error); 
      // Even if API fails, proceed with client-side logout
    } finally {
      clearUserData();
      router.push('/auth/login'); 
      setLoading(false);
    }
  };
  
  const hasPermission = (permission) => {
    return permissions.includes(permission);
  };

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAuthenticated,
        error,
        permissions,
        hasPermission,
        login,
        logout,
        fetchUser,
        clearUserData // Expose clearUserData if needed by other parts of app
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
