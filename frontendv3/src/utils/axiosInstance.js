import axios from 'axios';
import Cookies from 'js-cookie'; // Import js-cookie

// Determine the base URL from environment variables, with a fallback
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL_BE || 'http://localhost:8000';

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json', // Added Accept header as in authContext
  },
  withCredentials: true, // Added withCredentials as in authContext
});

// Request interceptor to add the auth token to requests
// axiosInstance.interceptors.request.use(
//   (config) => {
//     const token = Cookies.get('access_token'); // Use Cookies to get the token
//     if (token) {
//       config.headers['Authorization'] = `Bearer ${token}`;
//     }
//     return config;
//   },
//   (error) => {
//     return Promise.reject(error);
//   }
// );

// Response interceptor to handle token refresh and global errors
axiosInstance.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    // Paths that should not trigger automatic token refresh
    const noRefreshPaths = ['/api/auth/login', '/api/auth/register', '/api/auth/refresh'];

    if (
      error.response?.status === 401 &&
      !originalRequest._retry &&
      originalRequest.url && // Ensure url is defined
      !noRefreshPaths.some(path => originalRequest.url.includes(path))
    ) {
      originalRequest._retry = true;
      try {
        // Attempt to refresh the token
        console.log('axiosInstance: Attempting to refresh token...');
        const refreshResponse = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {}, { withCredentials: true });
        
        if (refreshResponse.data && refreshResponse.data.access_token) {
          const newAccessToken = refreshResponse.data.access_token;
          console.log('axiosInstance: Token refreshed successfully. New access token:', newAccessToken);
          Cookies.set('access_token', newAccessToken, { expires: 1, path: '/' }); // Store new token
          // The request interceptor will pick up the new token for the Authorization header.
          // The browser will also have the new HTTP-only access_token if the backend set it correctly.
          return axiosInstance(originalRequest); // Retry original request
        } else {
          // Refresh succeeded but response format is unexpected
          console.error("axiosInstance: Refresh response did not contain access_token.", refreshResponse);
          // Propagate a more specific error or handle as per your app's logic
          return Promise.reject(new Error('Token refresh succeeded but no new token received.'));
        }

      } catch (refreshError) {
        // If refresh fails, we can't easily call clearUserData or router.push here
        // as this is a utility file. The component using axiosInstance should handle this.
        console.error("axiosInstance: Session refresh failed:", refreshError);
        // Instead of redirecting here, let the caller handle the final rejection.
        // For example, authContext's fetchUser can call clearUserData and redirect.
        // Add a more specific error message to be caught by authContext
        const detailedError = new Error("Session refresh failed via axiosInstance");
        detailedError.originalError = refreshError;
        return Promise.reject(detailedError);
      }
    }
    return Promise.reject(error);
  }
);

export default axiosInstance; 