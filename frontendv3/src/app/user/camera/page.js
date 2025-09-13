"use client";

import { useAuth } from '@/context/authContext';
import axiosInstance from "@/utils/axiosInstance";
import {
  CalculatorOutlined,
  HeartFilled,
  HeartOutlined,
  RedoOutlined,
  VideoCameraOutlined
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Col,
  Row,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
  message
} from 'antd';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from "react";

const { Title, Text, Paragraph } = Typography;
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL_BE || "http://localhost:8000";

// For initial mock display until real calculation is performed
const getCongestionLevel = (cameraId) => {
  let hash = 0;
  for (let i = 0; i < cameraId.length; i++) {
    hash = cameraId.charCodeAt(i) + ((hash << 5) - hash);
  }
  // Ensure result is between 1 and 5 (inclusive) for mock data
  return (Math.abs(hash) % 5) + 1; // Changed % 6 to % 5
};

// Congestion level text descriptions (Adjusted to 5 levels)
const getCongestionText = (level) => {
  const texts = {
    1: "Free flowing",
    2: "Light traffic",
    3: "Moderate traffic",
    4: "Heavy traffic",
    5: "Very heavy / Jammed", // Combined last level text
  };
  return texts[level] || "Unknown";
};

// Traffic level colors (Adjusted to 5 levels)
const getTrafficLevelColor = (level) => {
  const colors = {
    1: '#52c41a', // Free flowing - Green
    2: '#8bc34a', // Light traffic - Light Green
    3: '#fadb14', // Moderate traffic - Yellow
    4: '#fa8c16', // Heavy traffic - Orange
    5: '#f5222d', // Very heavy / Jammed - Red (using the previous level 5 color)
  };
  return colors[level] || 'grey';
};

export default function CameraListPage() {
  const router = useRouter();
  const { isAuthenticated, user } = useAuth();
  const [messageApi, contextHolder] = message.useMessage();

  const [cameras, setCameras] = useState([]);
  const [favoriteCameraIds, setFavoriteCameraIds] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [isCapturingThumbnails, setIsCapturingThumbnails] = useState(false);

  // State for bulk calculation
  const [calculatingAllCongestion, setCalculatingAllCongestion] = useState(false);

  const fetchCameraData = useCallback(async () => {
    // setLoading(true); // Only set loading on initial load or full refresh
    setError('');
    try {
      // User endpoint gets only ACTIVE cameras
      const response = await axiosInstance.get(`${API_BASE_URL}/api/cameras`);
      const fetchedCameras = response.data || [];
      console.log("Camera: ", response.data)
      // Apply initial mock congestion or previously fetched real data if available
      setCameras(currentCameras => {
        // Create a map of current cameras for efficient lookup
        const currentCameraMap = new Map(currentCameras.map(c => [c.id, c]));

        return fetchedCameras.map(fetchedCamera => {
          const existingCamera = currentCameraMap.get(fetchedCamera.id);
          const initialCongestionLevel = (existingCamera && existingCamera.real_congestion)
            ? existingCamera.congestion_level
            : getCongestionLevel(fetchedCamera.id); // Fallback to mock
          const initialCongestionText = (existingCamera && existingCamera.real_congestion)
            ? existingCamera.congestion_text
            : getCongestionText(initialCongestionLevel);

          return {
            ...fetchedCamera, // Take all data from the latest fetch
            congestion_level: initialCongestionLevel,
            congestion_text: initialCongestionText,
            // Preserve real_congestion flag if it existed
            real_congestion: existingCamera ? existingCamera.real_congestion : false,
            // Ensure ROI data from backend is preserved/updated
            roi: fetchedCamera.roi,
          };
        });
      });

    } catch (err) {
      console.error('Error fetching camera data:', err);
      const errorMsg = err.response?.data?.detail || 'Failed to load camera data. Please try again later.';
      setError(errorMsg);
      messageApi.error(errorMsg); // Show error to user
    } finally {
      // Loading state managed by initial useEffect and fetchFavoriteCameras
    }
  }, [messageApi]); // Depend only on messageApi

  const fetchFavoriteCameras = useCallback(async () => {
    if (!isAuthenticated) {
      // setLoading(false); // Don't set loading false here, let the main useEffect handle it
      return;
    }
    try {
      const response = await axiosInstance.get(`${API_BASE_URL}/api/user/favorites`);
      const favIds = new Set((response.data || []).map(fav => fav.camera_id));
      setFavoriteCameraIds(favIds);
    } catch (err) {
      console.error('Error fetching favorite cameras:', err);
      // Do not message.error here as it can be noisy during background updates
    } finally {
      setLoading(false); // Stop loading ONLY after favorites are fetched (or skipped)
    }
  }, [isAuthenticated]);

  useEffect(() => {
    setLoading(true); // Set loading true on mount
    fetchCameraData();
    if (isAuthenticated) {
      fetchFavoriteCameras(); // This will set loading false in its finally block
    } else {
      setLoading(false); // If not authenticated, stop loading after fetching cameras
    }
    // Only run on mount and when auth status changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated]);

  const handleRefreshThumbnails = async () => {
    if (isCapturingThumbnails || cameras.length === 0) {
      messageApi.info(isCapturingThumbnails ? "Thumbnail refresh already in progress." : "No cameras to refresh.");
      return;
    }

    setIsCapturingThumbnails(true);
    messageApi.loading({ content: `Requesting thumbnail refresh for all cameras...`, key: 'thumbnailRefreshAll', duration: 0 });

    try {
      // Get only Active camera IDs for refresh request, as this is the user page
      const activeCameraIds = cameras.filter(c => c.status === 'Active').map(c => c.id);
      if (activeCameraIds.length === 0) {
        messageApi.info("No active cameras to refresh thumbnails for.");
        setIsCapturingThumbnails(false);
        messageApi.destroy('thumbnailRefreshAll');
        return;
      }

      // NOTE: The backend endpoint /api/cameras/refresh-all-thumbnails seems designed to refresh ALL cameras,
      // not just specific ones. We'll call it as is, but be aware it might refresh inactive ones too.
      // If granular control is needed later, the backend API would need adjustment.
      const response = await axiosInstance.post(`${API_BASE_URL}/api/cameras/refresh-all-thumbnails`);
      messageApi.destroy('thumbnailRefreshAll');

      const data = response?.data || {};
      const { success_count, fail_count, total_cameras, message: response_message } = data;

      // More detailed feedback based on counts
      if (typeof total_cameras === 'number') {
        let summary = `Thumbnail refresh request sent.`; // Simpler initial message
        // Report based on returned counts if available
        if (typeof success_count === 'number' && typeof fail_count === 'number') {
          summary = `Refreshed ${success_count} thumbnails`;
          if (fail_count > 0) summary += ` (${fail_count} failed).`; else summary += `.`;
        }

        if (fail_count > 0) {
          messageApi.warning(summary);
        } else if (success_count > 0 || (success_count === 0 && fail_count === 0 && total_cameras > 0)) {
          // Show success even if 0 succeeded but 0 failed (e.g., if backend filtered)
          messageApi.success(summary);
        } else {
          messageApi.info(response_message || summary || "Refresh process initiated, checking results...");
        }
      } else {
        messageApi.info(response_message || "Thumbnail refresh request sent. Re-fetching data...");
      }

      // Wait a short delay for backend processing and file system updates
      await new Promise(resolve => setTimeout(resolve, 1500)); // Adjust delay as needed

      // Automatically fetch fresh camera data after thumbnails are refreshed
      await fetchCameraData(); // Fetch fresh data to show new thumbnails

    } catch (err) {
      console.error("Error requesting thumbnail refresh for all cameras:", err);
      messageApi.destroy('thumbnailRefreshAll');
      messageApi.error(err.response?.data?.detail || "Failed to initiate thumbnail refresh process.");
    }

    setIsCapturingThumbnails(false);
  };

  const handleCalculateAllCongestion = async () => {
    if (calculatingAllCongestion || cameras.length === 0) {
      messageApi.info(calculatingAllCongestion ? "Congestion calculation already in progress." : "No cameras to calculate congestion.");
      return;
    }

    setCalculatingAllCongestion(true);
    messageApi.loading({ content: `Calculating congestion for all cameras...`, key: 'calculateAllCongestion', duration: 0 });

    try {
      // Call the backend endpoint that calculates for ALL cameras
      const response = await axiosInstance.post(`${API_BASE_URL}/api/cameras/calculate-all-congestion`);
      messageApi.destroy('calculateAllCongestion');

      const responseData = response.data; // Should be AllCongestionResponse structure
      if (!responseData || !responseData.results) {
        throw new Error("Invalid response structure from congestion calculation endpoint.");
      }

      const calculationResults = responseData.results; // Array of CongestionResponseItem
      let successCount = 0;
      let failCount = 0;
      let roiMissingCount = 0;

      // Update the cameras state with the new congestion levels
      setCameras(prevCameras => {
        // Create a map for efficient lookup of results by camera_id
        const resultMap = new Map(calculationResults.map(res => [res.camera_id, res]));

        return prevCameras.map(cam => {
          const result = resultMap.get(cam.id);
          if (result) {
            if (result.success) {
              successCount++;
              return {
                ...cam,
                congestion_level: result.congestion_level,
                congestion_text: result.congestion_text,
                real_congestion: true // Mark as real data
              };
            } else {
              failCount++;
              if (result.error && result.error.toLowerCase().includes("roi not defined")) {
                roiMissingCount++;
              }
              // Keep previous state but mark as not real/error
              return {
                ...cam,
                // Keep mock level for display? Or set to 0? Let's keep mock/previous level.
                // congestion_level: 0, 
                congestion_text: result.error || "Calculation Failed",
                real_congestion: false // Mark as not real/error
              };
            }
          }
          // If a camera from frontend state wasn't in the calculation results (e.g., inactive)
          // return it unchanged but ensure real_congestion is false
          return { ...cam, real_congestion: false };
        });
      });

      // Display summary message
      let summary = `Congestion calculation attempted for ${responseData.cameras_processed} cameras.`;
      if (successCount > 0) summary += ` ${successCount} succeeded.`;
      if (failCount > 0) summary += ` ${failCount} failed`;
      if (roiMissingCount > 0) summary += ` (${roiMissingCount} due to missing/incomplete ROI).`;
      else if (failCount > 0) summary += `.`; // Add period if fails not due to ROI

      if (failCount === 0 && successCount > 0) {
        messageApi.success(summary);
      } else if (failCount > 0) {
        messageApi.warning(summary);
      } else {
        messageApi.info(summary); // E.g., if 0 cameras were processed
      }

      // No need to call fetchCameraData again here, as we updated the state directly

    } catch (err) {
      console.error("Error calculating congestion for all cameras:", err);
      messageApi.destroy('calculateAllCongestion');
      messageApi.error(err.response?.data?.detail || "Failed to initiate congestion calculation process.");
    }

    setCalculatingAllCongestion(false);
  };

  const toggleFavorite = async (camera_id, isCurrentlyFavorite) => {
    if (!isAuthenticated) {
      messageApi.warning("Please log in to manage favorite cameras.");
      return;
    }
    // Optimistic UI update
    const originalFavIds = new Set(favoriteCameraIds);
    setFavoriteCameraIds(prev => {
      const newSet = new Set(prev);
      if (isCurrentlyFavorite) {
        newSet.delete(camera_id);
      } else {
        newSet.add(camera_id);
      }
      return newSet;
    });

    try {
      if (isCurrentlyFavorite) {
        await axiosInstance.delete(`${API_BASE_URL}/api/user/favorites/${camera_id}`);
        messageApi.success("Removed from favorites");
      } else {
        await axiosInstance.post(`${API_BASE_URL}/api/user/favorites`, { camera_id });
        messageApi.success("Added to favorites");
      }
    } catch (err) {
      console.error("Error toggling favorite:", err);
      messageApi.error(err.response?.data?.detail || "Failed to update favorites.");
      // Revert UI on error
      setFavoriteCameraIds(originalFavIds);
    }
  };

  const handleCameraClick = (camera) => {
    if (camera && camera.id) {
      router.push(`/user/camera/${camera.id}`);
    } else {
      console.error("Camera object or camera.id is undefined", camera);
      messageApi.error("Cannot open camera details: Camera ID is missing.");
    }
  };

  const renderCameraCard = (camera) => {
    // Use camera.congestion_level directly, which is now updated by handleCalculateAllCongestion
    // Fallback to mock level if never calculated or calculation failed
    const congestionLevel = camera.real_congestion ? camera.congestion_level : getCongestionLevel(camera.id);
    const congestionText = camera.real_congestion ? camera.congestion_text : getCongestionText(congestionLevel);
    const isFavorite = favoriteCameraIds.has(camera.id);

    // Explicitly check for empty string or null/undefined for camera.thumbnail_url
    const baseThumbnailUrl = (camera.thumbnail_url && typeof camera.thumbnail_url === 'string' && camera.thumbnail_url.trim() !== '') 
      ? camera.thumbnail_url 
      : null;

    const imageSrc = baseThumbnailUrl
      ? (baseThumbnailUrl.startsWith('http')
          ? baseThumbnailUrl
          : `${API_BASE_URL}${baseThumbnailUrl}?v=${new Date(camera.updated_at || Date.now()).getTime()}`)
      : null; 

    // Determine tag color based on level or if it's mock/error
    const levelColor = camera.real_congestion ? getTrafficLevelColor(congestionLevel) : 'default'; // Use default color for mock

    return (
      <Col xs={24} sm={12} md={8} lg={6} key={camera.id}>
        <Card
          hoverable
          style={{ marginBottom: '16px' }} // Consistent margin
          cover={
            <div
              style={{ height: '160px', position: 'relative', backgroundColor: '#e0e0e0', overflow: 'hidden', cursor: 'pointer' }} // Background shows if no image
              onClick={() => handleCameraClick(camera)}
            >
              {imageSrc && ( // Conditionally render Image component only if imageSrc is truthy
                <Image
                  src={imageSrc}
                  alt={camera.name || 'Traffic camera'}
                  fill
                  style={{ objectFit: "cover" }}
                  onError={(e) => {
                    console.warn(`Failed to load image: ${imageSrc}`);
                    // Hide the image element itself on error, letting the background show
                    e.target.style.display = 'none';
                  }}
                  unoptimized={imageSrc && !imageSrc.startsWith('/')} // Check imageSrc before calling startsWith
                  priority={cameras.indexOf(camera) < 4}
                />
              )}
              {/* Display calculated level tag only if real_congestion is true and level > 0 */}
              {camera.real_congestion && congestionLevel > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  backgroundColor: 'rgba(0,0,0,0.6)',
                  padding: '4px 8px',
                  borderRadius: '12px'
                }}>
                  <Tag color={getTrafficLevelColor(congestionLevel)} style={{ margin: 0 }}> {/* Use actual color here */}
                    Level {congestionLevel}
                  </Tag>
                </div>
              )}
            </div>
          }
          actions={[
            // REMOVED individual calculate button

            isAuthenticated ? (
              <Button
                type="text"
                icon={isFavorite ? <HeartFilled style={{ color: 'red' }} /> : <HeartOutlined />}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleFavorite(camera.id, isFavorite);
                }}
                key={`fav-${camera.id}`}
              >
                {isFavorite ? 'Saved' : 'Save'}
              </Button>
            ) : null,
          ].filter(Boolean) // Filter out null actions
          }
          styles={{ body: { padding: '12px' } }}
        >
          {/* Tooltip shows detailed info on hover */}
          <Tooltip
            title={
              <div>
                <p><strong>Name:</strong> {camera.name}</p>
                <p><strong>Lat:</strong> {`${camera.location_detail.latitude}` || 'N/A'}</p>
                <p><strong>Long:</strong> {`${camera.location_detail.longitude}` || 'N/A'}</p>
                <p><strong>Traffic:</strong> {congestionText}</p>
                <p><strong>Status:</strong> {camera.status}</p>
              </div>
            }
          >
            {/* Clickable area for navigation */}
            <div onClick={() => handleCameraClick(camera)} style={{ cursor: 'pointer' }}>
              <Card.Meta
                title={<Text >{camera.name}</Text>}
                description={
                  <>
                    <Text type="secondary" ellipsis={true} style={{ maxWidth: '100%' }}>
                      {`  ${camera.location_detail.name}` || 'Unknown Location'}
                    </Text>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px' }}>
                      {/* Display tag based on real or mock data */}
                      <Tag color={levelColor}>
                        {congestionText}
                      </Tag>
                      {/* Status tag remains the same */}
                      <Tag color={camera.status === 'Active' ? 'green' : (camera.status === 'Maintenance' ? 'orange' : 'red')}>
                        {camera.status}
                      </Tag>
                    </div>
                  </>
                }
              />
            </div>
          </Tooltip>
        </Card>
      </Col>
    );
  };

  // Loading spinner logic 
  if (loading) { // Show spinner if loading is true (covers initial load and favorite fetching)
    return (
      <div style={{ padding: '24px', textAlign: 'center', minHeight: 'calc(100vh - 200px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Spin size="large" tip="Loading cameras..." />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      {contextHolder}
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <Title level={2} style={{ marginBottom: '0px' }}><VideoCameraOutlined style={{ marginRight: '12px' }} /> Traffic Cameras</Title>
          <Paragraph style={{ marginTop: '4px' }}>View real-time camera feeds and manage your favorites.</Paragraph>
        </div>
        <Space wrap> {/* Added wrap to Space */}
          <Button
            icon={<CalculatorOutlined />}
            onClick={handleCalculateAllCongestion}
            loading={calculatingAllCongestion}
            // Disable if no active cameras or already calculating
            disabled={cameras.filter(cam => cam.status === 'Active').length === 0 || calculatingAllCongestion}
          >
            Update Congestion Levels
          </Button>
          {/* Restore Refresh Thumbnails Button */}
          <Button
            icon={<RedoOutlined />}
            onClick={handleRefreshThumbnails}
            loading={isCapturingThumbnails}
            // Disable if no active cameras or already refreshing
            disabled={cameras.filter(cam => cam.status === 'Active').length === 0 || isCapturingThumbnails}
          >
            Refresh Thumbnails
          </Button>
        </Space>
      </div>

      {error && (
        <Alert message="Error" description={error} type="error" showIcon style={{ marginBottom: '24px' }} />
      )}

      {/* Check specifically for active cameras before showing "No Cameras" message */}
      {(!loading && cameras.filter(cam => cam.status === 'Active').length === 0 && !error) && (
        <Alert message="No Active Cameras" description="There are currently no active cameras to display." type="info" showIcon style={{ marginBottom: '24px' }} />
      )}

      <Row gutter={[16, 16]}>
        {/* Ensure only Active cameras are rendered on this user page */}
        {cameras.filter(cam => cam.status === 'Active').map(camera => renderCameraCard(camera))}
      </Row>

      {/* REMOVED Congestion Modal */}
    </div>
  );
}