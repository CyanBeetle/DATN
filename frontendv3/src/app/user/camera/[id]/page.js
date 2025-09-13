"use client";

import axiosInstance from '@/utils/axiosInstance';
import { CameraOutlined, HomeOutlined, VideoCameraOutlined } from '@ant-design/icons';
import { Alert, Avatar, Breadcrumb, Card, Col, List, Row, Spin, Tag, Typography } from 'antd';
import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

const { Title, Paragraph, Text } = Typography;
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL_BE || "http://localhost:8000";

export default function CameraDetailPage() {
  const params = useParams();
  const router = useRouter();
  const cameraId = params?.id;
  const [camera, setCamera] = useState(null);
  const [otherCameras, setOtherCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (cameraId) {
      setLoading(true);
      setError(null);
      setCamera(null);
      setOtherCameras([]);

      const fetchCurrentCamera = axiosInstance.get(`/api/cameras/${cameraId}`);
      const fetchAllCameras = axiosInstance.get('/api/cameras');

      Promise.all([fetchCurrentCamera, fetchAllCameras])
        .then(([currentCameraResponse, allCamerasResponse]) => {
          setCamera(currentCameraResponse.data);

          const filteredCameras = (allCamerasResponse.data || []).filter(
            (cam) => cam.id !== cameraId
          );
          setOtherCameras(filteredCameras);
          setError(null);
        })
        .catch(err => {
          console.error("Error fetching camera data:", err);
          let errMsg = "Failed to load camera data.";
          if (err.isAxiosError && err.response) {
            errMsg = err.response.data?.detail || "Failed to load camera data. The camera might be inactive or not found.";
          } else if (err.message) {
            errMsg = err.message;
          }
          setError(errMsg);
          setCamera(null);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setError("Camera ID not provided.");
      setLoading(false);
    }
  }, [cameraId]);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 200px)', padding: '20px' }}>
        <Spin size="large" tip="Loading camera details..." />
      </div>
    );
  }

  if (error && !camera) {
    return (
      <div style={{ padding: '20px' }}>
        <Alert message="Error" description={error} type="error" showIcon />
      </div>
    );
  }

  if (!camera) {
    return (
      <div style={{ padding: '20px' }}>
        <Alert message="Camera Not Found" description="The selected camera could not be found or is not available." type="warning" showIcon />
      </div>
    );
  }

  // Construct breadcrumb items for the new API
  const breadcrumbItems = [
    {
      // href: '/user/map', // Link handled in items mapping
      title: <HomeOutlined />,
      path: '/user/map',
    },
    {
      // href: '/user/camera',
      title: (
        <>
          <VideoCameraOutlined />
          <span> Cameras</span>
        </>
      ),
      path: '/user/camera',
    },
    {
      title: camera.name || 'Camera Details',
      // No path for the current page
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Breadcrumb 
        style={{ marginBottom: '16px' }} 
        items={breadcrumbItems.map(item => ({
          key: item.path || item.title, // Add key for React list rendering
          title: item.path ? <Link href={item.path}>{item.title}</Link> : item.title,
        }))}
      />

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={18}>
          <Title level={2} style={{ marginBottom: '8px' }}>{camera.name}</Title>
          <Paragraph type="secondary" style={{ marginBottom: '24px' }}>
            {camera.description || "Live stream from the selected camera."}
          </Paragraph>

          <Card
            title="Live Stream"
            style={{ marginBottom: '24px' }}
            styles={{
              root: { borderWidth: '0px' },
              head: { borderBottom: 'none' }
            }}
          >
            {camera.stream_url ? (
              <iframe
                src={camera.stream_url}
                title={camera.name}
                style={{
                  width: '100%',
                  height: 'calc(100vh - 420px)',
                  minHeight: '400px',
                  border: '1px solid #f0f0f0',
                  borderRadius: '8px'
                }}
                allowFullScreen
              />
            ) : (
              <Text>Stream URL is not available for this camera.</Text>
            )}
          </Card>

          <Card
            title="Camera Information"
            styles={{ root: { borderWidth: '0px' } }}
          >
            <Paragraph><strong>Status:</strong> <Tag color={camera.status === 'Active' ? 'green' : 'red'}>{camera.status}</Tag></Paragraph>
            <Paragraph><strong>Location:</strong> {camera.location_detail.name || 'N/A'}</Paragraph>
            {camera.location?.latitude && camera.location?.longitude && (
              <Paragraph>
                <strong>Coordinates:</strong> {camera.location.latitude.toFixed(6)}, {camera.location.longitude.toFixed(6)}
              </Paragraph>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={6}>
          <Card
            title="Other Cameras"
            styles={{
              root: { borderWidth: '0px' },
              body: { maxHeight: 'calc(100vh - 300px)', overflowY: 'auto' }
            }}
          >
            {otherCameras.length > 0 ? (
              <List
                itemLayout="horizontal"
                dataSource={otherCameras}
                renderItem={item => {
                  const itemImageSrc = item.thumbnail_url
                    ? (item.thumbnail_url.startsWith('http') ? item.thumbnail_url : `${API_BASE_URL}${item.thumbnail_url}`)
                    : '/camera_placeholder.jpg';
                  return (
                    <List.Item
                      key={item.id}
                      style={{ cursor: 'pointer' }}
                      onClick={() => router.push(`/user/camera/${item.id}`)}
                    >
                      <List.Item.Meta
                        avatar={<Avatar icon={<CameraOutlined />} src={itemImageSrc} />}
                        title={<Link href={`/user/camera/${item.id}`}>{item.name}</Link>}
                        description={item.location?.name || 'Unknown location'}
                      />
                    </List.Item>
                  );
                }}
              />
            ) : (
              <Text>No other cameras to display.</Text>
            )}
            {error && !camera && (
              <Alert message="Could not load other cameras." type="warning" showIcon style={{ marginTop: '10px' }} />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
} 