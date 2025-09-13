"use client";

import { useAuth } from '@/context/authContext';
import axiosInstance from '@/utils/axiosInstance';
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  DeleteOutlined,
  PlusOutlined,
  RedoOutlined,
  ReloadOutlined,
  SettingOutlined,
  VideoCameraOutlined,
  CameraOutlined,
} from '@ant-design/icons';
import {
  Alert,
  Button,
  Form,
  Input,
  InputNumber,
  message,
  Modal,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Tooltip,
  Typography
} from 'antd';
import dynamic from 'next/dynamic';
import Image from 'next/image';
import { useCallback, useEffect, useState } from 'react';

const { Title } = Typography;
const { Option } = Select;
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL_BE || "http://localhost:8000";

// Dynamically import ROIEditor to avoid SSR issues with browser-specific code
const ROIEditor = dynamic(() => import('@/components/camera/ROIEditor'), {
  ssr: false
});

export default function ManageCameraPage() {
  const { user, isAuthenticated, loading: authLoading } = useAuth();
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [messageApi, contextHolder] = message.useMessage();

  // ROI editing state
  const [selectedCameraForRoi, setSelectedCameraForRoi] = useState(null);
  const [isRefreshingAllThumbnails, setIsRefreshingAllThumbnails] = useState(false);
  const [isRoiModalVisible, setIsRoiModalVisible] = useState(false);

  // Add Camera Modal state
  const [isAddCameraModalVisible, setIsAddCameraModalVisible] = useState(false);
  const [addCameraForm] = Form.useForm();
  const [isAddingCamera, setIsAddingCamera] = useState(false);

  // State for managing messages via useEffect
  const [notification, setNotification] = useState(null);
  // State for loading individual thumbnails
  const [loadingThumbnails, setLoadingThumbnails] = useState({});

  // Fetch all cameras for admin
  const fetchAdminCameras = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await axiosInstance.get('/api/admin/cameras');
      console.log("Camera: ")
      setCameras(response.data || []);
    } catch (err) {
      console.error("Error fetching admin cameras:", err);
      const errorMsg = err.response?.data?.detail || "Failed to load cameras.";
      setError(errorMsg);
      messageApi.error(errorMsg);
      setCameras([]);
    } finally {
      setLoading(false);
    }
  }, [messageApi]);

  // Initial data loading
  useEffect(() => {
    if (!authLoading && isAuthenticated && user?.role === 'admin') {
      fetchAdminCameras();
    } else if (!authLoading && (!isAuthenticated || user?.role !== 'admin')) {
      setError("Access Denied: You must be an administrator to view this page.");
      setCameras([]);
    }
  }, [authLoading, isAuthenticated, user, fetchAdminCameras]);

  // Update camera status handler
  const handleStatusChange = async (camera_id, newStatus) => {
    messageApi.loading({ content: 'Updating status...', key: `status-${camera_id}` });
    try {
      await axiosInstance.put(`/api/admin/cameras/${camera_id}/status`, { status: newStatus });
      messageApi.success({ content: 'Status updated successfully!', key: `status-${camera_id}`, duration: 2 });
      setCameras(prevCameras =>
        prevCameras.map(cam =>
          cam.id === camera_id ? { ...cam, status: newStatus, updated_at: new Date().toISOString() } : cam
        )
      );
    } catch (err) {
      console.error("Error updating camera status:", err);
      messageApi.error({ content: err.response?.data?.detail || 'Failed to update status.', key: `status-${camera_id}`, duration: 3 });
    }
  };

  // Open ROI configuration modal
  const openRoiModal = (camera) => {
    console.log("Opening ROI editor for camera:", camera?.id);
    setSelectedCameraForRoi(camera);
    setIsRoiModalVisible(true);
  };

  // Handle ROI save
  const handleSaveRoi = async (cameraId, normalizedPoints, dimensions) => {
    messageApi.loading({ content: 'Saving ROI...', key: `roi-${cameraId}` });
    try {
      const payload = {
        roi_points: normalizedPoints,
        roi_dimensions: dimensions
      };

      const response = await axiosInstance.put(`/api/admin/cameras/${cameraId}/roi`, payload);

      if (response.data && response.data.roi) {
        // Update the camera in our state with the new ROI data
        setCameras(prevCameras =>
          prevCameras.map(cam =>
            cam.id === cameraId ? {
              ...cam,
              roi: response.data.roi,
              updated_at: new Date().toISOString()
            } : cam
          )
        );

        messageApi.success({
          content: 'ROI updated successfully!',
          key: `roi-${cameraId}`,
          duration: 2
        });

        // Close the modal
        closeRoiModal();
      } else {
        throw new Error("Invalid response data");
      }
    } catch (err) {
      console.error("Error saving ROI:", err.message);
      messageApi.error({
        content: err.response?.data?.detail || 'Failed to save ROI.',
        key: `roi-${cameraId}`,
        duration: 3
      });
    }
  };

  // Close ROI modal
  const closeRoiModal = () => {
    setIsRoiModalVisible(false);
    // Small delay before clearing the camera data to avoid UI flicker
    setTimeout(() => {
      setSelectedCameraForRoi(null);
    }, 300);
  };

  // Refresh thumbnails for all cameras
  const handleRefreshAllThumbnails = async () => {
    if (isRefreshingAllThumbnails) {
      setNotification({ type: 'info', content: "Thumbnail refresh already in progress." });
      return;
    }

    setIsRefreshingAllThumbnails(true);
    setNotification({ type: 'loading', content: `Requesting thumbnail refresh for all cameras...`, key: 'adminThumbnailRefreshAll', duration: 0 });

    try {
      const response = await axiosInstance.post(`${API_BASE_URL}/api/cameras/refresh-all-thumbnails`);
      // The loading message with key 'adminThumbnailRefreshAll' will be destroyed by the next notification with the same key

      const data = response?.data || {};
      const { success_count, fail_count, total_cameras, message: response_message } = data;

      if (typeof total_cameras === 'number') {
        let summary = `Thumbnail refresh request processed.`;
        if (typeof success_count === 'number' && typeof fail_count === 'number') {
          summary = `Refreshed ${success_count} thumbnails`;
          if (fail_count > 0) summary += ` (${fail_count} failed).`; else summary += `.`;
        }

        if (fail_count > 0) {
          setNotification({ type: 'warning', content: summary, key: 'adminThumbnailRefreshAll' });
        } else if (success_count > 0 || (success_count === 0 && fail_count === 0 && total_cameras > 0)) {
          setNotification({ type: 'success', content: summary, key: 'adminThumbnailRefreshAll' });
        } else {
          setNotification({ type: 'info', content: response_message || summary || "Refresh process initiated, checking results...", key: 'adminThumbnailRefreshAll' });
        }
      } else {
        setNotification({ type: 'info', content: response_message || "Thumbnail refresh request sent. Re-fetching data...", key: 'adminThumbnailRefreshAll' });
      }

      // Wait a short delay for backend processing and file system updates, then refresh list
      await new Promise(resolve => setTimeout(resolve, 1500));
      await fetchAdminCameras();

    } catch (err) {
      console.error("Error requesting thumbnail refresh for all cameras (admin):", err);
      setNotification({ type: 'error', content: err.response?.data?.detail || "Failed to initiate thumbnail refresh process.", key: 'adminThumbnailRefreshAll' });
    }
    setIsRefreshingAllThumbnails(false);
  };

  // Open Add Camera Modal
  const openAddCameraModal = () => {
    setIsAddCameraModalVisible(true);
  };

  // Close Add Camera Modal
  const closeAddCameraModal = () => {
    setIsAddCameraModalVisible(false);
    addCameraForm.resetFields(); // Reset form when closing
  };

  // Handle Add Camera submission
  const handleAddCamera = async (values) => {
    setIsAddingCamera(true);
    messageApi.loading({ content: 'Adding new camera...', key: 'addCamera' });

    const payload = {
      name: values.name,
      stream_url: values.stream_url,
      status: values.status,
      description: values.description,
      location_data: { // Changed 'location' to 'location_data'
        name: values.location_name || 'Unknown Location', // Default if not provided
        latitude: values.location_latitude,
        longitude: values.location_longitude,
      },
    };

    try {
      await axiosInstance.post('/api/admin/cameras', payload);
      messageApi.success({ content: 'Camera added successfully!', key: 'addCamera', duration: 2 });
      closeAddCameraModal();
      fetchAdminCameras(); // Refresh the list
    } catch (err) {
      console.error("Error adding camera:", err);
      messageApi.error({ content: err.response?.data?.detail || 'Failed to add camera.', key: 'addCamera', duration: 3 });
    } finally {
      setIsAddingCamera(false);
    }
  };

  // Handler for refreshing a single camera's thumbnail
  const handleRefreshSingleThumbnail = async (cameraId) => {
    setLoadingThumbnails(prev => ({ ...prev, [cameraId]: true }));
    messageApi.loading({ content: `Refreshing thumbnail for camera ${cameraId}...`, key: `thumb-${cameraId}` });

    try {
      const response = await axiosInstance.get(`/api/admin/cameras/${cameraId}/capture-frame`);
      // The backend endpoint should return the updated camera data or at least a success message
      // and the new frame URL.
      messageApi.success({ content: `Thumbnail for camera ${cameraId} refreshed! New URL: ${response.data.frame_url}`, key: `thumb-${cameraId}`, duration: 3 });
      
      // Fetch all cameras again to get the updated thumbnail URL and updated_at timestamp
      // which is used for cache-busting the image.
      await fetchAdminCameras(); 

    } catch (err) {
      console.error(`Error refreshing thumbnail for camera ${cameraId}:`, err);
      messageApi.error({ content: err.response?.data?.detail || `Failed to refresh thumbnail for camera ${cameraId}.`, key: `thumb-${cameraId}`, duration: 3 });
    } finally {
      setLoadingThumbnails(prev => ({ ...prev, [cameraId]: false }));
    }
  };

  // Check if ROI is properly configured
  const isRoiConfigured = (roi) => {
    return (
      roi &&
      Array.isArray(roi.points) &&
      roi.points.length === 4 &&
      typeof roi.roi_width_meters === 'number' &&
      roi.roi_width_meters > 0 &&
      typeof roi.roi_height_meters === 'number' &&
      roi.roi_height_meters > 0
    );
  };

  // Table column definitions
  const columns = [
    {
      title: 'Thumbnail',
      dataIndex: 'thumbnail_url',
      key: 'thumbnail',
      width: 100,
      render: (url, record) => {
        // Explicitly check for empty string or null/undefined for url
        const baseUrl = (url && typeof url === 'string' && url.trim() !== '') ? url : null;
        const imageSrc = baseUrl ? (baseUrl.startsWith('http') ? baseUrl : `${API_BASE_URL}${baseUrl}`) : null;
        
        return (
          <div style={{ width: 80, height: 60, position: 'relative', overflow: 'hidden', backgroundColor: '#e0e0e0' }}>
            {imageSrc ? (
              <Image
                src={`${imageSrc}?t=${new Date(record.updated_at || Date.now()).getTime()}`} // Use updated_at for cache busting
                alt={record.name || 'Camera thumbnail'}
                fill
                style={{ objectFit: 'cover' }}
                onError={(e) => {
                  console.warn(`Admin thumbnail failed: ${imageSrc}`);
                  e.target.style.display = 'none';
                }}
                unoptimized={imageSrc && !imageSrc.startsWith('/')}
                sizes="80px"
              />
            ) : (
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#bfbfbf' }}>No image</span>
            )}
          </div>
        );
      }
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      sorter: (a, b) => a.name.localeCompare(b.name),
      ellipsis: true,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 150,
      filters: [
        { text: 'Active', value: 'Active' },
        { text: 'Inactive', value: 'Inactive' },
        { text: 'Maintenance', value: 'Maintenance' },
      ],
      onFilter: (value, record) => record.status === value,
      render: (status, record) => (
        <Select
          value={status}
          style={{ width: 130 }}
          onChange={(newStatus) => handleStatusChange(record.id, newStatus)}
          onClick={(e) => e.stopPropagation()}
        >
          <Option value="Active"><Tag color="green">Active</Tag></Option>
          <Option value="Inactive"><Tag color="red">Inactive</Tag></Option>
          <Option value="Maintenance"><Tag color="orange">Maintenance</Tag></Option>
        </Select>
      ),
    },
    {
      title: 'ROI',
      dataIndex: 'roi',
      key: 'roi',
      width: 80,
      align: 'center',
      render: (roi) => {
        const configured = isRoiConfigured(roi);
        return configured ? (
          <Tooltip title={`ROI Defined (${roi.roi_width_meters?.toFixed(1)}m x ${roi.roi_height_meters?.toFixed(1)}m)`}>
            <Tag icon={<CheckCircleOutlined />} color="success">Yes</Tag>
          </Tooltip>
        ) : (
          <Tooltip title="ROI Not Configured">
            <Tag icon={<CloseCircleOutlined />} color="default">No</Tag>
          </Tooltip>
        );
      },
      filters: [
        { text: 'Configured', value: true },
        { text: 'Not Configured', value: false },
      ],
      onFilter: (value, record) => isRoiConfigured(record.roi) === value,
    },
    {
      title: 'Stream URL',
      dataIndex: 'stream_url',
      key: 'stream_url',
      render: (url) => (
        <Tooltip title={url}>
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={e => e.stopPropagation()}
            style={{
              maxWidth: '200px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              display: 'inline-block'
            }}
          >
            {url}
          </a>
        </Tooltip>
      ),
      ellipsis: true,
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      fixed: 'right',
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="Configure ROI">
            <Button
              icon={<SettingOutlined />}
              type="primary"
              onClick={(e) => {
                e.stopPropagation();
                openRoiModal(record);
              }}
            />
          </Tooltip>
          <Tooltip title="Refresh Thumbnail">
            <Button
              icon={<CameraOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                handleRefreshSingleThumbnail(record.id);
              }}
              loading={loadingThumbnails[record.id]}
            />
          </Tooltip>
          <Tooltip title="Delete Camera (Not active)">
            <Button
              danger
              icon={<DeleteOutlined />}
              onClick={(e) => { e.stopPropagation(); }}
              disabled
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // useEffect to display notifications
  useEffect(() => {
    if (notification) {
      const { type, content, key, duration } = notification;
      switch (type) {
        case 'loading':
          messageApi.loading({ content, key, duration: duration || 0 });
          break;
        case 'success':
          if (key) messageApi.destroy(key);
          messageApi.success(content, duration);
          break;
        case 'warning':
          if (key) messageApi.destroy(key);
          messageApi.warning(content, duration);
          break;
        case 'error':
          if (key) messageApi.destroy(key);
          messageApi.error(content, duration);
          break;
        case 'info':
        default:
          if (key) messageApi.destroy(key);
          messageApi.info(content, duration);
          break;
      }
      setNotification(null); // Reset after displaying
    }
  }, [notification, messageApi]);

  if (authLoading) {
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 200px)' }}><Spin size="large" tip="Authenticating..." /></div>;
  }

  if (!isAuthenticated || user?.role !== 'admin') {
    return <div style={{ padding: '24px' }}><Alert message="Access Denied" description="You do not have permission to access this page." type="error" showIcon /></div>;
  }

  return (
    <div style={{ padding: '24px' }}>
      {contextHolder}
      <Title level={2} style={{ marginBottom: '24px' }}>
        <VideoCameraOutlined style={{ marginRight: '12px' }} /> Manage Cameras (UC13)
      </Title>

      <Space style={{ marginBottom: '16px' }}>
        <Button
          icon={<ReloadOutlined />}
          onClick={fetchAdminCameras}
          loading={loading}
        >
          Refresh List
        </Button>
        <Button
          icon={<RedoOutlined />}
          onClick={handleRefreshAllThumbnails}
          loading={isRefreshingAllThumbnails}
          disabled={loading}
        >
          Refresh All Thumbnails
        </Button>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={openAddCameraModal}
          disabled={loading || isRefreshingAllThumbnails} // Disable if other main loading is active
        >
          Add New Camera
        </Button>
      </Space>

      {error && !loading && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: '16px' }}
        />
      )}

      <Table
        columns={columns}
        dataSource={cameras}
        rowKey="id"
        loading={loading && cameras.length === 0}
        scroll={{ x: 1000 }}
        pagination={{ pageSize: 10 }}
        onRow={(record) => ({
          onClick: () => openRoiModal(record),
        })}
      />

      {/* ROI Editor Modal */}
      <Modal
        title={`Configure ROI for ${selectedCameraForRoi?.name || ''}`}
        open={isRoiModalVisible && !!selectedCameraForRoi}
        footer={null}
        onCancel={closeRoiModal}
        width={800}
        destroyOnClose={true}
      >
        {selectedCameraForRoi && (
          <ROIEditor
            camera={selectedCameraForRoi}
            isVisible={isRoiModalVisible && !!selectedCameraForRoi}
            onSave={(id, points, dims) => handleSaveRoi(id, points, dims)}
            onCancel={closeRoiModal}
          />
        )}
      </Modal>

      {/* Add Camera Modal */}
      <Modal
        title="Add New Camera"
        open={isAddCameraModalVisible}
        onCancel={closeAddCameraModal}
        confirmLoading={isAddingCamera}
        onOk={() => {
          addCameraForm
            .validateFields()
            .then(values => {
              // Form.resetFields() is called on modal close
              handleAddCamera(values);
            })
            .catch(info => {
              console.log('Validate Failed:', info);
            });
        }}
        okText="Add Camera"
      >
        <Form
          form={addCameraForm}
          layout="vertical"
          name="add_camera_form"
          initialValues={{ status: 'Active', location_name: 'Unknown Location' }}
        >
          <Form.Item
            name="name"
            label="Camera Name"
            rules={[{ required: true, message: 'Please input the camera name!' }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="stream_url"
            label="Stream URL"
            rules={[
              { required: true, message: 'Please input the stream URL!' },
              { type: 'url', message: 'Please enter a valid URL!' }
            ]}
          >
            <Input placeholder="https://example.com/stream" />
          </Form.Item>
          <Form.Item
            name="status"
            label="Status"
            rules={[{ required: true, message: 'Please select a status!' }]}
          >
            <Select>
              <Option value="Active">Active</Option>
              <Option value="Inactive">Inactive</Option>
              <Option value="Maintenance">Maintenance</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="description"
            label="Description"
          >
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item
            name="location_name"
            label="Location Name (Optional)"
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="location_latitude"
            label="Latitude"
            rules={[{ required: true, message: 'Please input the latitude!' }]}
          >
            <InputNumber style={{ width: '100%' }} placeholder="-90.0 to 90.0" />
          </Form.Item>
          <Form.Item
            name="location_longitude"
            label="Longitude"
            rules={[{ required: true, message: 'Please input the longitude!' }]}
          >
            <InputNumber style={{ width: '100%' }} placeholder="-180.0 to 180.0" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}