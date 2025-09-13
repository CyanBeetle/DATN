"use client";

import { useAuth } from '@/context/authContext';
import axiosInstance from '@/utils/axiosInstance';
import {
  BellOutlined,
  CarOutlined,
  CheckCircleOutlined,
  CheckOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  EnvironmentOutlined,
  EyeOutlined,
  HomeOutlined,
  SyncOutlined,
  UserOutlined
} from '@ant-design/icons';
import {
  Alert,
  Badge,
  Button,
  Card,
  Divider,
  Drawer,
  Empty,
  Form, Input,
  List,
  message,
  Modal,
  Popover,
  Select, Space,
  Spin,
  Tabs,
  Tag,
  Tooltip,
  Typography
} from 'antd';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;
const { confirm } = Modal;

const AdminReportsPage = () => {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [messageApi, contextHolder] = message.useMessage();

  // States
  const [loading, setLoading] = useState(false);
  const [allReports, setAllReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [detailsModalVisible, setDetailsModalVisible] = useState(false);
  const [responseDrawerVisible, setResponseDrawerVisible] = useState(false);
  const [responseForm] = Form.useForm();
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [notifications, setNotifications] = useState([]);

  const [activeTabKey, setActiveTabKey] = useState('New');
  const [filters, setFilters] = useState({ status: 'New' });

  const [imagePreviewVisible, setImagePreviewVisible] = useState(false);
  const [previewImage, setPreviewImage] = useState('');

  const fetchReports = useCallback(async (currentFilters) => {
    setLoading(true);
    try {
      const params = { ...currentFilters };
      if (params.status === 'all-reports') delete params.status;

      const response = await axiosInstance.get('/api/admin/reports', { params });

      const mappedReports = response.data.map(report => ({
        ...report,
        id: report._id,
        type: report.report_type,
        responseMessage: report.resolution_notes,
      }));
      setAllReports(mappedReports);

    } catch (error) {
      console.error('Error fetching reports:', error);
      messageApi.error(error.response?.data?.detail || 'Failed to load reports');
    } finally {
      setLoading(false);
    }
  }, [messageApi]);

  const fetchNotifications = useCallback(async () => {
    if (!isAuthenticated || user?.role !== 'admin') return;
    try {
      const response = await axiosInstance.get('/api/admin/notifications');
      setNotifications(response.data.map(n => ({ ...n, id: n._id || n.id })));
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  }, [isAuthenticated, user]);

  useEffect(() => {
    if (isAuthenticated && user) {
      if (user.role?.toLowerCase() !== 'admin') {
        messageApi.error('You do not have permission to access this page');
        router.push('/user/map');
        return;
      }
      fetchReports(filters);
      fetchNotifications();
    } else if (isAuthenticated === false) {
      router.push('/auth/login');
    }
  }, [isAuthenticated, user, router, messageApi, fetchNotifications, filters, fetchReports]);

  const viewReportDetails = async (reportId) => {
    try {
      setLoading(true);
      const response = await axiosInstance.get(`/api/reports/${reportId}`);
      const report = response.data;
      setSelectedReport({
        ...report,
        type: report.report_type,
        // submittedAt: report.created_at,
        // processedAt: report.updated_at,
        responseMessage: report.resolution_notes,
      });
      setDetailsModalVisible(true);
    } catch (error) {
      console.error('Error fetching report details:', error);
      messageApi.error(error.response?.data?.detail || 'Failed to load report details');
    } finally {
      setLoading(false);
    }
  };

  const openResponseDrawer = (reportToRespond) => {
    setSelectedReport(reportToRespond);
    responseForm.setFieldsValue({
      status: reportToRespond.status || 'Processing',
      admin_reply: reportToRespond.admin_reply || reportToRespond.responseMessage || ''
    });
    setResponseDrawerVisible(true);
  };

  const handleResponseSubmit = async (values) => {
    if (!selectedReport) return;
    setConfirmLoading(true);
    try {
      const payload = {
        status: values.status,
        admin_reply: values.admin_reply || null
      };
      await axiosInstance.patch(`/api/admin/reports/${selectedReport._id}`, payload);

      messageApi.success('Report updated successfully');
      setResponseDrawerVisible(false);
      fetchReports(filters);
      fetchNotifications();

      if (detailsModalVisible && selectedReport.id === (await axiosInstance.get(`/api/reports/${selectedReport.id}`)).data.id) {
        viewReportDetails(selectedReport.id);
      } else if (detailsModalVisible) {
        setDetailsModalVisible(false);
      }

    } catch (error) {
      console.error('Error updating report:', error);
      messageApi.error(error.response?.data?.detail || 'Failed to update report');
    } finally {
      setConfirmLoading(false);
    }
  };

  const markNotificationRead = async (notificationId) => {
    try {
      await axiosInstance.patch(`/api/admin/notifications/${notificationId}/read`);
      setNotifications(prev => prev.map(n =>
        n.id === notificationId ? { ...n, is_read: true } : n
      ));
    } catch (error) {
      console.error('Error marking notification as read:', error);
      messageApi.error('Failed to mark notification as read');
    }
  };

  const handleFilterChange = (filterKey, value) => {
    const newFilters = { ...filters };
    if (value) {
      newFilters[filterKey] = value;
    } else {
      delete newFilters[filterKey];
    }
    if (filterKey === 'status') {
      setActiveTabKey(value || 'all-reports');
    }
    setFilters(newFilters);
  };

  const handleTabChange = (key) => {
    setActiveTabKey(key);
    const newFilters = { ...filters };
    if (key === 'all-reports') {
      delete newFilters.status;
    } else {
      newFilters.status = key;
    }
    setFilters(newFilters);
  };

  const formattedDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString('en-GB', { hour12: true });
  };

  const getStatusTag = (status) => {
    switch (status) {
      case 'New':
        return <Tag icon={<ClockCircleOutlined />} color="gold">New</Tag>;
      case 'Processing':
        return <Tag icon={<SyncOutlined spin />} color="processing">Processing</Tag>;
      case 'Verified':
        return <Tag icon={<CheckCircleOutlined />} color="cyan">Verified</Tag>;
      case 'Resolved':
        return <Tag icon={<CheckCircleOutlined />} color="success">Resolved</Tag>;
      case 'Rejected':
        return <Tag icon={<CloseCircleOutlined />} color="error">Rejected</Tag>;
      default:
        return <Tag color="default">{status || 'Unknown'}</Tag>;
    }
  };

  const getTypeTag = (type) => {
    if (type === 'incident') {
      return <Tag icon={<CarOutlined />} color="magenta">Traffic Incident</Tag>;
    } else if (type === 'infrastructure') {
      return <Tag icon={<HomeOutlined />} color="orange">Infrastructure Issue</Tag>;
    }
    return <Tag color="blue">{type || 'General'}</Tag>;
  };

  const notificationsContent = (
    <div style={{ width: 350, maxHeight: 400, overflowY: 'auto' }}>
      {notifications.length === 0 ? (
        <Empty description="No new notifications" image={Empty.PRESENTED_IMAGE_SIMPLE} />
      ) : (
        <List
          itemLayout="horizontal"
          dataSource={notifications.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))}
          renderItem={(notification) => (
            <List.Item
              style={{
                padding: '8px 12px',
                cursor: 'pointer',
                backgroundColor: notification.is_read ? 'transparent' : '#e6f7ff'
              }}
              onClick={() => {
                viewReportDetails(notification.report_id);
                if (!notification.is_read) {
                  markNotificationRead(notification.id);
                }
              }}
            >
              <List.Item.Meta
                title={<Text strong>{notification.message || `New Report: ${notification.report_title}`}</Text>}
                description={<>
                  {getTypeTag(notification.report_type)}
                  <Text type="secondary" style={{ fontSize: '12px' }}>{formattedDate(notification.created_at)}</Text>
                </>}
              />
              {!notification.is_read && <Badge status="processing" />}
            </List.Item>
          )}
        />
      )}
    </div>
  );

  const handleImagePreview = (url) => {
    setPreviewImage(url);
    setImagePreviewVisible(true);
  };

  if (!isAuthenticated && isAuthenticated !== undefined) {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="Authentication Required"
          description="Please log in with an admin account to access this page."
          type="error"
          showIcon
        />
      </div>
    );
  }

  if (isAuthenticated && user && user.role?.toLowerCase() !== 'admin') {
    return (
      <div style={{ padding: '24px' }}>
        <Alert
          message="Access Denied"
          description="You do not have permission to access this page. This page is restricted to admin users only."
          type="error"
          showIcon
        />
      </div>
    );
  }

  const reportsToShow = allReports;
  const pendingCount = allReports.filter(r => r.status === 'New').length;

  const tabItems = [
    {
      key: 'New',
      label: <Badge count={pendingCount} size="small" offset={[10, -2]}>Pending ({pendingCount})</Badge>,
    },
    {
      key: 'Processing',
      label: 'Processing',
    },
    {
      key: 'Verified',
      label: 'Verified',
    },
    {
      key: 'Resolved',
      label: 'Resolved',
    },
    {
      key: 'Rejected',
      label: 'Rejected',
    },
    {
      key: 'all-reports',
      label: 'All Reports',
    }
  ];

  return (
    <div className="p-4 md:p-6 lg:p-8">
      {contextHolder}

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <Title level={2} style={{ margin: 0 }}>Manage Reports</Title>

        <Popover
          content={notificationsContent}
          title={<Title level={4} style={{ margin: 0 }}>Notifications</Title>}
          trigger="click"
          placement="bottomRight"
          overlayStyle={{ width: 350 }}
        >
          <Badge count={notifications.filter(n => !n.is_read).length || 0}>
            <Button icon={<BellOutlined />} size="large" shape="circle" />
          </Badge>
        </Popover>
      </div>

      <Tabs activeKey={activeTabKey} onChange={handleTabChange} items={tabItems} />

      <Card className="mt-4">
        <div className="mb-4 flex flex-wrap items-center gap-4">
          <Text strong>Filters:</Text>
          <Select
            allowClear
            placeholder="Filter by type"
            style={{ width: 200 }}
            value={filters.report_type}
            onChange={(value) => handleFilterChange('report_type', value)}
          >
            <Select.Option value="incident">Traffic Incident</Select.Option>
            <Select.Option value="infrastructure">Infrastructure Issue</Select.Option>
          </Select>
          <Input.Search
            allowClear
            placeholder="Search by title, description..."
            style={{ width: 250 }}
            onSearch={(value) => handleFilterChange('search', value)}
            onChange={(e) => { if (!e.target.value) handleFilterChange('search', null); }}
            enterButton
          />
          <Button icon={<SyncOutlined />} onClick={() => fetchReports(filters)}>Refresh</Button>
        </div>

        <Spin spinning={loading && reportsToShow.length === 0}>
          {reportsToShow.length === 0 && !loading ? (
            <Empty description={`No reports found for status: ${activeTabKey}`} />
          ) : (
            <List
              itemLayout="vertical"
              dataSource={reportsToShow}
              renderItem={(item) => (
                <List.Item
                  key={item.id}
                  actions={[
                    <Tooltip title="View Details" key={`view-${item.id}`}>
                      <Button
                        icon={<EyeOutlined />}
                        onClick={() => viewReportDetails(item.id)}
                        type="text"
                      />
                    </Tooltip>,
                    (item.status === 'New' || item.status === 'Processing') && (
                      <Tooltip title="Update Status & Respond" key={`respond-${item.id}`}>
                        <Button
                          icon={<CheckOutlined />}
                          onClick={() => openResponseDrawer(item)}
                          type="primary"
                          ghost
                        />
                      </Tooltip>
                    ),
                  ].filter(Boolean)}
                  extra={
                    item.image_url && (
                      <div style={{ width: 150, height: 100, position: 'relative', borderRadius: '4px', overflow: 'hidden', cursor: 'pointer' }} onClick={() => handleImagePreview(`${axiosInstance.defaults.baseURL}${item.image_url}`)}>
                        <Image
                          alt={item.title || 'Report thumbnail'}
                          src={`${axiosInstance.defaults.baseURL}${item.image_url}`}
                          fill
                          style={{ objectFit: 'cover' }}
                        />
                      </div>
                    )
                  }
                >
                  <List.Item.Meta
                    title={<a onClick={() => viewReportDetails(item.id)} style={{ fontSize: '1.1em' }}>{item.title}</a>}
                    description={<Space size="small">{getTypeTag(item.type)}{getStatusTag(item.status)}</Space>}
                  />
                  <Paragraph ellipsis={{ rows: 2, expandable: true, symbol: 'more' }}>
                    {item.description}
                  </Paragraph>
                  <Space size="small" wrap split={<Divider type="vertical" />}>
                    <Text type="secondary" style={{ fontSize: '12px' }}><UserOutlined /> {item.created_by_username}</Text>
                    <Text type="secondary" style={{ fontSize: '12px' }}><ClockCircleOutlined /> {formattedDate(item.submitted_at)}</Text>
                    {item.location && <Text type="secondary" style={{ fontSize: '12px' }}><EnvironmentOutlined /> {item.location}</Text>}
                  </Space>
                </List.Item>
              )}
            />
          )}
        </Spin>
      </Card>

      <Modal
        title={<Title level={4} style={{ margin: 0 }}>{selectedReport?.title || 'Report Details'}</Title>}
        open={detailsModalVisible}
        width={700}
        onCancel={() => setDetailsModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailsModalVisible(false)}>
            Close
          </Button>,
          (selectedReport?.status === 'New' || selectedReport?.status === 'Processing') && (
            <Button
              key="process"
              type="primary"
              icon={<CheckOutlined />}
              onClick={() => {
                setDetailsModalVisible(false);
                openResponseDrawer(selectedReport);
              }}
            >
              Update Status / Respond
            </Button>
          )
        ].filter(Boolean)}
      >
        {selectedReport && (
          <Spin spinning={loading && selectedReport.id === (allReports.find(r => r.id === selectedReport.id))?.id}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>{getTypeTag(selectedReport.type)}{getStatusTag(selectedReport.status)}</Space>
              <Paragraph><Text strong>Description:</Text> {selectedReport.description}</Paragraph>
              {selectedReport.location && <Paragraph><EnvironmentOutlined /> <Text strong>Location:</Text> {selectedReport.location}</Paragraph>}
              <Paragraph><UserOutlined /> <Text strong>Reported by:</Text> {selectedReport.created_by_username}</Paragraph>
              <Paragraph><ClockCircleOutlined /> <Text strong>Submitted:</Text> {formattedDate(selectedReport.submitted_at)}</Paragraph>

              {selectedReport.image_url && (
                <div>
                  <Text strong>Attachment:</Text>
                  <div style={{ marginTop: '8px', position: 'relative', width: '100%', minHeight: '200px', maxHeight: '400px' }}>
                    <Image
                      src={`${axiosInstance.defaults.baseURL}${selectedReport.image_url}`}
                      alt={selectedReport.title || 'Report attachment'}
                      fill
                      style={{ objectFit: 'contain', borderRadius: '4px', cursor: 'pointer' }}
                      onClick={() => handleImagePreview(`${axiosInstance.defaults.baseURL}${selectedReport.image_url}`)}
                    />
                  </div>
                </div>
              )}

              {(selectedReport.status === 'Verified' || selectedReport.status === 'Resolved' || selectedReport.status === 'Rejected') && selectedReport.processed_at && (
                <Paragraph><CheckCircleOutlined /> <Text strong>Last Updated:</Text> {formattedDate(selectedReport.processed_at)}</Paragraph>
              )}

              {selectedReport.responseMessage && (
                <Card size="small" title="Admin Resolution/Response" style={{ marginTop: 10 }}>
                  <Paragraph>{selectedReport.responseMessage}</Paragraph>
                </Card>
              )}
            </Space>
          </Spin>
        )}
      </Modal>

      <Drawer
        title={<Title level={4} style={{ margin: 0 }}>Update Report Status & Respond</Title>}
        placement="right"
        width={500}
        open={responseDrawerVisible}
        onClose={() => setResponseDrawerVisible(false)}
        footer={
          <div style={{ textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setResponseDrawerVisible(false)}>Cancel</Button>
              <Button
                type="primary"
                onClick={() => responseForm.submit()}
                loading={confirmLoading}
                icon={<CheckOutlined />}
              >
                Confirm Update
              </Button>
            </Space>
          </div>
        }
      >
        {selectedReport && (
          <Form
            form={responseForm}
            layout="vertical"
            onFinish={handleResponseSubmit}
            initialValues={{
              status: selectedReport.status || 'Processing',
              admin_reply: selectedReport.admin_reply || selectedReport.responseMessage || ''
            }}
          >
            <Title level={5}>{selectedReport.title}</Title>
            <Paragraph type="secondary" ellipsis={{ rows: 3, expandable: true, symbol: 'more' }}>{selectedReport.description}</Paragraph>
            <Divider />
            <Form.Item
              name="status"
              label="New Status"
              rules={[{ required: true, message: 'Please select a new status' }]}
            >
              <Select placeholder="Select new status">
                <Select.Option value="New">New</Select.Option>
                <Select.Option value="Processing">Processing</Select.Option>
                <Select.Option value="Verified">Verified</Select.Option>
                <Select.Option value="Resolved">Resolved</Select.Option>
                <Select.Option value="Rejected">Rejected</Select.Option>
              </Select>
            </Form.Item>

            <Form.Item
              name="admin_reply"
              label="Resolution Notes / Response to User (Optional)"
            >
              <TextArea
                rows={5}
                placeholder="Enter any notes about the resolution or a message for the user..."
              />
            </Form.Item>
          </Form>
        )}
      </Drawer>

      <Modal
        open={imagePreviewVisible}
        title="Image Preview"
        footer={null}
        onCancel={() => setImagePreviewVisible(false)}
        width="80vw"
        style={{ maxWidth: '800px' }}
      >
        {previewImage && (
          <div style={{ position: 'relative', width: '100%', height: '70vh' }}>
            <Image alt="Report attachment preview" src={previewImage} fill style={{ objectFit: 'contain' }} />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default AdminReportsPage;