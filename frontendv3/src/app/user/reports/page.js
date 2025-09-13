"use client";

import { useAuth } from '@/context/authContext';
import axiosInstance from '@/utils/axiosInstance';
import {
  CameraOutlined,
  CarOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  EnvironmentOutlined,
  HomeOutlined,
  PlusOutlined,
  UserOutlined
} from '@ant-design/icons';
import {
  Avatar,
  Badge,
  Button,
  Card,
  Divider, Empty,
  Form, Input,
  List,
  message,
  Modal,
  Select, Spin,
  Tabs,
  Tag,
  Typography,
  Upload
} from 'antd';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Option } = Select;

const ReportsPage = () => {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [form] = Form.useForm();
  const [messageApi, contextHolder] = message.useMessage();

  // States
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [userReports, setUserReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [detailsModalVisible, setDetailsModalVisible] = useState(false);
  const [fileList, setFileList] = useState([]);

  // Fetch user reports - wrapped in useCallback
  const fetchUserReports = useCallback(async () => {
    setLoading(true);
    try {
      const response = await axiosInstance.get('/api/reports/my-reports');
      const mappedReports = response.data.map(report => ({
        id: report._id,
        title: report.title,
        type: report.report_type,
        description: report.description,
        location: report.location,
        status: report.status,
        created_at: report.submitted_at,
        processed_at: report.processed_at,
        created_by_username: report.created_by_username,
        response: report.resolution_notes,
        image_url: report.image_url
      }));
      setUserReports(mappedReports);
    } catch (error) {
      console.error('Error fetching user reports:', error);
      messageApi.error(error.response?.data?.detail || 'Failed to load your reports');
    } finally {
      setLoading(false);
      console.log("User reports: ", userReports);
    }
  }, [messageApi]);

  // Component mount effect
  useEffect(() => {
    if (isAuthenticated) {
      fetchUserReports();
    } else if (isAuthenticated === false) {
      router.push('/auth/login');
    }
  }, [isAuthenticated, router, fetchUserReports]);

  // Handle form submission
  const onFinish = async (values) => {
    setSubmitting(true);

    const formData = new FormData();
    formData.append("report_title", values.title);
    formData.append("report_description", values.description);
    formData.append("report_type", values.type); // ví dụ: "urban" hoặc "incident"

    if (values.locationId) {
      formData.append("location_id", values.locationId);
    }
    if (values.locationName) {
      formData.append("location_name", values.locationName);
    }
    if (values.latitude) {
      formData.append("location_latitude", values.latitude);
    }
    if (values.longitude) {
      formData.append("location_longitude", values.longitude);
    }

    if (fileList.length > 0 && fileList[0].originFileObj) {
      formData.append("image", fileList[0].originFileObj);
    }

    try {
      const response = await axiosInstance.post('/api/reports', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log(response.data)
      messageApi.success('Report submitted successfully!');
      form.resetFields();
      setFileList([]);

      fetchUserReports();

    } catch (error) {
      console.error('Error submitting report:', error);
      messageApi.error(error.response?.data?.detail || 'Failed to submit report. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleUploadChange = ({ fileList: newFileList }) => {
    setFileList(newFileList);
  };

  const handleUploadRemove = () => {
    setFileList([]);
  };

  // View report details
  const viewReportDetails = async (reportId) => {
    try {
      setLoading(true);
      const response = await axiosInstance.get(`/api/reports/${reportId}`);
      const report = response.data;

      setSelectedReport({
        id: report.id,
        title: report.title,
        type: report.type,
        description: report.description,
        location: report.location,
        status: report.status,
        created_at: report.created_at,
        processed_at: report.processed_at,
        created_by_username: report.created_by_username,
        response: report.response,
        image_url: report.image_url
      });
      setDetailsModalVisible(true);
    } catch (error) {
      console.error('Error fetching report details:', error);
      messageApi.error(error.response?.data?.detail || 'Failed to load report details');
    } finally {
      setLoading(false);
    }
  };

  // Format date for display
  const formattedDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString('en-GB', { hour12: true });
  };

  // Get status tag display
  const getStatusTag = (status) => {
    switch (status) {
      case 'Submitted':
        return <Tag icon={<ClockCircleOutlined />} color="processing">Submitted</Tag>;
      case 'Processing':
        return <Tag icon={<Spin size="small" style={{ marginRight: '5px' }} />} color="blue">Processing</Tag>;
      case 'Verified':
        return <Tag icon={<CheckCircleOutlined />} color="cyan">Verified</Tag>;
      case 'Resolved':
        return <Tag icon={<CheckCircleOutlined />} color="success">Resolved</Tag>;
      case 'Rejected':
        return <Tag icon={<CloseCircleOutlined />} color="error">Rejected</Tag>;
      default:
        return <Tag color="default">{status}</Tag>;
    }
  };

  // Get report type tag display
  const getTypeTag = (type) => {
    if (type === 'incident') {
      return <Tag icon={<CarOutlined />} color="error">Traffic Incident</Tag>;
    } else if (type === 'infrastructure') {
      return <Tag icon={<HomeOutlined />} color="warning">Infrastructure Issue</Tag>;
    }
    return <Tag color="default">{type}</Tag>;
  };

  // Handle location selection (in a real implementation, this could integrate with a map)
  const handleLocationSelect = () => {
    messageApi.info('Location selection would open a map interface');
  };

  // Not authenticated view
  if (!isAuthenticated && isAuthenticated !== undefined) {
    return (
      <div style={{ padding: '24px' }}>
        <Card>
          <Title level={4}>Authentication Required</Title>
          <Paragraph>
            You need to log in to submit reports or view your report history.
          </Paragraph>
          <Button type="primary" onClick={() => router.push('/auth/login')}>
            Log In
          </Button>
        </Card>
      </div>
    );
  }

  const tabItems = [
    {
      key: 'new',
      label: 'Submit New Report',
      children: (
        <Card>
          <Form
            form={form}
            layout="vertical"
            onFinish={onFinish}
            initialValues={{ type: 'incident' }}
          >
            <Form.Item
              name="type"
              label="Report Type"
              rules={[{ required: true, message: 'Please select a report type' }]}
            >
              <Select>
                <Option value="incident">
                  <CarOutlined /> Traffic Incident (Accident, Congestion, Obstacle)
                </Option>
                <Option value="infrastructure">
                  <HomeOutlined /> Infrastructure Issue (Road, Signals, Signs)
                </Option>
              </Select>
            </Form.Item>

            <Form.Item
              name="title"
              label="Title"
              rules={[{ required: true, message: 'Please enter a title' }]}
            >
              <Input placeholder="Brief description of the incident or issue" />
            </Form.Item>

            <Form.Item
              name="description"
              label="Detailed Description"
              rules={[{ required: true, message: 'Please enter a detailed description' }]}
            >
              <TextArea
                rows={4}
                placeholder="Provide details about what happened, severity, and any other relevant information"
              />
            </Form.Item>

            <Form.Item
              name="location"
              label="Location"
              rules={[{ required: true, message: 'Please specify the location' }]}
            >
              <Input
                prefix={<EnvironmentOutlined />}
                placeholder="Enter address or location description"
                suffix={
                  <Button type="text" size="small" onClick={handleLocationSelect}>
                    Select on Map
                  </Button>
                }
              />
            </Form.Item>

            <Form.Item
              name="image"
              label="Upload Image (Optional)"
              extra="Upload a photo of the incident or issue to help officials assess the situation. Max 5MB (JPEG, PNG, GIF)."
            >
              <Upload
                fileList={fileList}
                maxCount={1}
                listType="picture-card"
                beforeUpload={() => false}
                onChange={handleUploadChange}
                onRemove={handleUploadRemove}
              >
                {fileList.length >= 1 ? null : (
                  <div>
                    <CameraOutlined />
                    <div style={{ marginTop: 8 }}>Upload</div>
                  </div>
                )}
              </Upload>
            </Form.Item>

            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={submitting}
                icon={<PlusOutlined />}
              >
                Submit Report
              </Button>
            </Form.Item>
          </Form>
        </Card>
      )
    },
    {
      key: 'reports',
      label: (
        <Badge count={userReports.filter(r => r.status === 'pending').length || 0} size="small">
          Your Reports
        </Badge>
      ),
      children: (
        <Card>
          <Spin spinning={loading}>
            {userReports.length === 0 ? (
              <Empty description="You haven't submitted any reports yet" />
            ) : (
              <List
                itemLayout="horizontal"
                dataSource={userReports}
                renderItem={(item) => 
                {
                  console.log('Rendering item:', item.id, item.title);
                  return (
                  <List.Item
                    actions={[
                      <Button
                        key="view"
                        type="link"
                        onClick={() => {
        console.log('Button clicked! item.id INSIDE onClick:', item.id); // LOG Ở ĐÂY
        viewReportDetails(item.id);
      }}
                      >
                        View Details
                      </Button>
                    ]}
                  >
                    <List.Item.Meta
                      avatar={item.image_url ?
                        <Avatar
                          src={`${axiosInstance.defaults.baseURL}${item.image_url}`}
                          alt={item.title || 'Report image'}
                        />
                        : null}
                      title={
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span>{item.title}</span>
                          {getStatusTag(item.status)}
                          {getTypeTag(item.type)}
                        </div>
                      }
                      description={
                        <div>
                          <div>{item.description.length > 100 ? item.description.substring(0, 100) + '...' : item.description}</div>
                          <div style={{ marginTop: '8px', fontSize: '12px', color: '#888' }}>
                            <EnvironmentOutlined /> {item.location} • <ClockCircleOutlined /> Submitted: {formattedDate(item.created_at)}
                          </div>
                        </div>
                      }
                    />
                  </List.Item>
                )
                }
                  }
              />
            )}
          </Spin>
        </Card>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {contextHolder}

      <Title level={2}>Traffic Reports & Feedback</Title>
      <Paragraph>
        Report traffic incidents, congestion, or infrastructure issues. Your reports help us improve traffic
        management and address problems quickly.
      </Paragraph>

      <Tabs defaultActiveKey="new" items={tabItems} />

      {/* Report Details Modal */}
      <Modal
        title={selectedReport?.title || 'Report Details'}
        open={detailsModalVisible}
        footer={[
          <Button key="close" onClick={() => setDetailsModalVisible(false)}>
            Close
          </Button>
        ]}
        onCancel={() => setDetailsModalVisible(false)}
        width={700}
      >
        {selectedReport && (
          <div>
            <div style={{ marginBottom: '16px', display: 'flex', gap: '8px' }}>
              {getTypeTag(selectedReport.type)}
              {getStatusTag(selectedReport.status)}
            </div>

            <Paragraph style={{ marginBottom: '16px' }}>
              <strong>Description:</strong> {selectedReport.description}
            </Paragraph>

            {selectedReport.location && (
              <Paragraph style={{ marginBottom: '16px' }}>
                <EnvironmentOutlined /> <strong>Location:</strong> {selectedReport.location}
              </Paragraph>
            )}

            <Paragraph style={{ marginBottom: '16px' }}>
              <UserOutlined /> <strong>Reported by:</strong> {selectedReport.created_by_username}
            </Paragraph>

            <Paragraph style={{ marginBottom: '16px' }}>
              <ClockCircleOutlined /> <strong>Submitted:</strong> {formattedDate(selectedReport.created_at)}
            </Paragraph>

            {selectedReport.status === 'Verified' && (
              <>
                <Divider />
                <Paragraph style={{ marginBottom: '16px' }}>
                  <CheckCircleOutlined /> <strong>Processed on:</strong> {formattedDate(selectedReport.processed_at)}
                </Paragraph>

                {selectedReport.response && (
                  <div style={{ background: '#f9f9f9', padding: '16px', borderRadius: '4px' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Official Response:</div>
                    <Paragraph>{selectedReport.response}</Paragraph>
                  </div>
                )}

                {!selectedReport.response && (
                  <div style={{ background: '#f9f9f9', padding: '16px', borderRadius: '4px' }}>
                    <Paragraph>Your report has been processed. Thank you for your contribution.</Paragraph>
                  </div>
                )}
              </>
            )}

            {selectedReport.image_url && (
              <div style={{ margin: '16px 0' }}>
                <Text strong>Attached Image:</Text>
                <div style={{ marginTop: '8px', position: 'relative', maxWidth: '100%', height: 'auto', maxHeight: '300px' }}>
                  <Image
                    src={`${axiosInstance.defaults.baseURL}${selectedReport.image_url}`}
                    alt={selectedReport.title || 'Report attachment'}
                    width={500}
                    height={300}
                    style={{ objectFit: 'contain', border: '1px solid #eee', borderRadius: '4px' }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ReportsPage;