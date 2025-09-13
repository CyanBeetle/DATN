"use client";

import { useState, useEffect, useCallback } from 'react';
import { 
  Card, 
  Typography, 
  Button, 
  Table, 
  Space, 
  Tag, 
  message, 
  Modal, 
  Spin,
  Alert,
  Descriptions,
  List,
  Tooltip
} from 'antd';
import { 
  EyeOutlined, 
  InfoCircleOutlined,
  CodeSandboxOutlined, // Using this for model set or generic model icon
  PaperClipOutlined, // For scaler
  SlidersOutlined, // For parameters
  TableOutlined, // For input/output shape
} from '@ant-design/icons';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/authContext';
import axiosInstance from '@/utils/axiosInstance';

const { Title, Text, Paragraph } = Typography;

const PredictionModelsPage = () => {  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [messageApi, contextHolder] = message.useMessage();
  
  // State variables
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);  
  const [selectedModel, setSelectedModel] = useState(null); // For details
  const [detailsModalVisible, setDetailsModalVisible] = useState(false);
  const [error, setError] = useState(null);
  
  // Fetch models from new API endpoint
  const fetchModels = useCallback(async () => {
    setLoading(true);
    try {
      const response = await axiosInstance.get('/api/admin/ml/models-overview');
      // The response is a list of models. Each model is an object.
      // Add a unique key for table rows. model.id from the backend (e.g., "Set1/model.keras") should be unique.
      const processedModels = response.data.map(model => ({
        ...model,
        key: model.id, // Rely on the backend-generated unique ID
      }));
      setModels(processedModels);
      setError(null);
    } catch (error) {
      console.error('Error fetching models:', error);
      messageApi.error(error.response?.data?.detail || 'Failed to load prediction models from new endpoint');
      setError(error);
    } finally {
      setLoading(false);
    }
  }, [messageApi]);

  // Load models on component mount
  useEffect(() => {
    if (isAuthenticated && user) {
      if (user.role.toLowerCase() !== 'admin') {
        messageApi.error('You do not have permission to access this page');
        router.push('/user/map');
        return;
      }
      fetchModels();
    } else if (isAuthenticated === false) {
      router.push('/auth/login');
    }
  }, [isAuthenticated, user, router, messageApi, fetchModels]);
  
  // Format date for display (if any date fields were present)
  // const formatDate = (dateString) => {
  //   if (!dateString) return 'N/A';
  //   const date = new Date(dateString);
  //   return date.toLocaleString();
  // };

  const displayParameters = (params) => {
    if (!params || typeof params !== 'object' || Object.keys(params).length === 0) {
      return <Text type="secondary">N/A</Text>;
    }
    // Handle simple string error messages for parameters
    if (typeof params === 'string') {
      return <Text type="danger">{params}</Text>;
    }
    if (params.error) {
      return <Text type="danger">{params.error}</Text>;
    }
    return (
      <List
        size="small"
        bordered
        dataSource={Object.entries(params)}
        renderItem={([key, value]) => (
          <List.Item>
            <Text strong>{key}:</Text> {String(value)}
          </List.Item>
        )}
      />
    );
  };
  
  // Table columns definition
  const columns = [
    {
      title: 'Model Set',
      dataIndex: 'set_name',
      key: 'set_name',
      sorter: (a, b) => (a.set_name || '').localeCompare(b.set_name || ''),
      render: (text) => text ? <Tag color="blue"><CodeSandboxOutlined /> {text}</Tag> : 'N/A',
    },
    {
      title: 'Model Name',
      dataIndex: 'model_name',
      key: 'model_name',
      sorter: (a, b) => a.model_name.localeCompare(b.model_name),
    },
    {
      title: 'Scaler Name',
      dataIndex: 'scaler_name',
      key: 'scaler_name',
      render: (text) => text ? <><PaperClipOutlined /> {text}</> : <Text type="secondary">N/A</Text>,
    },
    {
      title: 'Input Shape',
      dataIndex: 'input_shape',
      key: 'input_shape',
      render: (shape) => shape ? <Tag><TableOutlined /> {shape}</Tag> : <Text type="secondary">N/A</Text>,
    },
    {
      title: 'Output Shape',
      dataIndex: 'output_shape',
      key: 'output_shape',
      render: (shape) => shape ? <Tag><TableOutlined /> {shape}</Tag> : <Text type="secondary">N/A</Text>,
    },
    {
      title: 'Actions',
      key: 'actions',      render: (_, record) => (
        <Space>
          <Button
            type="link"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedModel(record);
              setDetailsModalVisible(true);
            }}
          >
            Details
          </Button>
        </Space>
      )
    }  ];
  
  // Not authenticated or not admin view
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
  
  if (isAuthenticated && user && user.role !== 'admin') {
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
  
  return (
    <Spin spinning={loading && models.length === 0} tip="Loading prediction models..." size="large" style={{ minHeight: 'calc(100vh - 48px)' }}>
      <div style={{ padding: '24px' }}>
        {contextHolder}
        {error && !loading && (
          <Alert
            message="Error Loading Models"
            description={error.response?.data?.detail || error.message || 'Failed to load prediction models.'}
            type="error"
            showIcon
            style={{ marginBottom: '24px' }}
          />
        )}
        <Title level={2}><CodeSandboxOutlined /> Discovered Prediction Models</Title>
        <Paragraph>
          View details of machine learning models discovered in the model storage.
          These models are available for traffic forecasting.
        </Paragraph>

        <Card>
          <Table 
            columns={columns}
            dataSource={models}
            rowKey="key" // Using the key field
            loading={loading}
            pagination={{ pageSize: 10 }}
          />
        </Card>
        
        {/* Model Details Modal */}
        <Modal
          title={
            <Space>
              <InfoCircleOutlined />
              <span>Model Details: {selectedModel?.model_name}</span>
            </Space>
          }
          open={detailsModalVisible}
          onCancel={() => setDetailsModalVisible(false)}
          footer={<Button onClick={() => setDetailsModalVisible(false)}>Close</Button>}
          width={800}
          styles={{ body: { maxHeight: '60vh', overflowY: 'auto' } }}
        >
          {selectedModel && (
            <Descriptions title="" bordered column={1} layout="vertical">
              <Descriptions.Item label="Model Set">
                {selectedModel.set_name}
              </Descriptions.Item>
              <Descriptions.Item label="Model Display Name">
                {selectedModel.model_name} 
              </Descriptions.Item>
              <Descriptions.Item label="Horizon Name (Raw)">
                {selectedModel.horizon_name}
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={selectedModel.status === 'Available' ? 'green' : 'orange'}>{selectedModel.status}</Tag>
                {selectedModel.status_load_error && 
                  <Tag color="red" style={{ marginLeft: 8 }}>{selectedModel.status_load_error}</Tag>}
              </Descriptions.Item>
               <Descriptions.Item label="Model File Path">
                {selectedModel.model_path ? <Text copyable code>{selectedModel.model_path}</Text> : 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Scaler Name">
                {selectedModel.scaler_name || 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Scaler File Path">
                {selectedModel.scaler_path ? <Text copyable code>{selectedModel.scaler_path}</Text> : 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Features Order File Path">
                {selectedModel.features_order_path ? <Text copyable code>{selectedModel.features_order_path}</Text> : 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Input Features Order">
                {selectedModel.features_order ? 
                  (typeof selectedModel.features_order === 'string' || selectedModel.features_order.error ? 
                    <Text type="danger">{selectedModel.features_order.error || selectedModel.features_order}</Text> :
                    <List
                      size="small"
                      bordered
                      dataSource={selectedModel.features_order}
                      renderItem={(item, index) => <List.Item>{index + 1}. {item}</List.Item>}
                      style={{ maxHeight: '150px', overflowY: 'auto' }}
                    />
                  )
                  : <Text type="secondary">N/A</Text>}
              </Descriptions.Item>
              <Descriptions.Item label="All Models Metadata File Path">
                {selectedModel.all_models_metadata_path ? <Text copyable code>{selectedModel.all_models_metadata_path}</Text> : 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Model Specific Metadata">
                {selectedModel.model_specific_metadata ? 
                  (typeof selectedModel.model_specific_metadata === 'string' || selectedModel.model_specific_metadata.error ? 
                    <Text type="danger">{selectedModel.model_specific_metadata.error || selectedModel.model_specific_metadata}</Text> :
                    <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px', maxHeight: '200px', overflowY: 'auto' }}>
                      {JSON.stringify(selectedModel.model_specific_metadata, null, 2)}
                    </pre>
                  )
                  : <Text type="secondary">N/A</Text>}
              </Descriptions.Item>
              <Descriptions.Item label="Input Shape">
                {selectedModel.input_shape || 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Output Shape">
                {selectedModel.output_shape || 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="Model Parameters (from Keras config)" span={1}>
                {displayParameters(selectedModel.parameters)}
              </Descriptions.Item>
              <Descriptions.Item label="Model Summary (from Keras)" span={1}>
                {selectedModel.summary ? 
                  <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', backgroundColor: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
                    {selectedModel.summary}
                  </pre> 
                  : <Text type="secondary">N/A</Text>}
              </Descriptions.Item>            </Descriptions>
          )}
        </Modal>
        
      </div>
    </Spin>
  );
};

export default PredictionModelsPage;