"use client";
import { useState, useEffect, useCallback } from "react";
import { 
  Row, Col, Card, Statistic, Typography, Spin, 
  Alert, Button, List, Space, Badge
} from "antd";
import { 
  DatabaseOutlined, CameraOutlined, MessageOutlined,
  BellOutlined, RightOutlined, WarningOutlined, CheckCircleOutlined,
  InfoCircleOutlined, LineChartOutlined
} from "@ant-design/icons";
import axiosInstance from '@/utils/axiosInstance';
import { useRouter } from "next/navigation";
import { useAuth } from '@/context/authContext';
import Link from 'next/link';
import { Line } from '@ant-design/charts';
import dayjs from 'dayjs';

const { Title, Text, Paragraph } = Typography;

const getCongestionInfoForChart = (speed) => {
  if (speed < 10) return { level: 'F', color: '#8B0000', description: 'Gridlock' };
  if (speed < 20) return { level: 'E', color: '#FF0000', description: 'Severe' };
  if (speed < 30) return { level: 'D', color: '#FFA500', description: 'Heavy' };
  if (speed < 40) return { level: 'C', color: '#FFFF00', description: 'Moderate' };
  if (speed < 50) return { level: 'B', color: '#90EE90', description: 'Light' };
  return { level: 'A', color: '#008000', description: 'Free Flow' };
};

const DashBoardPage = () => {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();
  const [dashboardData, setDashboardData] = useState({
    stats: {
      totalCameras: 0,
      activeCameras: 0,
      inactiveCameras: 0,
      maintenanceCameras: 0,
      totalReports: 0,
      pendingReports: 0,
      totalModels: 0,
    },
    recentReports: [],
  });
  const [loadingDashboard, setLoadingDashboard] = useState(true);
  const [error, setError] = useState(null);

  const [availableMlModels, setAvailableMlModels] = useState([]);
  const [autoForecastData, setAutoForecastData] = useState(null);
  const [loadingAutoForecast, setLoadingAutoForecast] = useState(false);
  const [autoForecastError, setAutoForecastError] = useState(null);

  // Dynamic title for the auto forecast chart
  const [autoForecastChartTitle, setAutoForecastChartTitle] = useState("Traffic Forecast Snapshot");

  const fetchDashboardData = useCallback(async () => {
    setLoadingDashboard(true);
    setError(null);
    try {
      const [camerasResponse, reportsResponse, modelsApiResponse] = await Promise.all([
        axiosInstance.get('/api/admin/cameras').catch(e => { console.error("Error fetching cameras:", e); return { data: [], error: true }; }),
        axiosInstance.get('/api/admin/reports').catch(e => { console.error("Error fetching reports:", e); return { data: [], error: true }; }),
        axiosInstance.get('/api/ml/models').catch(e => { console.error("Error fetching admin models:", e); return { data: {models: []}, error: true }; })
      ]);

      let activeCameras = 0, inactiveCameras = 0, maintenanceCameras = 0;
      const allCameras = camerasResponse.data || [];
      allCameras.forEach(camera => {
        if (camera.status === 'Active') activeCameras++;
        else if (camera.status === 'Inactive') inactiveCameras++;
        else if (camera.status === 'Maintenance') maintenanceCameras++;
      });

      const allReports = reportsResponse.data || [];
      const pendingReports = allReports.filter(report => 
        report.status && (report.status.toLowerCase() === 'submitted' || report.status.toLowerCase() === 'processing')
      ).length;
      const sortedReports = [...allReports].sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0));
      const recentFetchedReports = sortedReports.slice(0, 3);
      
      const allAdminModels = modelsApiResponse.data?.models || (Array.isArray(modelsApiResponse.data) ? modelsApiResponse.data : []) || [];
      const totalAdminModels = allAdminModels.length;

      setDashboardData({
        stats: {
          totalCameras: allCameras.length, activeCameras, inactiveCameras, maintenanceCameras,
          totalReports: allReports.length, pendingReports,
          totalModels: totalAdminModels,
        },
        recentReports: recentFetchedReports,
      });

    } catch (err) {
      console.error("Error processing dashboard data:", err);
      setError(err.message || "An error occurred while fetching or processing general dashboard data.");
    } finally {
      setLoadingDashboard(false);
    }
  }, []);

  const fetchMlModelsForChart = useCallback(async () => {
    try {
      const response = await axiosInstance.get('/api/ml/models');
      setAvailableMlModels(response.data || []);
    } catch (err) {
      console.error('Error fetching available ML models for chart:', err);
      setAutoForecastError('Failed to load ML models for the forecast chart.');
      setAvailableMlModels([]);
    } 
  }, []);

  const fetchAutoForecast = useCallback(async () => {
    if (!availableMlModels || availableMlModels.length === 0) {
      console.log("No ML models available yet for auto forecast.")
      // Update chart title even if no models are available initially
      setAutoForecastChartTitle("Traffic Forecast (No Models Available)");
      return;
    }
    setLoadingAutoForecast(true);
    setAutoForecastError(null);
    setAutoForecastData(null);

    const selectedModel = availableMlModels[0];
    const currentTime = dayjs();
    const payload = {
      model_identifier: selectedModel.id,
      day_of_week: currentTime.day() === 0 ? 6 : currentTime.day() - 1,
      time_of_day: currentTime.format('HH:mm'),
      weather_harsh: false,
    };

    try {
      const response = await axiosInstance.post('/api/ml/predict', payload);
      if (response.data && response.data.predicted_speeds) {
        const enrichedPredictedSpeeds = response.data.predicted_speeds.map(speed => ({
          speed: parseFloat(speed.toFixed(2)),
          congestion: getCongestionInfoForChart(speed)
        }));
        setAutoForecastData({...response.data, enrichedPredictedSpeeds});
        // Update chart title with model info
        const horizon = selectedModel.inferred_horizon || "Current";
        setAutoForecastChartTitle(`${horizon} Forecast for Mai Chí Thọ - Hầm Thủ Thiêm`);

      } else {
        setAutoForecastData(response.data);
        setAutoForecastChartTitle("Forecast Data Incomplete for Mai Chí Thọ - Hầm Thủ Thiêm");
      }
    } catch (err) {
      console.error('Error fetching auto traffic forecast:', err);
      setAutoForecastError(err.response?.data?.detail || 'Failed to generate automatic traffic forecast.');
      setAutoForecastData(null);
      setAutoForecastChartTitle("Forecast Error for Mai Chí Thọ - Hầm Thủ Thiêm");
    } finally {
      setLoadingAutoForecast(false);
    }
  }, [availableMlModels]);

  useEffect(() => {
    if (isAuthenticated === false) {
      router.push('/auth/login');
      return;
    }
    if (isAuthenticated && user) {
      const isAdmin = user.role && user.role.toLowerCase() === 'admin';
      if (!isAdmin) {
        router.push('/user/map'); 
        return;
      }
      setLoadingDashboard(true);
      setError(null);
      setAutoForecastError(null);

      fetchDashboardData();
      fetchMlModelsForChart(); 
    }
  }, [isAuthenticated, user, router, fetchDashboardData, fetchMlModelsForChart]);

  useEffect(() => {
    if (availableMlModels.length > 0) {
      fetchAutoForecast();
    }
  }, [availableMlModels, fetchAutoForecast]);

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  const getRelativeTime = (dateString) => {
    if (!dateString) return 'N/A';
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now - date;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);
    
    if (diffSec < 0) return 'Future date';
    if (diffSec < 60) return `${diffSec} seconds ago`;
    if (diffMin < 60) return `${diffMin} minutes ago`;
    if (diffHour < 24) return `${diffHour} hours ago`;
    if (diffDay < 30) return `${diffDay} days ago`;
    
    return formatDate(dateString);
  };
  
  const getReportStatusBadge = (status) => {
    const s = status ? status.toLowerCase() : '';
    if (s === 'submitted') return <Badge status="processing" text="Submitted" />;
    if (s === 'processing') return <Badge status="warning" text="Processing" />;
    if (s === 'verified' || s === 'resolved') return <Badge status="success" text={status} />;
    if (s === 'rejected' || s === 'invalid') return <Badge status="error" text={status} />;
    return <Badge status="default" text={status || "Unknown"} />;
  };

  if (isAuthenticated === undefined || (isAuthenticated === false && typeof isAuthenticated === 'boolean')) {
    return (
      <div style={{ padding: '24px', textAlign: 'center', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 'calc(100vh - 100px)' }}>
        <Spin size="large" tip="Authenticating..." />
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
  
  const autoChartConfig = autoForecastData?.enrichedPredictedSpeeds
  ? {
      data: autoForecastData.predicted_timestamps.map((ts, index) => ({
        timestamp: ts,
        value: autoForecastData.enrichedPredictedSpeeds[index].speed,
        congestionLevel: autoForecastData.enrichedPredictedSpeeds[index].congestion.level,
        type: autoForecastData.model_info?.name || 'Prediction',
      })),
      xField: 'timestamp',
      yField: 'value',
      seriesField: 'congestionLevel',
      color: ({ congestionLevel }) => getCongestionInfoForChart(autoForecastData.enrichedPredictedSpeeds.find(p => p.congestion.level === congestionLevel)?.speed ?? 0).color,
      xAxis: { title: { text: 'Time' }, label: { rotate: 0.3 } },
      yAxis: { title: { text: 'Speed (km/h)' }, min: 0, max: 70 },
      smooth: true,
      tooltip: { formatter: (datum) => ({ name: `${datum.congestionLevel} (${datum.type})`, value: `${datum.value} km/h at ${datum.timestamp}` }) },
      legend: { position: 'top-right' },
      height: 300,
    }
  : null;

  return (
    <Spin spinning={loadingDashboard && !dashboardData.stats.totalCameras} tip="Loading admin dashboard..." size="large" style={{ minHeight: 'calc(100vh - 48px)' }}>
      <div style={{ padding: '24px' }}>
        {error && (
          <Alert
            message="Error Loading Dashboard Data"
            description={error}
            type="error"
            showIcon
            style={{ marginBottom: '24px' }}
          />
        )}

        {!loadingDashboard && !error && dashboardData && dashboardData.stats && (
          <>
            <Title level={2}>Admin Dashboard</Title>
            <Paragraph>
              Welcome to the administration dashboard. Overview of system components.
            </Paragraph>
            
            <div style={{ marginBottom: '24px' }}>
              <Row gutter={[16, 16]}>
                <Col xs={24} md={12} lg={8}>
                  <Card 
                    hoverable 
                    style={{ height: '100%' }}
                    onClick={() => router.push('/admin/prediction-models')}
                  >
                    <div style={{ textAlign: 'center' }}>
                      <DatabaseOutlined style={{ fontSize: '36px', color: '#1890ff', marginBottom: '16px' }} />
                      <Title level={4}>Prediction Models</Title> 
                      <Paragraph>
                        Overview of deployed forecast models.
                      </Paragraph>
                      <Space>
                        <Statistic
                          title="Total Models"
                          value={dashboardData.stats.totalModels}
                          valueStyle={{ color: '#1890ff' }}
                        />
                      </Space>
                      <Button type="primary" block style={{ marginTop: '16px' }} onClick={() => router.push('/admin/prediction-models')}>
                        View Models
                      </Button>
                    </div>
                  </Card>
                </Col>
                
                <Col xs={24} md={12} lg={8}>
                  <Card 
                    hoverable 
                    style={{ height: '100%' }}
                    onClick={() => router.push('/admin/managecamera')}
                  >
                    <div style={{ textAlign: 'center' }}>
                      <CameraOutlined style={{ fontSize: '36px', color: '#722ed1', marginBottom: '16px' }} />
                      <Title level={4}>Camera Management</Title>
                      <Paragraph>
                        Status of traffic cameras in the system.
                      </Paragraph>
                      <Row gutter={[16, 16]} justify="center"> {/* gutter để tạo khoảng cách giữa các cột và hàng */}
                        <Col span={12}> {/* Chiếm 12/24 cột, tức là 1 nửa */}
                            <Statistic
                                title="Active"
                                value={dashboardData.stats.activeCameras}
                                valueStyle={{ color: '#52c41a' }}
                            />
                        </Col>
                        <Col span={12}>
                            <Statistic
                                title="Maintenance"
                                value={dashboardData.stats.maintenanceCameras}
                                valueStyle={{ color: '#faad14' }}
                            />
                        </Col>
                        <Col span={12}>
                            <Statistic
                                title="Inactive"
                                value={dashboardData.stats.inactiveCameras}
                                valueStyle={{ color: '#ff4d4f' }}
                            />
                        </Col>
                        <Col span={12}>
                            <Statistic
                                title="Total"
                                value={dashboardData.stats.totalCameras}
                            />
                        </Col>
                    </Row>
                      <Button type="primary" style={{ marginTop: '16px', backgroundColor: '#722ed1', borderColor: '#722ed1' }}>
                        Manage Cameras
                      </Button>
                    </div>
                  </Card>
                </Col>
                
                <Col xs={24} md={12} lg={8}>
                  <Card 
                    hoverable 
                    style={{ height: '100%' }}
                    onClick={() => router.push('/admin/reports')}
                  >
                    <div style={{ textAlign: 'center' }}>
                      <MessageOutlined style={{ fontSize: '36px', color: '#fa8c16', marginBottom: '16px' }} />
                      <Title level={4}>User Reports</Title>
                      <Paragraph>
                        Review and process user-submitted reports.
                      </Paragraph>
                       <Space>
                        <Statistic 
                          title="Pending" 
                          value={dashboardData.stats.pendingReports}
                          valueStyle={{ color: dashboardData.stats.pendingReports > 0 ? '#fa8c16' : '#8c8c8c' }}
                        />
                        <Statistic 
                          title="Total Reports" 
                          value={dashboardData.stats.totalReports}
                        />
                      </Space>
                      <Button
                        type="primary"
                        block
                        style={{ marginTop: '16px', backgroundColor: '#fa8c16', borderColor: '#fa8c16' }}
                        onClick={() => router.push('/admin/reports')}
                      >
                        Process Reports
                      </Button>
                    </div>
                  </Card>
                </Col>
              </Row>
            </div>
            
            <Row gutter={[16, 16]} style={{ marginTop: '24px' }}>
              <Col xs={24} lg={12}>
                <Card title="Recent Reports" extra={<Link href="/admin/reports">View All Reports</Link>}>
                  {dashboardData.recentReports?.length > 0 ? (
                    <List
                      dataSource={dashboardData.recentReports}
                      renderItem={item => (
                        <List.Item
                          actions={[
                            <Button 
                              key={`view-${item.id}`} 
                              type="link" 
                              onClick={() => router.push(`/admin/reports?report_id=${item.id}`)}
                            >
                              View <RightOutlined />
                            </Button>
                          ]}
                        >
                          <List.Item.Meta
                            title={<Link href={`/admin/reports?report_id=${item.id}`}>{item.title || `Report ID: ${item.id}`}</Link>}
                            description={
                              <Space>
                                {getReportStatusBadge(item.status)}
                                <Text type="secondary">{getRelativeTime(item.created_at || item.submittedAt)}</Text>
                                {item.type && <Text strong>Type: {item.type}</Text>}
                              </Space>
                            }
                          />
                        </List.Item>
                      )}
                    />
                  ) : (
                    <div style={{ textAlign: 'center', padding: '20px 0' }}>
                      <Text type="secondary">No recent reports found.</Text>
                    </div>
                  )}
                </Card>
              </Col>

              <Col xs={24} lg={12}>
                <Card title={<Space><LineChartOutlined />{autoForecastChartTitle}</Space>}>
                  {loadingAutoForecast && <div style={{textAlign: 'center', padding: '20px'}}><Spin tip="Generating forecast..." /></div>}
                  {autoForecastError && !loadingAutoForecast && 
                    <Alert message="Forecast Error" description={autoForecastError} type="error" showIcon />}
                  {!loadingAutoForecast && !autoForecastError && autoChartConfig && 
                    <Line {...autoChartConfig} />}
                  {!loadingAutoForecast && !autoForecastError && !autoChartConfig && 
                    <div style={{ textAlign: 'center', padding: '20px 0' }}><Text type="secondary">No forecast data to display. Models might be unavailable or an error occurred.</Text></div>}
                </Card>
              </Col>
            </Row>
          </>
        )}
      </div>
    </Spin>
  );
};

export default DashBoardPage;
