"use client";

import {
  Alert,
  Button,
  Col,
  InputNumber,
  Row,
  Space,
  Spin,
  Typography,
  message
} from 'antd';
import { useCallback, useEffect, useRef, useState } from 'react';

const { Paragraph } = Typography;
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL_BE || "http://localhost:8000";

const ROIEditorComponent = ({ camera, isVisible, onSave, onCancel }) => {
  const [messageApi, contextHolder] = message.useMessage();

  // ROI state
  const [roiPoints, setRoiPoints] = useState([]);
  const [normalizedPoints, setNormalizedPoints] = useState([]);
  const [roiWidthMeters, setRoiWidthMeters] = useState(50);
  const [roiHeightMeters, setRoiHeightMeters] = useState(100);

  // Image state
  const [imageSrc, setImageSrc] = useState("");
  const [imageDims, setImageDims] = useState({ width: 0, height: 0 });
  const [imageError, setImageError] = useState(false);
  const [imageOffset, setImageOffset] = useState({ x: 0, y: 0 });
  const [imageLoading, setImageLoading] = useState(false);

  // Refs
  const svgContainerRef = useRef(null);
  const imageRef = useRef(null);
  const [draggingPointIndex, setDraggingPointIndex] = useState(null);
  const imageTimestamp = useRef(Date.now());

  // Initialize the ROI dimensions from camera data
  useEffect(() => {
    if (camera?.roi) {
      if (typeof camera.roi.roi_width_meters === 'number') {
        setRoiWidthMeters(camera.roi.roi_width_meters);
      }
      if (typeof camera.roi.roi_height_meters === 'number') {
        setRoiHeightMeters(camera.roi.roi_height_meters);
      }
    }
  }, [camera]);

  // Effect that loads the image when camera changes
  useEffect(() => {
    if (!isVisible || !camera?.id) {
      return; // Don't load if not visible
    }

    const thumbnailUrl = camera.thumbnail_url;
    if (!thumbnailUrl) {
      setImageError(true);
      messageApi.error("Camera data is missing the thumbnail URL.");
      return;
    }

    // Create full URL with timestamp for cache busting
    imageTimestamp.current = Date.now();
    const fullImageUrl = thumbnailUrl.startsWith('http')
      ? `${thumbnailUrl}?t=${imageTimestamp.current}`
      : `${API_BASE_URL}${thumbnailUrl}?t=${imageTimestamp.current}`;
    console.log("[ROIEditor] Loading image:", fullImageUrl);
    setImageLoading(true);
    setImageError(false);
    setImageSrc(fullImageUrl);

    // Reset dimensions until image loads
    setImageDims({ width: 0, height: 0 });
    setImageOffset({ x: 0, y: 0 });

  }, [camera, isVisible, messageApi]);

  // Handle image load complete - setup canvas dimensions
  const handleImageLoadComplete = useCallback((img) => {
    console.log("[ROIEditor] Image loaded successfully", img.naturalWidth, img.naturalHeight);
    setImageLoading(false);

    const container = svgContainerRef.current;
    if (!container || img.naturalWidth === 0 || img.naturalHeight === 0) {
      console.error("[ROIEditor] Invalid image dimensions or container");
      setImageError(true);
      return;
    }

    const containerWidth = container.offsetWidth;
    const containerHeight = container.offsetHeight;
    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;

    // Calculate image dimensions to fit in container
    const imgRatio = imgWidth / imgHeight;
    const containerRatio = containerWidth / containerHeight;

    let displayWidth, displayHeight, offsetX = 0, offsetY = 0;

    if (imgRatio > containerRatio) {
      // Image is wider than container
      displayWidth = containerWidth;
      displayHeight = containerWidth / imgRatio;
      offsetY = (containerHeight - displayHeight) / 2;
    } else {
      // Image is taller than container
      displayHeight = containerHeight;
      displayWidth = containerHeight * imgRatio;
      offsetX = (containerWidth - displayWidth) / 2;
    }

    setImageDims({ width: displayWidth, height: displayHeight });
    setImageOffset({ x: offsetX, y: offsetY });

    // If camera has existing ROI points, convert them to pixel coordinates
    if (camera?.roi?.points && camera.roi.points.length === 4) {
      const pixelPoints = camera.roi.points.map(point => ({
        x: point.x * displayWidth,
        y: point.y * displayHeight
      }));
      setRoiPoints(pixelPoints);
      setNormalizedPoints(camera.roi.points);
    } else {
      // Clear any points
      setRoiPoints([]);
      setNormalizedPoints([]);
    }
  }, [camera]);

  // Image error handler
  const handleImageError = useCallback((e) => {
    console.error("[ROIEditor] Image failed to load:", e);
    setImageLoading(false);
    setImageError(true);
    setImageDims({ width: 0, height: 0 });
    messageApi.error("Failed to load camera image. Please try refreshing all thumbnails.");
  }, [messageApi]);

  // Reset state when component is hidden
  useEffect(() => {
    if (!isVisible) {
      setImageSrc(null);
      setRoiPoints([]);
      setNormalizedPoints([]);
      setImageDims({ width: 0, height: 0 });
      setImageOffset({ x: 0, y: 0 });
      setImageError(false);
      setImageLoading(false);
    }
  }, [isVisible]);

  // Calculate normalized points when ROI points change
  useEffect(() => {
    if (roiPoints.length > 0 && imageDims.width > 0 && imageDims.height > 0) {
      const normalized = roiPoints.map(p => ({
        x: Math.max(0, Math.min(1, p.x / imageDims.width)),
        y: Math.max(0, Math.min(1, p.y / imageDims.height))
      }));
      setNormalizedPoints(normalized);
    } else if (normalizedPoints.length > 0 && roiPoints.length === 0) {
      setNormalizedPoints([]);
    }
  }, [roiPoints, imageDims]);

  // Get mouse position relative to image
  const getMousePosition = useCallback((e) => {
    const svg = svgContainerRef.current?.querySelector('svg');
    if (!svg || !imageDims.width || !imageDims.height) return null;

    const rect = svg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert to image coordinates
    const imageX = x - imageOffset.x;
    const imageY = y - imageOffset.y;

    // Clamp to image bounds
    return {
      x: Math.max(0, Math.min(imageDims.width, imageX)),
      y: Math.max(0, Math.min(imageDims.height, imageY))
    };
  }, [imageDims, imageOffset]);

  // Handle SVG canvas click to add points
  const handleSvgClick = useCallback((e) => {
    if (imageError || roiPoints.length >= 4) {
      if (roiPoints.length >= 4) {
        messageApi.info("Already have 4 points. Drag points to adjust or reset.");
      }
      return;
    }

    const pos = getMousePosition(e);
    if (pos) {
      setRoiPoints(prev => [...prev, pos]);
    }
  }, [imageError, roiPoints.length, getMousePosition, messageApi]);

  // Handle drag operations
  const handlePointMouseDown = useCallback((index, e) => {
    e.stopPropagation();
    setDraggingPointIndex(index);
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (draggingPointIndex === null) return;

    const pos = getMousePosition(e);
    if (pos) {
      setRoiPoints(points =>
        points.map((p, i) => i === draggingPointIndex ? pos : p)
      );
    }
  }, [draggingPointIndex, getMousePosition]);

  const handleMouseUp = useCallback(() => {
    setDraggingPointIndex(null);
  }, []);

  // Add/remove event listeners for dragging
  useEffect(() => {
    if (draggingPointIndex !== null) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [draggingPointIndex, handleMouseMove, handleMouseUp]);

  // Save ROI handler
  const handleSave = useCallback(() => {
    if (normalizedPoints.length !== 4) {
      messageApi.error("Please define exactly 4 points for the ROI");
      return;
    }

    if (!roiWidthMeters || roiWidthMeters <= 0 || !roiHeightMeters || roiHeightMeters <= 0) {
      messageApi.error("Please enter valid ROI dimensions (meters)");
      return;
    }

    onSave(camera.id, normalizedPoints, {
      width: roiWidthMeters,
      height: roiHeightMeters
    });
  }, [camera?.id, normalizedPoints, roiWidthMeters, roiHeightMeters, onSave, messageApi]);

  // Reset points handler  
  const handleReset = useCallback(() => {
    setRoiPoints([]);
    setNormalizedPoints([]);
  }, []);

  // Render ROI editor UI
  const isImageReady =  true;
  const showLoadingIndicator = isVisible && (imageLoading || (!imageSrc && !imageError));
  console.log("Value: ", imageLoading)
  if (!isVisible) return null;

  return (
    <div className="roi-editor">
      {contextHolder}

      <Row gutter={16} style={{ marginBottom: '16px' }} align="middle">
        <Col flex="auto">
          <Space wrap>
            <span>ROI Width (m):</span>
            <InputNumber
              min={0.1}
              max={1000}
              step={0.1}
              value={roiWidthMeters}
              onChange={setRoiWidthMeters}
              style={{ width: '80px' }}
            />
            <span>ROI Height (m):</span>
            <InputNumber
              min={0.1}
              max={1000}
              step={0.1}
              value={roiHeightMeters}
              onChange={setRoiHeightMeters}
              style={{ width: '80px' }}
            />
          </Space>
        </Col>
        <Col>
          <Button
            onClick={handleReset}
            disabled={roiPoints.length === 0 || imageError || imageLoading}
          >
            Reset Points
          </Button>
        </Col>
      </Row>

      <Paragraph type="secondary">
        Click on the image to define the 4 corners of the Region of Interest. Drag points to adjust.
      </Paragraph>

      <div
        ref={svgContainerRef}
        style={{
          position: 'relative',
          width: '100%',
          aspectRatio: '4/3',
          backgroundColor: '#f0f0f0',
          overflow: 'hidden',
          border: '1px solid #ccc'
        }}
      >
        {/* Loading indicator */}
        {showLoadingIndicator && (
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            position: 'absolute',
            inset: 0
          }}>
            <Spin size="large" tip="Loading image..." />
          </div>
        )}

        {/* Error message */}
        {imageError && (
          <div style={{
            padding: '20px',
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Alert
              message="Error Loading Image"
              description="Could not load the camera snapshot. Please try refreshing all thumbnails."
              type="error"
              showIcon
            />
          </div>
        )}

        {/* Image and ROI editor */}
        {isImageReady && (
          <div style={{
            position: 'relative',
            width: '100%',
            height: '100%',
          }}>
            {/* The image */}
            <img
              src={imageSrc}
              alt="Camera frame for ROI"
              onLoad={(e) => handleImageLoadComplete(e.target)}
              onError={handleImageError}
              style={{
                position: 'absolute',
                width: imageDims.width,
                height: imageDims.height,
                left: imageOffset.x,
                top: imageOffset.y,
                userSelect: 'none',
                pointerEvents: 'none',
              }}
            />

            {/* SVG overlay for drawing points */}
            <svg
              width="100%"
              height="100%"
              onClick={handleSvgClick}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                cursor: roiPoints.length < 4 ? 'crosshair' : 'default',
                zIndex: 10
              }}
            >
              {/* ROI polygon */}
              {roiPoints.length > 1 && (
                <polygon
                  points={roiPoints.map(p => `${p.x + imageOffset.x},${p.y + imageOffset.y}`).join(' ')}
                  fill={roiPoints.length === 4 ? "rgba(255, 255, 0, 0.3)" : "none"}
                  stroke="yellow"
                  strokeWidth="2"
                />
              )}

              {/* Control points */}
              {roiPoints.map((point, index) => (
                <circle
                  key={index}
                  cx={point.x + imageOffset.x}
                  cy={point.y + imageOffset.y}
                  r="6"
                  fill="red"
                  stroke="black"
                  strokeWidth="1"
                  onMouseDown={(e) => handlePointMouseDown(index, e)}
                  style={{ cursor: 'move' }}
                />
              ))}
            </svg>
          </div>
        )}
      </div>

      <Space style={{ marginTop: '16px', justifyContent: 'flex-end', width: '100%' }}>
        <Button onClick={onCancel}>Cancel</Button>
        <Button
          type="primary"
          onClick={handleSave}
          disabled={normalizedPoints.length !== 4 || imageError || showLoadingIndicator}
        >
          Save ROI
        </Button>
      </Space>
    </div>
  );
};

export default ROIEditorComponent;