const generateMockCameras = () => {
const mockCameras = [
  {
    "id": "mock-cam-001",
    "name": "Nút giao Công trường Dân chủ",
    "description": "Camera giao thông khu vực trung tâm Quận 1",
    "stream_url": "https://mock.stream.url/001",
    "status": "Active",
    "location_id": "mock-loc-001",
    "location_detail": {
      "name": "Nút giao Công trường Dân chủ",
      "latitude": 10.777821,
      "longitude": 106.681948

    },
    "thumbnail_url": "/assets/mock-cam-001.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:30:00.000Z",
    "updated_at": "2025-05-20T07:30:00.000Z",
    "congestion_level": 4,
    "congestion_text": "Tắc nghẽn nhẹ"
  },
  {
    "id": "mock-cam-002",
    "name": "Nút giao Công trường Dân chủ",
    "description": "Điểm nóng giao thông Quận Bình Thạnh",
    "stream_url": "https://mock.stream.url/002",
    "status": "Active",
    "location_id": "mock-loc-002",
    "location_detail": {
      "name": "Nút giao Công trường Dân chủ",
      "latitude": 10.777736,
      "longitude": 106.681384
    },
    "thumbnail_url": "/assets/mock-cam-002.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:31:00.000Z",
    "updated_at": "2025-05-20T07:31:00.000Z",
    "congestion_level": 5,
    "congestion_text": "Tắc nghẽn nặng"
  },
  {
    "id": "mock-cam-003",
    "name": "Lý Chính Thắng - Nguyễn Thông",
    "description": "Cửa ngõ vào trung tâm TP.HCM",
    "stream_url": "https://mock.stream.url/003",
    "status": "Active",
    "location_id": "mock-loc-003",
    "location_detail": {
      "name": "Lý Chính Thắng - Nguyễn Thông",
      "latitude": 10.779386,
      "longitude": 106.682152
    },
    "thumbnail_url": "/assets/mock-cam-003.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:32:00.000Z",
    "updated_at": "2025-05-20T07:32:00.000Z",
    "congestion_level": 3,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-004",
    "name": "Kỳ Đồng - Bà Huyện Thanh Quan",
    "description": "Giao lộ lớn ở Thủ Đức",
    "stream_url": "https://mock.stream.url/004",
    "status": "Active",
    "location_id": "mock-loc-004",
    "location_detail": {
      "name": "Kỳ Đồng - Bà Huyện Thanh Quan",
      "latitude": 10.781678,
      "longitude": 106.681676
    },
    "thumbnail_url": "/assets/mock-cam-004.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:33:00.000Z",
    "updated_at": "2025-05-20T07:33:00.000Z",
    "congestion_level": 2,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-005",
    "name": "Võ Thị Sáu - Bà Huyện Thanh Quan",
    "description": "Ngã tư trọng yếu khu vực Đông TP.HCM",
    "stream_url": "https://mock.stream.url/005",
    "status": "Active",
    "location_id": "mock-loc-005",
    "location_detail": {
      "name": "Võ Thị Sáu - Bà Huyện Thanh Quan",
      "latitude": 10.780156,
      "longitude": 106.684228
    },
    "thumbnail_url": "/assets/mock-cam-005.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:34:00.000Z",
    "updated_at": "2025-05-20T07:34:00.000Z",
    "congestion_level": 4,
    "congestion_text": "Tắc nghẽn nhẹ"
  },
  {
    "id": "mock-cam-006",
    "name": "Điện Biên Phủ - Cách Mạng Tháng Tám",
    "description": "Khu vực sân bay Tân Sơn Nhất",
    "stream_url": "https://mock.stream.url/006",
    "status": "Active",
    "location_id": "mock-loc-006",
    "location_detail": {
      "name": "Điện Biên Phủ - Cách Mạng Tháng Tám",
      "latitude": 10.776657,
      "longitude": 106.683671
    },
    "thumbnail_url": "/assets/mock-cam-006.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:35:00.000Z",
    "updated_at": "2025-05-20T07:35:00.000Z",
    "congestion_level": 3,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-007",
    "name": "Nguyễn Đình Chiểu - Cách Mạng Tháng Tám",
    "description": "Cửa ngõ Tây Bắc TP.HCM",
    "stream_url": "https://mock.stream.url/007",
    "status": "Active",
    "location_id": "mock-loc-007",
    "location_detail": {
      "name": "Nguyễn Đình Chiểu - Cách Mạng Tháng Tám",
      "latitude": 10.775003,
      "longitude": 106.686724
    },
    "thumbnail_url": "/assets/mock-cam-007.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:36:00.000Z",
    "updated_at": "2025-05-20T07:36:00.000Z",
    "congestion_level": 5,
    "congestion_text": "Tắc nghẽn nặng"
  },
  {
    "id": "mock-cam-008",
    "name": "Cách Mạng Tháng Tám - Võ Văn Tần",
    "description": "Khu vực Quận 7",
    "stream_url": "https://mock.stream.url/008",
    "status": "Active",
    "location_id": "mock-loc-008",
    "location_detail": {
      "name": "Cách Mạng Tháng Tám - Võ Văn Tần",
      "latitude": 10.774272,
      "longitude": 106.688112
    },
    "thumbnail_url": "/assets/mock-cam-008.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:37:00.000Z",
    "updated_at": "2025-05-20T07:37:00.000Z",
    "congestion_level": 3,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-009",
    "name": "Cách Mạng Tháng Tám - Nguyễn Thị Minh Khai",
    "description": "Giao lộ Quận 10",
    "stream_url": "https://mock.stream.url/009",
    "status": "Active",
    "location_id": "mock-loc-009",
    "location_detail": {
      "name": "Cách Mạng Tháng Tám - Nguyễn Thị Minh Khai",
      "latitude": 10.773683,
      "longitude": 106.689558
    },
    "thumbnail_url": "/assets/mock-cam-009.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:38:00.000Z",
    "updated_at": "2025-05-20T07:38:00.000Z",
    "congestion_level": 4,
    "congestion_text": "Tắc nghẽn nhẹ"
  },
  {
    "id": "mock-cam-010",
    "name": "Cách Mạng Tháng Tám - Sương Nguyệt Anh",
    "description": "Ngã tư sầm uất Quận Phú Nhuận",
    "stream_url": "https://mock.stream.url/010",
    "status": "Active",
    "location_id": "mock-loc-010",
    "location_detail": {
      "name": "Cách Mạng Tháng Tám - Sương Nguyệt Anh",
      "latitude": 10.773113,
      "longitude": 106.690175
    },
    "thumbnail_url": "/assets/mock-cam-010.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:39:00.000Z",
    "updated_at": "2025-05-20T07:39:00.000Z",
    "congestion_level": 5,
    "congestion_text": "Tắc nghẽn nặng"
  },
  {
    "id": "mock-cam-011",
    "name": "Nguyễn Thị Minh Khai - Bà Huyện Thanh Quan",
    "description": "Giao lộ đông đúc Quận Tân Bình",
    "stream_url": "https://mock.stream.url/011",
    "status": "Active",
    "location_id": "mock-loc-011",
    "location_detail": {
      "name": "Nguyễn Thị Minh Khai - Bà Huyện Thanh Quan",
      "latitude": 10.774833,
      "longitude": 106.690562
    },
    "thumbnail_url": "/assets/mock-cam-011.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:40:00.000Z",
    "updated_at": "2025-05-20T07:40:00.000Z",
    "congestion_level": 4,
    "congestion_text": "Tắc nghẽn nhẹ"
  },
  {
    "id": "mock-cam-012",
    "name": "Nguyễn Thị Minh Khai - Trương Định",
    "description": "Khu vực Quận 9",
    "stream_url": "https://mock.stream.url/012",
    "status": "Active",
    "location_id": "mock-loc-012",
    "location_detail": {
      "name": "Nguyễn Thị Minh Khai - Trương Định",
      "latitude": 10.776015,
      "longitude": 106.691687
    },
    "thumbnail_url": "/assets/mock-cam-012.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:41:00.000Z",
    "updated_at": "2025-05-20T07:41:00.000Z",
    "congestion_level": 2,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-013",
    "name": "Nguyễn Đình Chiểu - Trương Định",
    "description": "Cửa ngõ Đông TP.HCM đi Đồng Nai",
    "stream_url": "https://mock.stream.url/013",
    "status": "Active",
    "location_id": "mock-loc-013",
    "location_detail": {
      "name": "Nguyễn Đình Chiểu - Trương Định",
      "latitude": 10.777927,
      "longitude": 106.689564
    },
    "thumbnail_url": "/assets/mock-cam-013.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:42:00.000Z",
    "updated_at": "2025-05-20T07:42:00.000Z",
    "congestion_level": 3,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-014",
    "name": "Ba Tháng Hai - Cao Thắng",
    "description": "Khu vực Quận 2 (nay là TP. Thủ Đức)",
    "stream_url": "https://mock.stream.url/014",
    "status": "Active",
    "location_id": "mock-loc-014",
    "location_detail": {
      "name": "Ba Tháng Hai - Cao Thắng",
      "latitude": 10.773792,
      "longitude": 106.677780
    },
    "thumbnail_url": "/assets/mock-cam-014.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:43:00.000Z",
    "updated_at": "2025-05-20T07:43:00.000Z",
    "congestion_level": 1,
    "congestion_text": "Thông thoáng"
  },
  {
    "id": "mock-cam-015",
    "name": "Điện Biên Phủ - Cao Thắng",
    "description": "Giao lộ lớn Quận Bình Tân",
    "stream_url": "https://mock.stream.url/015",
    "status": "Active",
    "location_id": "mock-loc-015",
    "location_detail": {
      "name": "Điện Biên Phủ - Cao Thắng",
      "latitude": 10.772764,
      "longitude": 106.679060
    },
    "thumbnail_url": "/assets/mock-cam-015.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:44:00.000Z",
    "updated_at": "2025-05-20T07:44:00.000Z",
    "congestion_level": 5,
    "congestion_text": "Tắc nghẽn nặng"
  },
  {
    "id": "mock-cam-016",
    "name": "Nguyễn Đình Chiểu - Cao Thắng",
    "description": "Khu vực Quận 8",
    "stream_url": "https://mock.stream.url/016",
    "status": "Active",
    "location_id": "mock-loc-016",
    "location_detail": {
      "name": "Nguyễn Đình Chiểu - Cao Thắng",
      "latitude": 10.770032,
      "longitude": 106.682119
    },
    "thumbnail_url": "/assets/mock-cam-016.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:45:00.000Z",
    "updated_at": "2025-05-20T07:45:00.000Z",
    "congestion_level": 3,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-017",
    "name": "Cao Thắng - Võ Văn Tần 2",
    "description": "Khu vực Quận 5/10",
    "stream_url": "https://mock.stream.url/017",
    "status": "Active",
    "location_id": "mock-loc-017",
    "location_detail": {
      "name": "Cao Thắng - Võ Văn Tần 2",
      "latitude": 10.769036,
      "longitude": 106.683238
    },
    "thumbnail_url": "/assets/mock-cam-017.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:46:00.000Z",
    "updated_at": "2025-05-20T07:46:00.000Z",
    "congestion_level": 4,
    "congestion_text": "Tắc nghẽn nhẹ"
  },
  {
    "id": "mock-cam-018",
    "name": "Cao Thắng - Võ Văn Tần 1",
    "description": "Khu vực trung tâm Quận 1",
    "stream_url": "https://mock.stream.url/018",
    "status": "Active",
    "location_id": "mock-loc-018",
    "location_detail": {
      "name": "Cao Thắng - Võ Văn Tần 1",
      "latitude": 10.768471,
      "longitude": 106.683882
    },
    "thumbnail_url": "/assets/mock-cam-018.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:47:00.000Z",
    "updated_at": "2025-05-20T07:47:00.000Z",
    "congestion_level": 2,
    "congestion_text": "Bình thường"
  },
  {
    "id": "mock-cam-019",
    "name": "Nguyễn Thị Minh Khai - Cống Quỳnh",
    "description": "Khu vực Quận 10",
    "stream_url": "https://mock.stream.url/019",
    "status": "Active",
    "location_id": "mock-loc-019",
    "location_detail": {
      "name": "Nguyễn Thị Minh Khai - Cống Quỳnh",
      "latitude": 10.768312,
      "longitude": 106.684343
    },
    "thumbnail_url": "/assets/mock-cam-019.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:48:00.000Z",
    "updated_at": "2025-05-20T07:48:00.000Z",
    "congestion_level": 1,
    "congestion_text": "Thông thoáng"
  },
  {
    "id": "mock-cam-020",
    "name": "Nguyễn Thị Minh Khai - Nguyễn Thượng Hiền",
    "description": "Cửa ngõ vào khu chế xuất Quận 7",
    "stream_url": "https://mock.stream.url/020",
    "status": "Active",
    "location_id": "mock-loc-020",
    "location_detail": {
      "name": "Nguyễn Thị Minh Khai - Nguyễn Thượng Hiền",
      "latitude": 10.770357,
      "longitude": 106.686449
    },
    "thumbnail_url": "/assets/mock-cam-020.jpg",
    "roi": {},
    "online": true,
    "deleted": false,
    "created_at": "2025-05-20T07:49:00.000Z",
    "updated_at": "2025-05-20T07:49:00.000Z",
    "congestion_level": 0,
    "congestion_text": "Thông thoáng"
  }
];
return mockCameras;
}
export default generateMockCameras;