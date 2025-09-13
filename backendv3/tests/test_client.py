# test_cors.py (hoặc test_client.py)
from fastapi.testclient import TestClient
# Đảm bảo import đúng app của bạn. Ví dụ: from app.main import app
# Nếu main.py nằm trực tiếp trong thư mục backendv3 và bạn chạy pytest từ backendv3
from app.main import app
# Hoặc nếu main.py nằm trong backendv3/app/main.py và bạn chạy pytest từ backendv3
# from app.main import app 


client = TestClient(app)

def test_cors_put_request():
    print(">>> Running test_cors_put_request - Version 2025-05-21-FINAL <<<")
    # 1. Định nghĩa Origin mà trình duyệt frontend sẽ gửi
    FRONTEND_ORIGIN = "http://localhost:3000"

    # 2. Header cho yêu cầu OPTIONS (preflight)
    #    "Access-Control-Request-Headers" cần liệt kê các header mà yêu cầu chính sẽ gửi
    preflight_headers = {
        "Origin": FRONTEND_ORIGIN,
        "Access-Control-Request-Method": "PUT",
        "Access-Control-Request-Headers": "Content-Type, Authorization",
    }

    # ====================================================================
    # PHẦN TEST YÊU CẦU OPTIONS (PREFLIGHT)
    # ====================================================================
    # Gửi preflight request
    response_options = client.options("/api/admin/cameras/some_dummy_id/status", headers=preflight_headers)
    
    print("\n--- OPTIONS Response Headers ---")
    print(response_options.headers)
    print("-------------------------------")

    assert response_options.status_code == 200 # Preflight request thường trả về 200 OK

    # 2.1. Kiểm tra Access-Control-Allow-Origin
    assert "access-control-allow-origin" in response_options.headers
    # Khi allow_origins chứa "*" VÀ allow_credentials=True, FastAPI sẽ trả về Origin đã gửi
    assert response_options.headers["access-control-allow-origin"] == FRONTEND_ORIGIN 
    assert response_options.headers["vary"] == "Origin" # Header này cũng thường xuất hiện khi allow_origins có wildcard

    # 2.2. Kiểm tra Access-Control-Allow-Methods
    assert "access-control-allow-methods" in response_options.headers
    actual_methods = set(m.strip() for m in response_options.headers["access-control-allow-methods"].split(','))
    # FastAPI khi allow_methods=["*"] sẽ trả về GET,HEAD,POST,PUT,DELETE,PATCH,OPTIONS
    expected_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"} 
    assert actual_methods == expected_methods

    # 2.3. Kiểm tra Access-Control-Allow-Headers
    assert "access-control-allow-headers" in response_options.headers
    actual_allowed_headers_str = response_options.headers["access-control-allow-headers"]
    expected_request_headers_str = preflight_headers["Access-Control-Request-Headers"]
    actual_allowed_headers_set = set(h.strip() for h in actual_allowed_headers_str.split(','))
    expected_request_headers_set = set(h.strip() for h in expected_request_headers_str.split(','))
    assert actual_allowed_headers_set == expected_request_headers_set

    # 2.4. Kiểm tra Access-Control-Allow-Credentials
    assert "access-control-allow-credentials" in response_options.headers
    assert response_options.headers["access-control-allow-credentials"] == "true"

    # ====================================================================
    # PHẦN TEST YÊU CẦU PUT (YÊU CẦU THỰC TẾ)
    # ====================================================================
    # Header cho yêu cầu PUT thực tế. Cần có Origin và các header khác.
    put_request_headers = {
        "Origin": FRONTEND_ORIGIN,
        "Content-Type": "application/json",
        "Authorization": "Bearer some_valid_token_here" # Cần token nếu endpoint yêu cầu xác thực
    }
    # Đây là một ID camera hợp lệ để endpoint có thể xử lý (hoặc ít nhất là không lỗi ngay lập tức)
    camera_id_for_test = "682d6e334e67980679635af9" 
    
    # Gửi PUT request thực tế
    response_put = client.put(
        f"/api/admin/cameras/{camera_id_for_test}/status",
        headers=put_request_headers,
        json={"status": "Active"} # Đảm bảo body này khớp với Pydantic model của endpoint
    )

    print("\n--- PUT Response Headers ---")
    print(response_put.headers)
    print("-------------------------------")

    # 3.1. Kiểm tra Access-Control-Allow-Origin
    assert "access-control-allow-origin" in response_put.headers
    assert response_put.headers["access-control-allow-origin"] == FRONTEND_ORIGIN
    assert response_put.headers["vary"] == "Origin" # Header này cũng thường xuất hiện

    # 3.2. Kiểm tra Access-Control-Allow-Credentials
    assert "access-control-allow-credentials" in response_put.headers
    assert response_put.headers["access-control-allow-credentials"] == "true"

    # 3.3. Kiểm tra Status Code của yêu cầu PUT
    # Vẫn mong đợi server lỗi 500 nếu bạn chưa sửa lỗi backend,
    # nhưng ít nhất CORS đã được giải quyết.
    assert response_put.status_code == 500 
    # Nếu bạn đã sửa lỗi 500, thì nên assert response_put.status_code == 200
    assert response_put.json() == {"detail": "An internal server error occurred while updating camera status."} 
    # Hoặc thông báo lỗi 500 cụ thể của bạn từ exception handler