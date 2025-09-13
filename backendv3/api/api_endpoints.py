# Define API Version
API_VERSION = "/api"

# --- Authentication Endpoints (UC01) ---
AUTH_PREFIX = f"{API_VERSION}/auth"
AUTH_ENDPOINTS = {
    "LOGIN": f"{AUTH_PREFIX}/login",
    "LOGOUT": f"{AUTH_PREFIX}/logout",
    "REFRESH_TOKEN": f"{AUTH_PREFIX}/refresh",
    "GET_ME": f"{AUTH_PREFIX}/me",
    "VERIFY_TOKEN": f"{AUTH_PREFIX}/verify-token",
}

# --- User Traffic Map Endpoints (UC02 - Simplified) ---
USER_MAP_PREFIX = f"{API_VERSION}/user/map"
USER_MAP_ENDPOINTS = {
    "GET_TRAFFIC_DENSITY": f"{USER_MAP_PREFIX}/traffic-density", 
    "GET_CONGESTION_STATUS": f"{USER_MAP_PREFIX}/congestion-status", # New endpoint for overall congestion
}

# --- User Route Finding Endpoints (UC03 - Simplified) ---
USER_ROUTES_PREFIX = f"{API_VERSION}/user/routes"
USER_ROUTES_ENDPOINTS = {
    "FIND_ROUTE": f"{USER_ROUTES_PREFIX}/find",
    "GET_SAVED_ROUTES": f"{USER_ROUTES_PREFIX}/saved",
    "SAVE_ROUTE": f"{USER_ROUTES_PREFIX}/save",
    "GET_SAVED_ROUTE_DETAIL": f"{USER_ROUTES_PREFIX}/saved/{{route_id}}",
    "DELETE_SAVED_ROUTE": f"{USER_ROUTES_PREFIX}/saved/{{route_id}}",
}

# --- User Account Management Endpoints (UC08) ---
USER_ACCOUNT_PREFIX = f"{API_VERSION}/user/account"
USER_ACCOUNT_ENDPOINTS = {
    "GET_PROFILE": f"{USER_ACCOUNT_PREFIX}/profile",
    "UPDATE_PROFILE": f"{USER_ACCOUNT_PREFIX}/profile",
    "UPDATE_PASSWORD": f"{USER_ACCOUNT_PREFIX}/password",
}

# --- User Report Submission Endpoints (UC09) ---
USER_REPORTS_PREFIX = f"{API_VERSION}/user/reports"
USER_REPORTS_ENDPOINTS = {
    "SUBMIT_REPORT": f"{USER_REPORTS_PREFIX}",
    "GET_SUBMITTED_REPORTS": f"{USER_REPORTS_PREFIX}",
}

# --- Admin Traffic Map Endpoints (UC02 - Simplified) ---
ADMIN_MAP_PREFIX = f"{API_VERSION}/admin/map"
ADMIN_MAP_ENDPOINTS = {
    "GET_TRAFFIC_DENSITY": f"{ADMIN_MAP_PREFIX}/traffic-density",
    "GET_CONGESTION_STATUS": f"{ADMIN_MAP_PREFIX}/congestion-status", # New endpoint for overall congestion
}

# --- Admin Route Finding Endpoints (UC03 - Simplified) ---
ADMIN_ROUTES_PREFIX = f"{API_VERSION}/admin/routes"
ADMIN_ROUTES_ENDPOINTS = {
    "FIND_ROUTE": f"{ADMIN_ROUTES_PREFIX}/find",
    "GET_SAVED_ROUTES": f"{ADMIN_ROUTES_PREFIX}/saved",
    "SAVE_ROUTE": f"{ADMIN_ROUTES_PREFIX}/save",
    "GET_SAVED_ROUTE_DETAIL": f"{ADMIN_ROUTES_PREFIX}/saved/{{route_id}}",
    "DELETE_SAVED_ROUTE": f"{ADMIN_ROUTES_PREFIX}/saved/{{route_id}}",
}

# --- Admin Account Management Endpoints (UC08) ---
ADMIN_ACCOUNT_PREFIX = f"{API_VERSION}/admin/account"
ADMIN_ACCOUNT_ENDPOINTS = {
    "GET_PROFILE": f"{ADMIN_ACCOUNT_PREFIX}/profile",
    "UPDATE_PROFILE": f"{ADMIN_ACCOUNT_PREFIX}/profile",
    "UPDATE_PASSWORD": f"{ADMIN_ACCOUNT_PREFIX}/password",
}

# --- Admin Report Submission Endpoints (UC09) ---
ADMIN_SUBMIT_REPORTS_PREFIX = f"{API_VERSION}/admin/reports/submit"
ADMIN_SUBMIT_REPORTS_ENDPOINTS = {
    "SUBMIT_REPORT": f"{ADMIN_SUBMIT_REPORTS_PREFIX}",
    "GET_SUBMITTED_REPORTS_BY_ADMIN": f"{ADMIN_SUBMIT_REPORTS_PREFIX}/me",
}

# --- Admin Training Data Management Endpoints (UC11) ---
# ADMIN_TRAINING_DATA_PREFIX = f"{API_VERSION}/admin/training-data"
# ADMIN_TRAINING_DATA_ENDPOINTS = {
#     "LIST_DATASETS": f"{ADMIN_TRAINING_DATA_PREFIX}",
#     "UPLOAD_DATASET": f"{ADMIN_TRAINING_DATA_PREFIX}/upload",
#     "GET_DATASET_DETAILS": f"{ADMIN_TRAINING_DATA_PREFIX}/{{dataset_id}}",
#     "DELETE_DATASET": f"{ADMIN_TRAINING_DATA_PREFIX}/{{dataset_id}}",
#     "DOWNLOAD_DATASET": f"{ADMIN_TRAINING_DATA_PREFIX}/{{dataset_id}}/download",
# }

# --- Admin Prediction Model Management Endpoints (UC12) ---
# ADMIN_MODELS_PREFIX = f"{API_VERSION}/admin/prediction-models"
# ADMIN_MODELS_ENDPOINTS = {
#     "LIST_MODELS": f"{ADMIN_MODELS_PREFIX}",
#     "GET_MODEL_DETAILS": f"{ADMIN_MODELS_PREFIX}/{{model_id}}",
#     "RETRAIN_MODEL": f"{ADMIN_MODELS_PREFIX}/{{model_id}}/retrain",
#     "SET_DEFAULT_MODEL": f"{ADMIN_MODELS_PREFIX}/{{model_id}}/set-default",
#     "DELETE_MODEL": f"{ADMIN_MODELS_PREFIX}/{{model_id}}",
#     "GET_DEFAULT_MODEL": f"{ADMIN_MODELS_PREFIX}/default",
# }

# --- Admin Camera Management Endpoints (UC13 - Simplified) ---
ADMIN_CAMERAS_PREFIX = f"{API_VERSION}/admin/cameras"
ADMIN_CAMERAS_ENDPOINTS = {
    "LIST_CAMERAS": f"{ADMIN_CAMERAS_PREFIX}",
    "ADD_CAMERA": f"{ADMIN_CAMERAS_PREFIX}",
    "GET_CAMERA_DETAILS": f"{ADMIN_CAMERAS_PREFIX}/{{camera_id}}",
    "UPDATE_CAMERA_DETAILS": f"{ADMIN_CAMERAS_PREFIX}/{{camera_id}}",
    "DELETE_CAMERA": f"{ADMIN_CAMERAS_PREFIX}/{{camera_id}}",
    "UPDATE_CAMERA_STATUS": f"{ADMIN_CAMERAS_PREFIX}/{{camera_id}}/status",
    "UPDATE_ROI_CONFIG": f"{ADMIN_CAMERAS_PREFIX}/roi-config", # Simplified ROI config for all cameras
    "CAPTURE_REFERENCE_FRAME": f"{ADMIN_CAMERAS_PREFIX}/capture-reference", # Simplified frame capture
}

# --- Admin Report Processing Endpoints (UC14) ---
ADMIN_PROCESS_REPORTS_PREFIX = f"{API_VERSION}/admin/reports/process"
ADMIN_PROCESS_REPORTS_ENDPOINTS = {
    "LIST_REPORTS_TO_PROCESS": f"{ADMIN_PROCESS_REPORTS_PREFIX}",
    "GET_REPORT_DETAILS_FOR_PROCESSING": f"{ADMIN_PROCESS_REPORTS_PREFIX}/{{report_id}}",
    "UPDATE_REPORT_STATUS": f"{ADMIN_PROCESS_REPORTS_PREFIX}/{{report_id}}/status",
    "REPLY_TO_REPORT_SUBMITTER": f"{ADMIN_PROCESS_REPORTS_PREFIX}/{{report_id}}/reply",
}

# --- User Endpoints Collection ---
USER_ENDPOINTS_COLLECTION = {
    "AUTH": AUTH_ENDPOINTS,
    "MAP": USER_MAP_ENDPOINTS,
    "ROUTES": USER_ROUTES_ENDPOINTS,
    "ACCOUNT": USER_ACCOUNT_ENDPOINTS,
    "REPORTS": USER_REPORTS_ENDPOINTS,
}

# --- Admin Endpoints Collection ---
ADMIN_ENDPOINTS_COLLECTION = {
    "AUTH": AUTH_ENDPOINTS, # Admins also use auth endpoints
    "MAP": ADMIN_MAP_ENDPOINTS,
    "ROUTES": ADMIN_ROUTES_ENDPOINTS,
    "ACCOUNT": ADMIN_ACCOUNT_ENDPOINTS,
    "SUBMIT_REPORTS": ADMIN_SUBMIT_REPORTS_ENDPOINTS, # For admin submitting reports
    # "TRAINING_DATA": ADMIN_TRAINING_DATA_ENDPOINTS, # Commented out
    # "PREDICTION_MODELS": ADMIN_MODELS_ENDPOINTS, # Commented out
    "MANAGE_CAMERAS": ADMIN_CAMERAS_ENDPOINTS,
    "PROCESS_REPORTS": ADMIN_PROCESS_REPORTS_ENDPOINTS, # For admin processing reports
}

# Role-Specific Endpoint Aggregation
ROLE_SPECIFIC_ENDPOINTS = {
    "USER": USER_ENDPOINTS_COLLECTION,
    "ADMIN": ADMIN_ENDPOINTS_COLLECTION,
}

# Export all endpoint collections for potential flat access
ALL_ENDPOINTS = {
    **AUTH_ENDPOINTS,
    **USER_MAP_ENDPOINTS,
    **USER_ROUTES_ENDPOINTS,
    **USER_ACCOUNT_ENDPOINTS,
    **USER_REPORTS_ENDPOINTS,
    **ADMIN_MAP_ENDPOINTS,
    **ADMIN_ROUTES_ENDPOINTS,
    **ADMIN_ACCOUNT_ENDPOINTS,
    **ADMIN_SUBMIT_REPORTS_ENDPOINTS,
    # **ADMIN_TRAINING_DATA_ENDPOINTS, # Commented out
    # **ADMIN_MODELS_ENDPOINTS, # Commented out
    **ADMIN_CAMERAS_ENDPOINTS,
    **ADMIN_PROCESS_REPORTS_ENDPOINTS,
}