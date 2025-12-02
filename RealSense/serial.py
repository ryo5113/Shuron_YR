import pyrealsense2 as rs

# RealSenseカメラを初期化
pipeline = rs.pipeline()
config = rs.config()
pipeline.start(config)

# カメラの情報を取得
pipeline_profile = pipeline.get_active_profile()
device = pipeline_profile.get_device()
serial_number = device.get_info(rs.camera_info.serial_number)

print(f"RealSenseカメラのシリアル番号: {serial_number}")

# RealSenseカメラをシャットダウン
pipeline.stop()