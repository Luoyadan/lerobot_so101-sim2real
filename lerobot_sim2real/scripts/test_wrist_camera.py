import cv2

def variance_of_laplacian(image):
    """计算图像清晰度：值越低越模糊"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 选择正确的相机编号，通常 USB 相机是 0 或 1
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    # 在窗口上显示清晰度指标
    text = f"Sharpness (Laplacian Var): {fm:.2f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("SO101 Wrist Camera Test", frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
