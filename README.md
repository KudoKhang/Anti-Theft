# Anti Theft
Ứng dụng giải thuật trừ nền (Background subtraction) để phát hiện chuyển động trong video nền tĩnh, chúng ta có thể xây dựng một ứng dụng đơn giản nhưng khá hữu ích đó là "Camera chống trộm"

<p align="center">
	<img src="https://github.com/KudoKhang/Anti-Theft/blob/main/Sources/antitheft.gif?raw=true" />
</p>

# How it work
Background subtraction là một trong những giải thuật đơn giản và phổ biến trong lĩnh vực Thị giác máy tính (Computer Vision). Thuật toán được sử dựng nhằm xác định đối tượng chuyển động trong video nền tĩnh. Có khá nhiều thuật toán Background subtraction như: Frame Difference, Mean Filter, KNN, Mixture of Gaussian (MOG, MOG2)...
Trong project này mình chọn dùng MOG2. Thuật toán MOG2 khá hiệu quả và được hỗ trợ trực tiếp từ Opencv. Để sử dụng ta có thể gọi trực tiếp ra bằng hàm:

```python
backSub = cv2.createBackgroundSubtractorMOG2()
```

## Bước 1: Xác định khu vực giám sát
```python
top_left, bottom_right = (400, 200), (1000, 700) # ROI
```
## Bước 2: Xử lý từng frame
- Chuyển qua ảnh xám
```python
cap = cv2.VideoCapture('./Sources/video.mp4')
while True:
	_, frame = cap.read()
	fgMask = backSub.apply(frame)
	fgMask = cv2.cvtColor(fgMask, 0) # 0: grayScale
```
- Khử nhiễu
```python
kernel = np.ones((5,5), np.uint8)
fgMask = cv2.erode(fgMask, kernel, iterations=1)
fgMask = cv2.dilate(fgMask, kernel, iterations=1)
fgMask = cv2.GaussianBlur(fgMask, (3,3), 0)
fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
_,fgMask = cv2.threshold(fgMask,130,255,cv2.THRESH_BINARY)
fgMask = cv2.Canny(fgMask, 20, 200)
```
- Xác định vùng chuyển động

```python
contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

## Bước 3: Kiểm tra
- Xác định xem tọa độ vật chuyển động có nằm trong vùng quan sát hay không. Nếu có thì tiến hành phát chuông cảnh báo

```python
for i in range(len(contours)):
	(x, y, w, h) = cv2.boundingRect(contours[i])
	cx = x + w/2
	cy = y + h/2
	logic = top_left[0] < cx < bottom_right[0] and top_left[1] < cy < bottom_right[1]
	area = cv2.contourArea(contours[i])
	if area < 500:
		continue
	if logic:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(frame, "Warning", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
		mixer.music.play()
```

# Usage
Để sử dụng project:

```bash
git clone https://github.com/KudoKhang/Anti-Theft
cd Anti-Theft
python app.py
```

