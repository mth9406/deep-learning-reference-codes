'''Affine transformation function
cv2.warpAffine(src, M, dsize, ...)
src: 입력 영상
M: 2X3 affine transformation matrix
dsize: dst size (0,0) 이면 src와 같은 크기
flags: 보간법 ex: cv2.INTER_LINEAR ...
borderMode: cv2.BORDER_CONSTANT
borderValue: borderMode=cv2.BORDER_CONSTANT 일때의 상수값
'''
(1) translate.py : translate (이동.)
M = np.array([[1,0,m], [0,1,n]]) # x, y <- x+m, y+n

(2) shear_transformation.py : 층밀림 변환, x, y축 방향에 대해 따로 정의.
M = np.array([[1,m,0],[n,1,0]]) # x, y <- x+my, nx+y
dst의 크기 지정하는 법 눈여겨 볼 것!

(3) resize.py: 영상의 확대와 축소 예시

(4) flip.py: 대칭 변환

(5) pryDown.py, pryUp.py: up_sampling, down_sampling

* cv2.pryDown(src, dst= None, dstsize= None, borderType= None)
src: 입력 영상
dst: 출력 영상
dstsize: 출력 영상 크기, None인 경우 원본의 가로, 세로 절반 크기로 리턴.
borderType: 가장자리 픽셀 확장 방식
동작 원리: 5x5 가우시안 필터 적용, 짝수 행과 열을 제거.

* cv2.pryUp(src, dst=None, dstsize=None, borderType=None)
src: 입력 영상
dst: 출력 영상
dstsize: 출력 영상 크기, None의 경우 원본의 가로, 세로 2배 크기로 리턴.
borderType: 가장자리 픽셀 확장 방식

(6) rotation.py
* cv2.getRotationMatrix2D(center, angle, scale) -> retval
center: 회전 중심 좌표 (x,y)
angle: 반시계 방향 degree
scale: scale
(사용 예시)
cv2.getRotationMatrix2D(center, angle, sc) 
--> 
rotated = cv2.warpAffine(src, rotMatrix, (0,0), 
                        borderMode= cv2.BORDER_CONSTANT, borderValue=0)

(7) Affine transformation and Perspective transformation
affine_transformation.py
* cv2.getAffineTransform(3 src points, 3 dst points) 
--> cv2.warpAffineTransform()

perspective_transformation.py
* cv2.getPerspectiveTransform(4 src points, 4 dst points) 
--> cv2.warpPerspectiveTransform()

(8) remap.py
* cv2.remap(src, map1, map2, interpolation)
src: 입력 영상
map1: 결과 영상의 x좌표가 참조할 입력 영상의 x 좌표
map2: 결과 영상의 x좌표가 참조할 입력 영상의 y 좌표
