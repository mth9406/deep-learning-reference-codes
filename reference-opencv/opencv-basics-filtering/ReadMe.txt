'''기본적인 2D 필터링

cv2.filter2D(src, ddepth, kernel,  dst=None, anchor= None, delta= None,
            borderType= None) -> dst
src: 입력 영상

ddepth: 출력 영상 데이터 타입
(ex) cv2.CV_8U, cv2.CV_32F, cv2.CV_64F
-1을 지정하면 src와 같은 타입의 dst영상을 생성.

kernel: filter mask 행렬, float.

anchor: 고정점 위치 default:(-1,-1) 이면 필터 중앙을 고정점으로 사용

delta: 추가적으로 더할 값 (bias)

borderType: 가정자리 픽셀 확장 방식
(ex) BORDER_CONSTANT: 000 ABCD 000
     BORDER_REPLICATE: AAA ABCD AAA
     BORDER_REFLICT: CBA ABCD DCB
     BORDER_REFLECT_101 DCB ABCD CBA
'''
[Blurring 효과]
mean_filtering.py: 평균 값 필터, Blurring 효과

blur.py: 평균값 필터 (cv2 사용) cv2.blur()

gaussian_filtering.py: 가우시안 필터링 cv2.GaussianBlur()

[Shapening 효과]
unsharp_mask.py: 흑백 영상에서 shapening 기법 
--> unsharpFilter()
unsharp_mask_color.py: 컬러 영상에서 shapening 기법 
--> unsharpFilterColor()
tip: 웬만하면 image끼리 합, 덧셈은 cv2.addWeighted 사용하기!
# image끼리 합, 덧셈, 뺄셈... 은 float형에서 계산하고
# 최종 dst에서 dtype= uint8 사용.

[Denoise 효과]
median.py : cv2.medianBlur() 사용 예시
          : salt&pepper noise 제거 효과, 영상의 화질이 안 좋아짐 ...
          : 요즘은 잘 사용하는 필터가 아님.

*bilateralFilter.py: edge preserving gaussian blur.
                   : cv2.bilateralFilter() 사용 예시