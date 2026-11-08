# 피자 인사이트 2026

피자 프랜차이즈 9개 매장의 2020–2026년 월별 실적(가상 데이터)에서 뽑은 시사점 3가지와, 이를 담은 30초 키네틱 타이포그래피 영상입니다.

- 페이지: https://niugnepppp.github.io/dashboard/
- 영상: [pizza-insights-30s.mp4](pizza-insights-30s.mp4)

## 시사점

1. **성장의 엔진이 바뀌었다** — 주문 수는 2023년 이후 정체, 성장은 객단가(+13%)가 만들었다.
2. **인건비가 이익을 깎는다** — 1인당 인건비 +89%, 매출 대비 인건비 24.4% → 29.2%.
3. **자사앱의 착시** — 자사앱 주문 31%까지 성장했지만 배달앱 비중(43%)과 수수료+쿠폰 비용(매출의 16%)은 그대로.

## 영상 다시 만들기 (macOS)

```bash
pip3 install pillow numpy imageio-ffmpeg
python3 make_video.py --out pizza-insights-30s.mp4
```

음성은 macOS 내장 한국어 음성(Yuna), 음악·효과음은 NumPy로 합성합니다.
