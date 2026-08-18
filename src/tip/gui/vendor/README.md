# vendor — 제3자 라이브러리

여기 있는 파일은 우리가 쓴 것이 아니라 **three.js r128** 배포본을 그대로 받아 둔 것이다.

| 파일 | 출처 |
|---|---|
| `three.min.js` | https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js |
| `OrbitControls.js` | https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js |
| `CSS2DRenderer.js` | https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/renderers/CSS2DRenderer.js |

라이선스: **MIT** (Copyright 2010-2021 Three.js Authors, SPDX-License-Identifier: MIT).
`three.min.js` 첫 줄에 라이선스 고지가 들어 있다.

## 왜 CDN 을 안 쓰나

예전에는 위 세 주소를 `<script src>` 로 직접 불렀다. 그러면 **인터넷이 막힌 자리에서 3D 가
통째로 죽고**(`THREE is not defined`) 화면에는 아무 설명도 안 나왔다. 계산·지표·프로토콜은
3D 없이도 동작하는데 뷰어만 검은 칸으로 남는 것이다.

저장소 안에 두면 그 일이 없고, 버전이 고정되는 효과도 덤이다. `gui/app.py` 의 `do_GET` 이
`/vendor/<파일명>` 으로 내어 준다(경로 탈출을 막으려고 `basename` 만 쓴다).

## 갱신하려면

세 파일을 같은 버전으로 함께 받아야 한다 — `examples/js/*` 는 코어 버전에 묶여 있다.
r128 은 `THREE.CSS2DRenderer` 를 전역에 붙이는 마지막 계열이라, 올릴 때는 `index.html` 의
`new THREE.CSS2DRenderer()` · `new THREE.OrbitControls()` 호출도 함께 고쳐야 한다.
