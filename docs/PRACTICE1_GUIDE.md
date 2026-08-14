# Practice 1 실행 가이드 — Two Metallic Spheres & Monopole

> DYMSTEC 교육자료 `[3-3] Sim4Life_Stimulation_A4_202602.pdf` p.46~58 을 순서대로 정리한 것.
> **목적**: 뉴런 솔버가 실제로 도는지 확인 + **`<uuid>_Input.h5` 참조 파일 확보**.
>
> 그 입력파일이 있으면 우리가 만들 섬유 리드필드 → GAF 경로에서
> `RwNeuronInput` 을 어떤 값으로 채워야 하는지 **추측 없이** 맞출 수 있다.
> (현재 `MotorModel.load_model()` 이 `topol()` 에서 AssertionError 로 막혀 있다.)

---

## 0. 먼저 — 프로젝트를 저장할 것

결과 폴더 경로가 정해져야 입력파일을 찾을 수 있다.
`FILE > Save As` → 예: `C:\Users\imrla\Desktop\Taemin\practice1.smash`

→ 결과는 `C:\Users\imrla\Desktop\Taemin\practice1.smash_Results\` 아래에 생긴다.

---

## 1. 모델 (p.46)

`Model` 탭에서:

| 대상 | 방법 | 값 |
|---|---|---|
| Electrode 1 | `Solids > Sphere` | Radius **1 mm**, Translation **(3, 0, 0)** → 이름 `Electrode 1` |
| Electrode 2 | `Solids > Sphere` | Radius **1 mm**, Translation **(-3, 0, 0)** → 이름 `Electrode 2` |
| Tissue | `Solids > Block` | 3axis Size **30 mm**, Translation (0,0,0) **Centered 체크** → 이름 `Tissue` |
| Axon | `Sketch > Spline` | Point 0 **(0, 10, 0)**, Point 1 **(0, -10, 0)** → 이름 `Axon` |

---

## 2. 축삭에 뉴런 모델 할당 (p.55) ★여기서 팝업이 떴던 지점

1. Model 탭에서 **Axon(spline) 선택**
2. Ribbon `Neuron | Axon Model` 클릭
3. 하단 Options 창에서 **Neuron Model = SENN** 선택
4. **`Discretize`** 버튼 클릭
5. Explorer 에 `Axon [SENN Neuron 20.00um]` 이 생기면 성공

> 여기서 *"The Yale NEURON solver was not found…"* 팝업이 뜨면 솔버가 아직 안 깔린 것이다.
> (이미 설치를 마쳤으니 안 떠야 정상.)

---

## 3. EM 시뮬레이션 — Electro Ohmic Quasi-Static

### 3-1. 재질 데이터베이스 (p.48)
Ribbon `Material Database` → **IT'IS LF 5.0** 선택 → 하단 **Active 체크**, **Priority = 1**

### 3-2. 시뮬레이션 생성 (p.48)
`New > EM LF Electro > **Ohmic Quasi-Static**`

> 인체처럼 lossy 한 매질이라 준정자계 해석이 필요하다. 전극이 전도체(인체)를 통해
> 접촉하므로 Electro **Ohmic** Quasi-Static 을 쓴다 (더 안정적이고 빠르다).

### 3-3. Setup (p.49)
`Setup` → **Frequency = 2000 Hz**

### 3-4. Materials (p.49)
Multi-Tree 에서 **Tissue** 를 시뮬레이션 폴더로 **Drag & Drop**
→ Tissue 선택 → Ribbon `Assign Materials` → **Brain [IT'IS LF 5.0]** 지정

### 3-5. Boundary Conditions (p.50)
- 외곽 6면: Dirichlet **0 V** (자동)
- **Electrode 1** → Boundary Conditions 로 Drag&Drop → Dirichlet **+5 V**
- **Electrode 2** → Boundary Conditions 로 Drag&Drop → Dirichlet **−5 V**

### 3-6. Sensors (p.50)
`Field Sensor Settings > Overall Field` 자동 생성 — Record E-field / H-field / Magnetic Vector 체크 확인

### 3-7. Grid (p.51)
- `Auto Grid Update` 활성화, Discretization = Automatic (Default)
- Padding: Manual, Top/Bottom **Max Step (10, 10, 10) mm**
- **Tissue**, **Electrode** 각각 우클릭 → `Move to New > Manual`
  - Tissue Max step **(1, 1, 1) mm**
  - Electrode Max step **(0.1, 0.1, 0.1) mm**

### 3-8. 실행
`Run` → 완료 대기

---

## 4. 뉴런 시뮬레이션 (p.57~58) ★참조 파일이 여기서 나온다

### 4-1. 생성 (p.57)
`New > Neuron > Neuron`

### 4-2. Setup (p.57)
- **Perform Titration** 체크
- Titration Convergence Criterion **1 %**
- Action Potential Detection **Threshold**
- **Threshold for Depolarization = 80 mV**

### 4-3. Neurons (p.58)
Multi-Tree 에서 **`Axon [SENN Neuron 20.00um]`** 을 `Neurons` 로 **Drag & Drop**
→ Automatic Axon Model Settings, Temperature **37 °C**

### 4-4. Sources (p.58) ★EM 결과를 뉴런의 소스로 연결
1. `Sources` 클릭 → Ribbon **`New Settings`**
2. **LF 시뮬레이션의 `Field Sensor Settings`** 를 뉴런의 `Source Settings` 로 **Drag & Drop**
3. 설정값:
   - Source Type: **Simulation Link**
   - Pulse Type: **Monopolar**
   - Initial Time **0.1 ms** · Pulse 1 Amplitude · Pulse 1 Duration **0.1**

### 4-5. 실행
`Run`

---

## 5. 확인할 것 · 알려줄 것

**① 2번 단계에서 팝업이 떴는가** — 안 떴으면 솔버 설치 성공.

**② 뉴런 시뮬이 끝까지 돌았는가**
- 완주 → **T-Neuro 사용 가능**. Action Potential / Titration Factor 결과 확인.
- 오류 → **문구 전체**를 그대로 공유할 것 (`license` / `feature` 가 있으면 라이선스,
  `not found` 면 설치, `network` 면 계산 리소스 문제 — 대응이 각각 다르다).

**③ ★입력파일 경로** — 이게 핵심이다.
```
<프로젝트>.smash_Results\  아래의  <uuid>_Input.h5
```
뉴런 시뮬레이션 쪽 것(둘 이상이면 나중에 생긴 것)의 **전체 경로**를 알려줄 것.
실행이 실패해도 **입력파일만 생성됐으면 충분하다.**

> 찾기 어려우면 Explorer 에서 뉴런 시뮬을 우클릭 → 결과/폴더 열기 계열 메뉴를 쓰거나,
> 그냥 `practice1.smash_Results` 폴더 안의 `*_Input.h5` 를 전부 알려줘도 된다.
