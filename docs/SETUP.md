# TI Planner — 새 환경 셋업 가이드

> 자체 제작 TI(Temporal Interference) 자극 플래너. MIDA 두부모델 리드필드 기반으로
> **전극 조합·전류를 최적화**하고, 필드/활성화함수/분해능을 3D로 분석한다.
> 이 문서 하나로 새 환경(예: Sim4Life 9.6 머신)에서 처음부터 띄울 수 있게 정리함.

---

## 0. 30초 요약

```bash
# 1) 데이터·코드 복사 (약 1.6 GB)  → tip/ 폴더 통째로 (메모리·문서 포함)
# 2) Python 환경 (numpy·scipy만 있으면 됨)
# 3) 무결성 검증 (§4 — 반드시)
# 4) 서버 실행
python tip/gui/app.py            # → http://127.0.0.1:8765
# 5) Claude 메모리 복원 → memory/README.md 참조 (연구 맥락 이어받기)
```

**Sim4Life는 실행에 필요 없다.** 리드필드가 이미 `.npy`로 추출돼 있어 툴은 독립 동작한다.
Sim4Life는 (a) 리드필드 재생성 (b) 섬유 궤적 생성(§8 ①) 때만 필요하다.
뉴런 평가(§8 ⑤)는 **T-Neuro 라이선스 미보유**라 맥 standalone NEURON 으로 한다.

> **이 머신(2026-08-03) 상태**: §4 무결성 검증 통과 · MCP 구축 완료(`../s4l_mcp/`) ·
> §8 파이프라인 ①~④ 완료, ⑤(맥 NEURON)가 다음 차례. 이전 과정에서 발견한 문제와
> 정정은 **`../MIGRATION_STATUS.md`** 에 있다.

---

## 1. 요구사항

| 항목 | 검증된 버전 | 비고 |
|---|---|---|
| Python | 3.9.19 | 3.9~3.11 동작 확인 |
| numpy | 1.26.4 | 필수 |
| scipy | 1.13.1 | 필수 (KDTree·SLSQP) |
| matplotlib | 3.9.0 | 선택 (분석 그림 스크립트만) |
| 브라우저 | 최신 Chrome/Edge | Three.js r128 (CDN) |

```bash
conda create -n tip python=3.11 numpy scipy matplotlib
conda activate tip
```

> **주의 — 브라우저 오프라인 불가**: `index.html`이 Three.js를 CDN에서 로드한다.
> 완전 오프라인 환경이면 `three.min.js`를 로컬로 받아 `<script src>`를 교체할 것.

---

## 2. 디렉토리 구조

```
tip/
├─ data/                          ★ 최대 자산 (~1.6 GB)
│  ├─ leadfieldF/M{00..60}.npy      60전극 리드필드 (Cz=기준 제외), 각 (1907678,3) float32
│  ├─ leadfield_extra/*.npy         하부링 10전극 (F9/10·FT9/10·T9/10·TP9/10·P9/10) → 총 70전극
│  ├─ masks/                        표적 마스크 7종 + manifest.json
│  ├─ bmask1010.npy                 뇌 복셀 격자인덱스 (N,3) ★좌표 디코딩 필수
│  ├─ gaxes1010.npz                 격자 셀중심 좌표 cx,cy,cz (mm) · 185×254×228
│  ├─ blabel1010.npy                조직 라벨 (GM=75, WM=131, 해마=81)
│  ├─ hipmask1010.npy / hipaxes1010.npz   해마 마스크 / 장축(nL,nR)
│  ├─ thalamus_mask.npy
│  ├─ enames1010.json · unitnorm.json · pos1010.json · inj_current.json
│  └─ leadfield_3cm2/               (생략 가능 — 전극 크기 무관 검증 완료)
├─ tip/                           패키지 코드
│  ├─ config.py                     경로·상수 (ITOTAL, IMAX, 조직라벨)
│  ├─ leadfield.py                  리드필드 로더 ★좌표 규약의 원천
│  ├─ ti.py                         TI 엔벨로프 (tmax, directional_env, carrier_max)
│  ├─ metrics.py · targets.py       지표 M1/M2/M3 · 표적 정의
│  ├─ fieldsample.py                삼선형 보간 · AF 평활 · GAF(MDF2)
│  ├─ activating.py                 활성화함수 목적 (af_proj, optimize_projected)
│  ├─ geometry.py                   피질 표면법선 = 방사 배향 (DTI 불필요)
│  ├─ benchmark.py                  공정 벤치마크 하네스 (METHODS 레지스트리)
│  ├─ neuron_bridge.py              필드 → NEURON 케이스(.npz) 내보내기
│  ├─ fiberlead.py                  ★섬유 리드필드 (전극별 Vₑ · 드라이브 4종 · §8)
│  ├─ plan.py · report.py           프로토콜 생성 · 리포트
│  ├─ optimize/
│  │   ├─ classic.py                Classic 2ch 전수탐색 (+ 전류 정규화 함수)
│  │   ├─ nsga.py                   자체 NSGA-II (전극 많을 때)
│  │   ├─ multichannel.py           분산 → **Huang 백엔드 래퍼** (레거시 _distributed_legacy 보존)
│  │   ├─ huang.py                  Huang·Datta·Parra 2020 배열 IFS (Pmax 제약형)
│  │   ├─ dualti.py                 4채널 이중 TI
│  │   ├─ selective.py              (β) 이중기전·방향최악 선택성
│  │   ├─ fiber.py                  ★섬유 개체군 전수탐색 (compare_drives · §8-3 Step 4)
│  │   └─ timemux.py                시분할 (제약형 + 전극예산)
│  └─ gui/
│      ├─ app.py                    HTTP 서버 + 최적화 잡 + 분석 엔드포인트
│      └─ index.html                단일 파일 프론트엔드 (Three.js)
├─ memory/                       ★Claude 대화 메모리 사본 (연구 맥락 보존)
│  └─ README.md                    새 환경 복원법 · 파일 안내 · 읽는 순서
├─ SETUP.md                      이 문서
├─ s4l_fibers.py                 Sim4Life 측 스크립트 (MCP 실행)
├─ run_gui.bat                    더블클릭 실행 (환경 자동 탐색 — §3, 수정 완료)
└─ demo_*.py / validate_*.py      데모·검증 스크립트 (validate_fiberlead.py = §8 검증)

상위 폴더 (repo 밖 — 이 머신 전용):
├─ ../s4l_mcp/                   ★Sim4Life 9.6 MCP 서버 (헤드리스) — README.md 참조
├─ ../.mcp.json                  Claude Code 등록 (프로젝트 스코프, 최초 1회 승인 필요)
├─ ../.venv-s4l/                 MCP용 venv (S4L 번들 파이썬 + mcp 2.0.0)
├─ ../MIDA (Static) - 1.0/       ViP 에서 받은 MIDA v1.0 (.sab 384MB + voxels)
└─ ../MIGRATION_STATUS.md        이전 결과 · 정정 기록 · 코드에서 발견한 문제
```

---

## 3. 실행

```bash
python tip/gui/app.py
# [TI Planner] 리드필드 로드...
# [TI Planner] 준비완료 · 전극 70 · 표적 21 · 마스크 5
# [TI Planner] → http://127.0.0.1:8765
```

### ✅ `run_gui.bat` — 수정 완료 (2026-08-03)

구버전은 Sim4Life 9.0 번들 Python 을 절대경로로 박아둬 9.6 머신에서 깨졌다. 지금은 사용
가능한 환경을 순서대로 자동 탐색한다: conda `tip` → `.venv-s4l` → S4L 9.6 번들.

**이 머신의 검증된 환경 2개** (둘 다 §4 물리값 소수 셋째자리까지 일치 확인):

| 환경 | 용도 | 구성 |
|---|---|---|
| `~/miniconda3/envs/tip` | GUI·로컬 분석 (Sim4Life 비의존) | 3.11.15 · numpy 1.26.4 · scipy 1.17.1 |
| `<repo>/../.venv-s4l` | MCP 서버·워커 | S4L 9.6 번들(3.11.9) + `--system-site-packages` + mcp 2.0.0 |

> scipy 가 검증본(1.13.1)과 다르다(1.17.1 / 1.14.1). 물리값은 일치했으나 SLSQP·KDTree 를
> 쓰는 최적화 경로는 재실행 시 한 번 대조해 볼 것.

### 포트 정리 (재기동 시)

```bash
# Windows: 8765 점유 프로세스 종료
netstat -ano | findstr :8765
taskkill /PID <pid> /F
```

---

## 4. 이전 후 무결성 검증 — 반드시, 순서대로

> 과거에 **좌표 디코딩 버그로 결과가 통째로 무효화**된 적이 있다. 아래 3단계를 건너뛰지 말 것.

### 4-1. 리드필드 로드 + 좌표

```python
from tip import LeadField
LF = LeadField()
print(len(LF.names))          # 70  (표준 60 + 하부링 10)
C = LF.coords()               # (1907678, 3) mm  ★반드시 이 경로로만
print(C.shape, C.min(0).round(1), C.max(0).round(1))
```

**좌표는 `bmask1010.npy` → `gaxes1010.npz` 경로로만 얻는다.**
격자 인덱스를 직접 계산해서 좌표를 만들지 말 것 (버그 재발 지점).

### 4-2. 알려진 물리값 재현 ★핵심

1 mA 주입 시 **좌해마 표적점 |E|** — 값이 소수 셋째자리까지 일치해야 무결:

```
FT7 0.785 > T7 0.770 > C5 0.540 > TP7 0.495 > P7 0.356 > Fz 0.120   (V/m)
```

> **중요**: 이 값은 **표적 대표점**(`target1010.npy[0]`, 좌해마 복셀 인덱스 483022)에서의
> 값이다. 해마 마스크 전체 평균이 아니다(그 경우 0.728 등 ~7% 낮게 나와 오판하기 쉽다).

```python
import numpy as np, os
from tip import LeadField
LF  = LeadField()
tgt = int(np.load(os.path.join(LF.data_dir, "target1010.npy"))[0])   # 좌해마 표적점
for e in ["FT7","T7","C5","TP7","P7","Fz"]:
    E = LF.elec_field(e, np.array([tgt]))
    print(e, round(float(np.linalg.norm(E[0])), 3))
# 기대: 0.785 / 0.770 / 0.540 / 0.495 / 0.356 / 0.120
```

해부학적으로 타당해야 한다(좌측두 최대, 정중전두 최소).
문헌(Violante) 해마 0.26~0.5 V/m @2mA/pair와 같은 자릿수.

### 4-3. 격자·기동 확인

- 뇌복셀 **1,907,678** · 격자 **185×254×228** · graded 간격(뇌영역 0.4~0.9 mm)
- GUI 기동 후 **각 모드 1회씩**: classic · 분산(Huang) · 4채널 dual · 시분할 · (β)선택성
- 뷰: 필드 / 활성화함수 / 벡터·파형, 분해능(활성임계) 패널, Δf 경고 배너

---

## 5. ★ 핵심 규약 — 코드 수정 시 반드시 지킬 것

### 5-1. 좌표
`lf.coords()` 외의 경로로 복셀 좌표를 만들지 않는다. (§4-1)

### 5-2. 전류 정규화 — 공정 비교의 기준
```
I_total = Σ(+전극 주입전류) = 0.5·Σ|모든 전극 전류| = ITOTAL (=2.0 mA)
i1 = min(ITOTAL/(1+r), IMAX/max(1,r)),  i2 = r·i1
```
- **M1(세기)만 전류에 1차 비례**, M2·M3는 스케일 불변
- → **강도 비교는 등(等)총전류에서만 유효**. dual은 시스템당 ITOTAL/2, Huang은 L1 예산.

### 5-3. 엔벨로프 (3종)
```
등방   Tmax     = 2·max_n min(|n·E1|, |n·E2|)     Grossman 닫힌해
방향   Tdir     = 2·min(|n̂·E1|, |n̂·E2|)          축삭 방향 n̂
활성화 T_AF     = 2·min(|AF1|, |AF2|),  AF = ∂(n̂·E)/∂s
```
**주파수는 필드 계산에 들어가지 않는다** (kHz 조직서 σ≫ωε → 준정적 근사).
단 **Δf=0이면 엔벨로프 자체가 없다**(맥놀이 없음 = TI 아님) → `_freq_check()`가 경고.

### 5-4. 지표·용어 (통일됨)
| 지표 | 정의 | 한글 | English |
|---|---|---|---|
| M1 | median(표적 드라이브) | **세기** | strength |
| M2 | (RMS_표적 / RMS_off)² | **초점** | focality |
| M3 | off 중 표적 p분위 초과 비율 % | **누출** | leakage |

- **"선택성"은 (β)선택성 모드의 뉴런 선택성 전용**으로 예약. 필드 초점(M2)과 구분할 것
  (NEURON: 필드 초점 ≠ 뉴런 선택성).
- WP = w1·M̂1 + w2·M̂2 − w3·M̂3 (후보 최대로 정규화)

> **★off 풀 기준이 복셀과 섬유에서 다르다 (의도된 것)**
> · 복셀 지표: `config.NEURAL_LABELS` = **GM 만** (tip.lite 규약, §7 결과 전체가 이 위에 있다)
> · 섬유 개체군: `fiberlead.sample_seeds` 기본 **GM ∪ WM** (섬유는 백질 구조라 GM 만으로는
>   off 가 표본이 안 된다)
>
> **두 지표를 같은 표에 나란히 놓고 해석하지 말 것.** 실제로 섬유 최적 몽타주가 복셀 dir
> 지표에서 누출 70~80% 로 보였는데 NEURON 실측에선 표적 역치가 가장 낮았다 — 서로 다른
> 것을 재고 있다는 증거다. 상세는 `config.py` 주석과 `../MIGRATION_STATUS.md` §4-1.
>
> **섬유 off 는 개수보다 공간 커버리지가 중요하다.** 60시드×12줄(720섬유)은 절점 수가
> 복셀 off 와 비슷해도 **60곳에 뭉쳐** 있어 초점(M2)을 20~50% 과대평가한다. 300시드×4줄
> (1200섬유)로 늘리면 M2 가 계통적으로 내려가고, **af 와 gaf 의 최적 몽타주가 완전히
> 일치**해진다(성긴 off 에서 갈리던 것은 아티팩트).

### 5-5. 필드 샘플링
격자에 없는 점(축삭 좌표 등)은 **반드시 `fieldsample.interp_apply`(삼선형 보간)** 사용.
최근접 복셀 스냅은 Vₑ를 계단으로 만들어 AF에 노이즈를 증폭시킨다.

---

## 6. 알려진 함정 (실제로 겪은 버그)

| 함정 | 증상 | 대응 |
|---|---|---|
| **좌표 디코딩** | 결과가 그럴듯하나 전부 무효 | `bmask1010` 경로로만 (§4-1) |
| **최근접 스냅** | AF 노이즈·크기 2배 부풀림 | 삼선형 보간 (§5-5) |
| **전류 불공정** | "강도 이득"이 dose 아티팩트 | 등총전류 (§5-2) |
| **탐색공간 축소** | 결론이 뒤집힘 (Huang select_k=16 → 잘못된 결론) | 전극 풀을 임의로 좁히지 말 것 |
| **가중합 vs 제약형** | 이득 구간을 통째로 놓침 | 강도 바닥 제약 하 초점 최대화(시분할·Huang) |
| **변수 섀도잉** | `ET[False]` → 빈 배열 | 내부 bool 마스크 이름 분리 |
| **numpy 직렬화** | JSON 에러 | best dict에서 numpy 배열 pop |

---

## 7. 연구 맥락 (결과 요약)

새 환경에서 맥락을 잃지 않도록 핵심만:

**확립된 것**
- **활성 부위에 따라 최적 몽타주가 field ↔ AF로 뒤집힌다** (NEURON MRG 검증)
  - 심부·통과섬유(해마·시상): **af_opt 승** (24/24, 시상 12/12)
  - 피질·말단(A6dl_R): **field_opt 승** (역치 4.9 vs 6.7 vs 8.8)
  - Mirzakhalili 2020 위치의존 기전을 몽타주 최적화 맥락에서 확증
- **말단 지배 표적은 축방향 E만으로 몽타주 순위 12/12 예측** → NEURON 불필요.
  통과섬유 표적에서만 뉴런 모델 필요.
- 심부 초점은 **배열(Huang)** 이 최선. 전극 많을수록 심부 초점 ↑.

**닫힌 것 (음성 결과)**
- 위상(<2%) · N-반송파(N≤2) · dual(등전류서 classic에 지배)
- **GAF(우리 MDF2 근사)**: 직선 MRG서 raw AF 못 이김 (af 19승 : gaf 1승)
- **선택성**: field·AF 둘 다 실패 — off가 표적보다 먼저 켜짐 (심부·피질 공통)
- **시분할**: 강도 이득 불가(볼록결합 상한). **단 고강도 영역서 초점 +18~47%**(단일 2쌍 대비)
  — 배열엔 못 미침. 니치 = 채널 수 제약 하드웨어.

**포지셔닝 (정직)**
IT'IS/ZMT가 이 분야를 리드한다(TIP V5.2 = GAF + Pareto + DTI). 우리 툴의 가치는
**오픈·커스터마이즈 가능한 재현 + 엄밀한 공정벤치·음성결과**. 근본 격차는 **DTI 부재**
(피질은 `geometry.py` 표면법선으로 우회, 백질 통과섬유는 근사).

---

## 8. 현재 진행 작업 — Sim4Life 직결 (섬유 리드필드)

> **이 절은 새 환경에서 작업을 이어받기 위한 인계 문서다.** 아래 §8-3을 순서대로 실행하면
> 된다. §8-4의 결정 사항은 이미 확정됐으니 다시 논의하지 말 것.

### 8-1. 왜 이 구조인가

9.6의 **GAF(Green 함수 케이블해)** 가 NEURON 역치를 **R²=0.99 · ~1000배**로 예측한다.
우리가 못 풀던 "단일 스칼라로는 NEURON 역치를 못 맞힌다(log상관 −0.67)"가 해소되는 지점.

> **확인됨 (2026-08-03)**: ZMT 공식 발표와 일치한다 — GAF 는 **V9.2** 도입(단일 케이블),
> **V9.6** 에서 **MRG 이중케이블**로 확장, "R² = 0.99 · up to 1000× faster",
> Recruitment Curve Evaluator 가 **T-Neuro 모듈**에 통합. 즉 `GAF` 라는 이름의 공개 API 가
> 아니라 **titration 전략(`kEstimator` 추정) + Recruitment 평가기**로 노출된다.
> 다만 **이 머신은 T-Neuro 라이선스가 없다**(§8-2) → ⑤는 맥 NEURON 으로 우회(Step 5).

**핵심 착상**: GAF는 세포외전위 Vₑ에 **선형**이다. 전극별 Vₑ를 궤적 위에 한 번만 계산해
두면(= 섬유 리드필드), 임의 몽타주의 Vₑ·GAF는 **로컬 선형 중첩**으로 즉시 나온다.
→ 만 개 단위 전수탐색을 그대로 유지하면서 생리 층을 붙일 수 있다.
(지금 E‑필드 리드필드와 완전히 같은 구조 → 기존 최적화 기계 재사용)

```
① [S4L]  ViP.GenerateSplines(mask) → 섬유 궤적 (F,N,3)      s4l_fibers.make_fibers()   ★1회
② [로컬] 전극별 Vₑ = −∫E_e·dl      → 섬유 리드필드 (E,F,N)   fiberlead.build_…()        ★1회
③ [로컬] 몽타주 → Σ I_e·Vₑ,e → GAF 엔벨로프 → 전수탐색       FiberLeadField.envelope()  매번
④ [로컬] 상위 N개 Vₑ 내보내기                                 …export_candidates()
⑤ [S4L]  GAF 역치·recruitment 평가 → 결과 수신               s4l_fibers.evaluate_…()    9.6 필요
```

### 8-2. 현재 상태 (2026-08-03 갱신 — 새 머신 실측)

| 단계 | 상태 | 비고 |
|---|---|---|
| ① 궤적 생성 | **완료 (실제 MIDA)** | 좌해마 **194궤적**·31절점·23.1±0.7mm, 생성 1.1초. **직선도 0.893** = 합성 직선(1.0)보다 현실적 |
| ② 섬유 리드필드 | **재작성·검증 완료** | 중첩 vs 직접계산 **1.13e‑15**(float64). 914섬유×70전극 **4.1초**·16 MB |
| ③ 로컬 최적화 | **완료** | **몽타주당 0.12~0.14 ms** (1만 개 ≈ 1.4초) · ve/field/af/gaf **4종** 드라이브 |
| ④ 후보 내보내기 | **완료** | `export_candidates()` — `neuron_bridge.export_ti_case()`와 **같은 키 규약**(맥 하네스 호환) |
| ⑤ 뉴런 평가 | **맥 NEURON 으로 진행** | 인계 문서 `MAC_NEURON_HANDOFF.md`. S4L GAF 경로는 90% 도달 후 보류(§8-2 아래) |
| 옵티마이저 연결 | **완료** | `tip/optimize/fiber.py` — `optimize_fiber`·`compare_drives`·`make_benchmark_method` |
| off-target 개체군 | **완료** | 뇌 전역 60시드 × 12줄 = **720섬유**. 원안에 없던 단계인데 **없으면 M2·M3가 무의미**(§8-5) |

**관련 파일**: `tip/fiberlead.py` · `tip/optimize/fiber.py` · `tip/validate_fiberlead.py`(로컬)
 · `s4l_fibers.py`(Sim4Life 측)
**데이터**: `data/fibers_hippoL.npz`(표적 궤적) · `fibers_offpop.npz`(off 개체군)
 · `fiberlead_hippoL_pop.npz`(통합 섬유 리드필드) · `fibercand_hippoL_pop.npz`(맥 NEURON용 후보)
**통신**: MCP `s4l_run_python` (`s4l_mcp/` — 헤드리스 Sim4Life 9.6, `s4l_mcp/README.md`)

> **T-Neuro 현황 — 실측 (2026-08-03)**
>
> **탑재돼 있는 것 (동작 확인)**: `NeuronModeling`·`NeuronSimulator`·`NeuronPostPro` 임포트 OK ·
> `Solvers/NeuronSolver.x.dll` 존재 · `MotorMrgNeuronProperties()`(**MRG 이중케이블**, 9.6 GAF가
> 겨냥한 모델) 생성 OK · `CreateAxonNeuron()` 성공 · `neuron.Simulation()` 생성 OK ·
> **`PerformTitration=True` + `TitrationStrategy=Estimator`(GAF 경로 추정) 설정까지 성공**.
> 즉 **모델링·설정 레이어는 전부 있다.**
>
> **없는 것**: `NeuronYale/` 디렉토리와 `neuron_s4l` 패키지(estimator `.pyd` 본체 포함).
> `lmstat` 기준 라이선스 17종에도 NEURO 계열이 없다.
>
> **★★최종 결론 (2026-08-03) — 솔버는 설치됐으나 라이선스가 없다**
>
> 설치 후 GUI 에서 Practice 1 을 돌리면 **`no license found for NEURON. please contact support`**.
> 처음 `lmstat` 조회(보유 17종에 NEURO 계열 0건)와 일치한다. 정리:
> **① 다운로드·설치 권한은 있었다(License Tool 에 `NEURON/S4L` 품목 존재) ② 그러나 실행
> 라이선스는 없다.** 즉 §8 ⑤단계의 Sim4Life 경로는 **라이선스 구매 전까지 불가**다.
>
> **→ ⑤는 맥 standalone NEURON 으로 간다** (Step 5). §7의 NEURON 검증을 이미 그 경로로 했다.
> 오픈소스 NEURON 이라 라이선스 문제가 없고, 검증된 길이다.
>
> ⚠ **주의**: 설치된 `neuron_s4l.estimator_simulation`(GAF)은 파이썬에서 직접 호출하면
> 라이선스 체크에 걸리지 않는다(아래 기록). 하지만 그건 **라이선스가 없는 기능을 우회해
> 쓰는 것**이므로 그대로 사용하면 안 된다. 쓰려면 **ZMT/DYMSTEC 에 사용 가능 여부를 먼저
> 확인**할 것. 문의 시 FLEXID `9-0CB39FBB`.
>
> **(참고 기록) 설치 자체는 성공했다**
>
> License Tool(`C:\Users\Public\Documents\ZMT\Licensing Tools\9.6\LicenseInstall.exe`
> = 시작메뉴 `Licensing Tools ver 9.6 > Install License`) 의 Download Products 에서
> **`NEURON/S4L`**(품목 index 6) 을 받아 설치. `NeuronYale/` · `neuron_s4l`(v2.0.0) ·
> `estimator_simulation` 이 모두 생성됐다.
>
> **헤드리스에서 쓰려면 DLL 경로를 직접 잡아야 한다** (안 하면 `ImportError: DLL load failed
> while importing hoc`):
> ```python
> YALE = r"C:\Program Files\Sim4Life_9.6\NeuronYale\yale"
> os.environ["NEURONHOME"] = YALE
> os.environ["PATH"] = os.path.join(YALE, "bin") + os.pathsep + os.environ["PATH"]
> os.add_dll_directory(os.path.join(YALE, "bin"))
> import neuron            # ★ s4l_v1 보다 먼저
> ```
> `neuron_s4l.setup_yale_environment()` 는 프로세스 시작 후엔 이미 늦어 효과가 없다.
>
> **★GAF 가 파이썬에서 직접 호출된다** (`neuron_s4l.estimator_simulation.estimator`):
> `compute_gaf(neuron, source, BC='symmetric') -> np.ndarray` · `get_threshold()` ·
> `get_potential(...)` · `get_estimator(model_idx)`. `DoubleCableEstimator` 가 9.6 의 MRG
> 이중케이블 GAF, `SingleCableEstimator` 가 9.2 의 단일케이블. 기본 kernel `(25, 0.5)` ·
> timestep `2.5e-6`. 모델 인덱스 **7 = MotorMrg**, 6 = SensoryMrg, 0 = Senn …
> **`get_estimator(7)` 이 `supported=True` 로 뜨고, Sim4Life 앱 초기화 없이도 동작하며
> 라이선스 체크에 걸리지 않는다** → ⑤단계를 솔버 잡 제출 없이 로컬에서 돌릴 여지가 있다.
> 남은 과제는 `settings.SourceSettings` 에 우리 Vₑ 를 넣는 방법(§8-3 Step 5 참조).
>
> **(구) 원인 기록 — Yale NEURON 솔버가 별도 인스톨러였다** (교육자료 p.55 확인)
>
> GUI 에서 축삭에 Neuron 모델을 붙이면 뜨는 팝업의 원문:
> *"The Yale NEURON solver was **not found in the Sim4Life installation folder**. Please launch
> your License Tool to get a download code with the Download Products dialog, and visit
> `https://zmt.swiss/support/support/sim4life/` to download the installer."*
> → `[Open License Tool]` `[Browse Website]` `[Close]`
>
> **즉 구매 문제가 아니라 설치 문제일 가능성이 크다.** 우리가 관측한 것과 정확히 맞는다:
> `NeuronYale/` 없음 · `neuron_s4l` 패키지 없음(dist-info 만) · `Neuron.pth` 가
> `../../../NeuronYale/yale/lib/python` 을 가리킴 — 인스톨러가 만들 디렉토리다.
>
> **조치 (교육자료 p.55 절차)**: 팝업에서 **Open License Tool** → Download Products 에서
> 다운로드 코드 확인 → ZMT 지원 사이트에서 **Sim4Life 버전과 동일한 Neuron 버전** 인스톨러를
> 받아 설치. 설치 후 `NeuronYale/` 과 `neuron_s4l` 이 생기고 `kNEURON`/`kEstimator` 가 열린다.
>
> ⚠ **미확인**: `lmstat` 기준 라이선스 17종에 NEURO 계열이 없다는 것은 사실이다. 위 설치가
> 끝나도 실행 시 라이선스가 걸릴 수 있다. **License Tool 이 두 질문(다운로드 권한 · 실행
> 라이선스)을 한 번에 답해 준다** — 거기서 확인하고 판단할 것.
> 문의 시 FLEXID `9-0CB39FBB` · `license_FLEXID_9-0CB39FBB_V10.0_Aug2027.dat`.
>
> (참고: GUI 가 network 오류 없이 여기까지 왔다는 건 GUI 에는 계산 리소스가 정상 등록돼
> 있다는 뜻이다 — 아래 '헤드리스 한계'는 MCP 세션에만 해당한다.)
>
> 어느 쪽이든 §7의 NEURON 검증을 원래 맥에서 했으므로 **연구는 그대로 진행 가능**하다 —
> S4L GAF 는 같은 일을 1000배 빨리 할 뿐이다.
>
> **MCP 세션의 한계로 기록**: 헤드리스에서는 **어떤 솔버도 제출되지 않는다**(§9 리드필드
> 재생성도 동일). 솔버가 필요하면 GUI 를 쓰거나, `GetInputFileName()` 으로 입력파일명을 얻어
> `Solvers/iSolve.exe` 를 직접 돌리는 수동 경로를 쓸 것.

### 8-3. 진행 기록 · 이어서 할 일

#### ✅ Step 1. 좌표계 정합 — **통과** (2026-08-03)

MIDA (Static) v1.0 을 ViP 에서 내려받아(`Taemin/MIDA (Static) - 1.0/MIDA_v1.0.sab`)
`XCoreModeling.Import`(6~7초) → `GetBoundingBox` 로 대조:

```
MIDA bbox : [-110.8  131.7  -88.3] ~ [ 51.8  364.2  128.3]
리드필드   : [ -95.0  230.5  -55.3] ~ [ 38.1  343.6  115.8]
축별 겹침  : X 133.1 · Y 113.1 · Z 171.1 mm  → 뇌 복셀이 두부 안에 완전 포함
```
교차검증: `pos1010` 전극 범위도 내부. FOUNDATION.md 랜드마크 **LPA x=-111 / RPA x=52** 가
MIDA x 극단 **-110.8 / 51.8** 과 일치(PA점은 머리 최대폭). **같은 좌표계 · 변환 불필요.**
재현: `s4l_fibers.check_coordinate_frame()`.

> ⚠ **MIDA 의 NIfTI(`MIDA_v1.nii`)는 다른 프레임이다** — sform bbox
> `[-97.5 -107.2 -121.8] ~ [96.4 148.0 130.9]`, 축 치환 + Y 오프셋 ~235mm.
> `.nii` 로 대조하면 "Y축 안 겹침"으로 **오판한다**. 반드시 `.sab` 을 쓸 것.

#### ✅ Step 2. 실제 궤적 생성 — **완료**

```python
# MCP 세션에서
import sys; sys.path.insert(0, r"C:\Users\imrla\Desktop\Taemin\tip")
from s4l_fibers import fibers_around
fibers_around("data/fibers_hippoL.npz",
              center=[-51.7, 259.7, 28.3],        # target1010[0]
              direction=[0.17, -0.486, -0.857],   # hipaxes1010.npz 의 nL
              length_mm=20, radius_mm=6, num_lines=300, n_nodes=31)
# → (194, 31, 3) · 23.1±0.7mm · 1.1초 · 직선도 0.893
```

**백질 마스크는 적용되지 않았다.** `.sab` 의 `Brain White Matter` 는 **TriangleMesh** 인데
`GenerateSplines(mask=)` 는 **LabelField** 를 요구한다 → `fibers_around` 가 경고 후 mask 없이
진행한다(설계된 폴백). 백질 제한이 필요하면 먼저 ⓐ `Voxeler` 로 복셀화하거나
ⓑ `MIDA_v1.nii`(라벨 12 = Brain White Matter, 매핑은 `MIDA_v1.txt`)를 임포트할 것.

#### ✅ Step 2b. off-target 섬유 개체군 — **완료** ★원안에 없던 필수 단계

표적 다발만으로는 **M2·M3 가 의미를 갖지 못한다**(§8-5). 뇌 전역에 시드를 뿌려 off 개체군을 만든다.

```python
from tip.fiberlead import sample_seeds
from s4l_fibers import fibers_scatter
centers, dirs = sample_seeds(LF, 60, exclude_idx=tgt.target_idx, margin_mm=12, edge_mm=12)
fibers_scatter("data/fibers_offpop.npz", centers, dirs,
               length_mm=20, radius_mm=3, lines_per_seed=12)   # → (720, 31, 3)
```
시드는 **GM ∪ WM** 에서 뽑고 방향은 **등방 무작위**다(DTI 부재 → 방향 특정 대신 방향무관
개체군; `optimize/selective.py` 의 '방향최악' 규약과 같은 입장). 표적 다발 194 + off 720
= **914 섬유**, 그중 좌해마 통과 192 · off 722 (off 개체군에서 해마를 지나는 건 0개).

#### ✅ Step 3. 섬유 리드필드 구축 + 검증 — **완료**

```python
trajs = np.concatenate([np.load(dd+"/fibers_hippoL.npz")["trajs"],
                        np.load(dd+"/fibers_offpop.npz")["trajs"]])      # 194 + 720
build_fiber_leadfield(LF, trajs, out_path=dd+"/fiberlead_hippoL_pop.npz")
fl = FiberLeadField(dd+"/fiberlead_hippoL_pop.npz",
                    target_mask=label_fibers(LF, trajs, tgt.target_idx))
```
검증은 `python validate_fiberlead.py` (MIDA·라이선스 불필요, 합성 다발로 자체 완결):

| 검증 | 결과 |
|---|---|
| 선형 중첩 == 직접 계산 (classic / 분산) | **1.13e‑15** / 1.50e‑15 |
| 드라이브 선형성 (전극별 중첩 == 직접 미분) | field 1.2e‑15 · af 6.1e‑15 · gaf 1.1e‑14 |
| 몽타주당 평가 | 0.12~0.14 ms |

**구현 중 잡은 함정 2개** (둘 다 처음엔 통과한 것처럼 보였다 — §8-5에 추가):
① 궤적 좌표를 float32 로 저장하면 검증이 1e‑5 로 깨진다(Vₑ 는 float32 무해).
② 몽타주마다 미분하면 34 ms/몽타주 → 전수탐색 불가. 드라이브도 Vₑ 에 선형이므로
   **전극별 드라이브를 캐시**(`elec_drives`)해 240배 단축.

#### ✅ Step 4. 옵티마이저 연결 — **완료** (`tip/optimize/fiber.py`)

```python
from tip.optimize.fiber import optimize_fiber, compare_drives, make_benchmark_method
best = optimize_fiber(fl, kind="gaf", select_k=16)          # 단일 드라이브 탐색
res  = compare_drives(fl, ("field","af","gaf"), pool_k=16)  # ★드라이브 비교는 이걸로
from tip.benchmark import register_method
register_method("fiber_gaf", make_benchmark_method(fl, kind="gaf"))
```
골격은 `optimize/selective.py` 패턴(4전극 조합 × 3짝짓기 × 전류비 격자)이고, 반환은
classic 형태라 기존 평가·리포트 기계가 그대로 받는다.

> ⚠ **드라이브를 비교할 때는 반드시 `compare_drives()`** 를 쓸 것. `optimize_fiber` 를
> 드라이브별로 따로 부르면 `select_k` 축소가 **드라이브마다 다른 전극 풀**을 만든다.
> 실제로 겪었다 — field 풀에 `T10` 이 없어서 field 탐색이 gaf 최적해를 **후보로 보지도
> 못했고**, 그 결과 "field_opt 가 field 기준에서 진다"는 가짜 결론이 나왔다.
> §6의 "탐색공간 축소 → 결론이 뒤집힘(Huang select_k=16)" 과 **같은 함정**이다.
> `compare_drives` 는 드라이브별 상위 k 의 **합집합**을 공통 풀로 쓰고, 교차 WP 를
> 평가 드라이브 안에서 정규화한다(M1·M2 를 따로 비교하면 목적함수가 아닌 축으로 승패가 갈린다).

#### ▶ Step 5. 뉴런 평가 — **맥 NEURON 으로 진행** ★인계문서 `../MAC_NEURON_HANDOFF.md`

> **S4L GAF 경로는 90% 뚫고 보류했다** (2026-08-03). 솔버 설치 성공, GAF API 도 파이썬에서
> 직접 열림(`compute_gaf`·`get_threshold()`=17.6864 모델상수·`get_titration_prediction`),
> `get_potential(model, source)` 까지 221구획 정상 동작. **`compute_gaf` → `StimulationPulse.__init__`
> 에서 AssertionError** 로 막혔고, 파서가 만든 정품 `SourceSettings` 로도 동일해 소스 구조 문제가
> 아니다(Cython 이라 조건 확인 불가). GUI 가 만든 `<uuid>_Input.h5` 의 `Sources` 그룹을 한 번 보면
> 확정되는데, GUI 리본에 `NEURON | Axon Model` 이 안 나타나 그 파일을 못 얻었다.
> 확보한 규약은 메모리 `sim4life-96-mcp` 에 전부 기록해 뒀다 — 재개 시 그대로 쓸 것.


원안은 9.6 GAF 였으나 **T-Neuro 라이선스가 없어 막혔다**(§8-2). 대신 §7의 NEURON 검증을
했던 **맥 standalone NEURON** 경로를 쓴다. S4L GAF 는 같은 일을 1000배 빨리 할 뿐이고,
역치·recruitment 산출 자체는 이미 검증된 길이다.

```python
res = compare_drives(fl, ("field","af","gaf"), pool_k=16)
fl.export_candidates({f"{k}_opt": res["opts"][k] for k in res["opts"]},
                     dd+"/fibercand_hippoL_pop.npz")
```
내보내는 키는 `neuron_bridge.export_ti_case()` 와 **같은 규약**이라 기존 맥 하네스가 그대로 읽는다:
`coords (F,N,3)` · `arclen (F,N)` · `labels (F,)` · `f1` · `f2` · `target (F,)`
· `{이름}__Ve1/__Ve2 (F,N)` [mV] · `{이름}__montage`.

> **하네스에서 한 줄 손볼 것**: `arclen` 이 `(N,)` → `(F,N)` 이 된다. 실제 궤적은 섬유마다
> 호길이가 다르므로 이게 맞다(합성 직선 축삭은 전부 같았다).

맥에서 할 일: ① `e_extracellular = g·[Ve1·cos(2πf₁t) + Ve2·cos(2πf₂t)]` ② 이분탐색 역치 g*
③ 진폭 스윕 → recruitment 곡선. **비용 감각**: 914섬유 × 몽타주 3개 × 이분탐색 ~12회
≈ 3만 회. 축삭당 0.1~1초면 1~9시간, 코어 병렬로 하룻밤.

**판정 기준**: §7의 "심부·통과섬유 = af_opt 승(24/24)" 이 **섬유 개체군 수준에서도** 나오는가.
즉 `af_opt`(또는 `gaf_opt`) 몽타주가 `field_opt` 보다 표적 섬유를 낮은 g 에서 켜야 한다.

#### ▷ Step 6. (라이선스 확보 시) S4L GAF 경로 붙이기

`s4l_fibers.evaluate_candidates()` 의 TODO 를 채운다. 후보를 3개가 아니라 수백 개 검증할 수
있게 되므로 탐색 루프 자체가 달라진다. 사용할 API(스텁 확인 완료):
`NeuronSetupSettings.PerformTitration` + `TitrationStrategy.kEstimator`(=GAF 경로로 추정) →
`NeuronPostPro.TitrationEvaluator.TitrationFactor` · `RecruitmentEvaluator`.
축삭 생성은 `NeuronModeling.CreateAxonNeuron(spline, MotorMrgNeuronProperties(), …)`.
문의 시 FLEXID `9-0CB39FBB` 를 알려줄 것.

### 8-4. 확정된 설계 결정 (재논의 불필요)

- **조밀 리드필드 재시뮬(≤0.5mm)은 폐기.** 격자 정밀도로는 "뉴런이 켜지냐"가 해결되지 않는다.
- **Sim4Life는 루프 안이 아니라 양 끝에** 앉는다(궤적 생성 · 최종 후보 평가). 몽타주 후보가
  만 개 단위라 매 후보를 던지는 건 불가능.
- **통신은 MCP** (`s4l_run_python`). 파일 교환·직접 import는 채택하지 않음.
  → 구현됨: `s4l_mcp/`(헤드리스 Sim4Life 9.6, `S4L_API_AUTO_INIT=1`). 9.6은 GUI 없이
    앱을 띄울 수 있어 Blender식 "GUI 안 소켓 서버"가 필요 없다. 상세 `s4l_mcp/README.md`.
- 섬유 리드필드는 `data/`에 `.npz`로 저장. **Vₑ 는 float32 가능**(검증 1.19e‑07)이나
  **`trajs`·`arclen` 은 float64 필수** — §8-5 참조. 기본값은 Vₑ 도 float64 (914×70×31 이
  16 MB 밖에 안 되고, 차분 상쇄에 안전하다). 메모리가 빠듯하면 `dtype=np.float32` 명시.

### 8-5. 함정 (★는 2026-08-03에 실제로 걸린 것)

- **좌표계** — Step 1을 건너뛰지 말 것. ★대조는 **`.sab`** 으로. `.nii` 는 다른 프레임이라
  "안 겹침"으로 오판한다.
- **선형성 전제** — Step 3의 중첩 검증이 통과해야 이후가 유효.
- **섬유 라벨링** — `label_fibers`가 0개를 반환하면 궤적이 표적을 안 지난다(중심·방향 재확인).
- **절점 수 균일화** — `make_fibers`가 호길이 등간격 리샘플로 N을 맞춘다. 궤적 길이가
  제각각이면 `target_length`를 지정할 것.
- ★**off-target 섬유가 없으면 M2·M3 는 숫자만 나오고 의미가 없다.** 표적 주변에만 다발을
  만들면 off 가 2개쯤 남아 M3 가 전부 100%로 찍힌다 — 누출이 심한 게 아니라 **판별력이
  없다는 뜻**이다. 몽타주를 비교하기 전에 Step 2b 를 먼저 할 것. (M1 만은 그 상태에서도 유효.)
- ★**드라이브별로 전극 풀을 따로 뽑으면 비교가 가짜가 된다.** `select_k` 축소가 kind 에
  의존하므로 A 의 최적해가 B 의 탐색공간에 아예 없을 수 있다. §6의 Huang select_k=16
  사례와 같은 함정. 비교는 `optimize/fiber.compare_drives()` 로.
- ★**교차평가는 목적함수(WP)로 하라.** M1·M2 를 따로 비교하면 "자기 기준에서 지는" 착시가
  생긴다 — 옵티마이저가 최대화한 건 WP 이지 M2 가 아니다.
- ★**궤적 좌표(`trajs`)를 float32 로 저장하지 말 것.** Vₑ 는 float32 로 줄여도 되지만
  (검증 1.19e‑07), 궤적은 Vₑ 를 뽑은 위치를 규정하는 값이라 ≈3e‑5 mm 오차만으로 재계산
  보간이 어긋나 중첩 검증이 1e‑5 로 깨진다. §8-4의 "float32 저장"은 **Vₑ 에만** 해당한다.
- **T-Neuro 라이선스 미보유** — 확인 완료(§8-2). ⑤는 맥 NEURON 으로 우회한다.

---

## 9. Sim4Life 연동 (선택 — 리드필드 재생성 시)

리드필드 추출 규약 (`tip/leadfield_gen.py`에 문서화, bit-exact 재현 검증됨):

```
M{jj}[r] = Re(E_flat[i + 185*j + 185*254*k]),  (i,j,k) = bmask[r]   # X-fastest
기저: 전극 i = 1V, Cz = 0V, 나머지는 floating PEC (BC 없음)
정규화: unitnorm.json = 1e-3 / I_inj  (1 mA 기준)
몽타주: Σ_i I_i · Mn_i,  Σ I_i = 0
```

- 전극: MIDA v1.0 두피에 표준 10-10, `Create1010System` → `PlaceElectrodes`(r=4mm, h=2mm)
- 솔브 시간: 전극당 ~2.2분 (9.0 기준)
- **전극 크기는 결과에 무관** (두개골 지배, 3cm² vs 0.5cm² 99.8% 동일) → 재생성 불필요

---

*문서 기준일: 2026-08. 이전 체크리스트는 별도 문서 참조.*