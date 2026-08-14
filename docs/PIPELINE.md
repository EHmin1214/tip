# 통합 파이프라인 설계 — TIP(프론트) · Sim4Life(백) · NEURON(백)

> 작성 2026-08-05. `SETUP.md` §8 의 5단계 구조를 **사람이 문서로 중계하던 것**에서
> **하나의 앱이 잡으로 돌리는 것**으로 바꾸는 설계.
> §8-4 의 확정 사항(S4L 은 루프 밖 · 섬유 리드필드 선형중첩 · `.npz` 저장)은 그대로 유지한다.

---

## 0. 한 장 요약

```
브라우저 (index.html)
   │  HTTP :8765
   ▼
tip/gui/app.py                    ← 라우팅·렌더링만. 얇게 유지
   │
   ▼
tip/orch/                         ★신규 — 잡 큐 · 캐시 · 백엔드 클라이언트
   ├─ jobs.py      잡 정의·상태·디스크 영속
   ├─ cache.py     내용주소 캐시 (입력해시 → 산출물)
   ├─ s4l.py       S4L 워커 클라이언트
   └─ nrn.py       WSL NEURON 실행기
   │
   ├──► s4l_mcp/worker.py  (stdio → TCP 승격, 세션 영속, 라이선스 1좌석)
   │       MIDA 임포트 · GenerateSplines · Voxeler · iSolve
   │
   └──► wsl.exe -d Ubuntu → nrn/runner.py  (neuron_env · 22코어)
           MRG titration → thresholds.npz
```

핵심은 **두 백엔드를 "잡 계약"으로 바꾸는 것**이다. 지금 S4L 은 MCP(=Claude 전용 stdio)로만
닿고, NEURON 은 맥에서 사람이 돌려 `.md` 로 주고받는다. 둘 다 GUI 에서 닿지 않는다.

---

## 1. 현재 실측 상태 (2026-08-05)

| 층 | 상태 | 근거 |
|---|---|---|
| TIP GUI | 동작. 비동기 잡 패턴 **이미 있음** (`/api/optimize` → job → `/api/progress`) | `tip/gui/app.py:651` |
| TIP 코어 | 리드필드 70전극×1.9M복셀 사전계산 · 최적화 8종 · `fiberlead` · `neuron_bridge` | — |
| S4L 헤드리스 | 모델링·복셀·궤적·후처리 OK | `s4l_mcp/README.md` |
| **S4L 솔버** | **★헤드리스 실행 확인됨 (아래 §2)** | 본 세션 실측 |
| WSL NEURON | NEURON 9.0.1 · **22코어 / 31GB** · numpy 2.2.6 · `/mnt/c` 접근 OK | 본 세션 실측 |
| **NEURON 자산** | **★없음** — MRG `.mod`·하네스·scipy 전부 부재 | 본 세션 실측 |

### ★ 유일한 물리적 부재는 NEURON 하네스다

맥에서 돌던 `fiber_pop.py` · `analyze_fiber_pop.py` · MRG(ModelDB 3810)가 **이 머신에 없다.**
`MAC_RESULTS_HANDOFF.md` §1 이 "재현 코드"로 목록만 남겼고 파일은 맥에 있다.
파이프라인에서 "생리학적 테스트"를 담당할 층이 통째로 비어 있는 상태다.

---

## 2. ★ 정정 — 헤드리스 S4L 에서 솔버는 돌아간다

`s4l_mcp/README.md` 의 "**솔버 실행 불가 (헤드리스 한계)**" 와
`MIGRATION_STATUS.md` §2-2 의 "헤드리스에서는 어떤 솔버도 제출되지 않는다"는 **오진이다.**

실측:

```
XSimulator.GetAvailableServers()  →  [('localhost', Uuid("f838e856-…"))]   ← 비어있지 않다
XSimulator.ConnectToLocalAres()   →  True
sim.WriteInputFile()              →  OK (134,616 bytes)
sim.RunSimulation(wait=True)      →  iSolve 기동 → QS_SOLVER 라이선스 체크아웃 → 솔버 실행
```

솔버 로그 원문:

```
[INFO]: [2026-Aug-05 13:21:09] Checking out license feature 'QS_SOLVER', …
[ERROR]: There seem to be two Dirichlet objects touching with different Dirichlet value!
```

즉 **끝까지 갔고, 실패 사유는 장난감 모델의 결함**이지 인프라가 아니다.

### 오진의 원인 — 요약된 오류 문구

`WriteInputFile()` 은 `failure_type` **없이** 예외를 던지는데, 예외 클래스의 기본 문자열이
`Failure type: network` 라서 네트워크 문제로 읽힌다. 소스에 정리 안 된 흔적이 그대로 있다:

```python
# C:\Program Files\Sim4Life_9.6\Python\Lib\site-packages\s4l_v1\_api\_simwrappers_simulation.py:462
def WriteInputFile(self) -> None:
    if not self.raw.WriteInputFile():
        # @Pedro: this needs cleanup with potentially new failure_type
        raise SimulationLaunchError("Write input file failed")
```

> `MIGRATION_STATUS.md` §2-2 가 남긴 교훈("오류 문구를 요약하지 말고 원문 그대로 볼 것")이
> **같은 파일 안에서 한 번 더 재발했다.** 이번엔 라이브러리 소스를 읽어야 드러나는 형태였다.

### 실제로 걸렸던 설정 결함 2가지 (재현 시 주의)

1. `AddOverallFieldSensorSettings()` 누락 — 출력 센서가 없으면 무효
2. Dirichlet 을 **솔리드 바디**에 부여 — EM LF 는 `sim.ListBoundaryPlanes()` 가 주는
   **경계평면**(`Plane X-`…`Plane Z+`)에 걸어야 한다

> unstructured 쪽은 별개로 `NotFoundException: Could not find model id`(메시 미생성)로 실패한다.
> **`Taemin/` 루트의 1928바이트 `*_Input.h5` 스텁이 그 실패의 잔재**다. 정상 입력파일은 100KB+.

### 결과 — 열리는 것

| 열리는 것 | 의미 |
|---|---|
| §9 리드필드 재생성 자동화 | GUI 불필요. 전극당 ~2.2분 × 70 ≈ **2.6시간** 무인 실행 |
| **스케일 감사 해결 수단** | 알려진 조건으로 다시 풀어 4~6배 과대의 원인을 잡을 수 있다 (§6) |
| S4L GAF 막힘 해소 경로 | GUI 참조 `Sources` 그룹을 헤드리스로 뽑아 `StimulationPulse` 조건 확정 가능 |

⚠ **미검증**: 70전극 전체 리드필드를 헤드리스로 완주한 적은 아직 없다. 위는 10³복셀 장난감이다.
⚠ Neuron 시뮬의 `WriteInputFile` 은 NEURO 라이선스가 없어 되는지 미확인. QS 로만 확인했다.

---

## 3. 설계 결정

### D1. S4L 워커를 stdio → TCP 로 승격

지금 `worker.py` 는 파이프 기반이고 클라이언트가 `server.py`(MCP) 하나다.
소켓으로 바꾸면 **MCP(Claude)와 GUI 가 같은 세션을 공유**한다.

그래야 하는 이유:
- S4L 앱 초기화 1.2~3.6초 + **MIDA `.sab` 임포트 6~7초**(384MB)
- **라이선스 좌석이 하나**다. 세션 둘이면 좌석 경쟁
- 고아 `AresApplication.exe` 누수 지점이 두 배가 된다

변경 범위는 작다 — `worker.py` 의 길이접두 JSON 프로토콜은 그대로 두고 전송만 파이프→소켓.
`server.py` 는 클라이언트 하나가 되고, `orch/s4l.py` 가 두 번째 클라이언트가 된다.

### D2. NEURON 은 데몬 없이 프로세스 단위 실행

```
wsl.exe -d Ubuntu -- bash -lc 'conda run -n neuron_env python /mnt/c/.../nrn/runner.py <job.json>'
```

데몬을 두지 않는 이유:
- NEURON 은 세션 상태를 안고 있을 필요가 없다 (케이스 `.npz` 만 있으면 됨)
- `h` 는 프로세스 싱글턴이라 **케이스 간 오염 위험**이 실재한다 → 격리가 오히려 안전
- 22코어 병렬은 러너 내부 `multiprocessing` 으로 충분

데이터는 `/mnt/c/Users/imrla/Desktop/Taemin/tip/data/` 를 직접 읽는다. **복사 불필요.**

> ⚠ `/mnt/c` 는 WSL2 에서 느리다(9p/드바이스 경유). 역치 스캔은 계산 지배적이라 문제없지만,
> 큰 `.npz` 를 반복 로드하면 체감된다. 러너는 **시작 시 한 번 읽어 메모리에 올린다.**

### D3. 4-tier 잡 모델

지연시간이 4자릿수 차이라 UI 취급이 달라야 한다.

| Tier | 내용 | 지연 | 백엔드 |
|---|---|---|---|
| **T0** | 몽타주 최적화 (리드필드 선형중첩) | **<1초** | 로컬 (기존) |
| **T1** | 섬유 궤적 · 마스크 · 표적 기하 | 초~분 | S4L |
| **T2** | NEURON titration (개체권 역치) | **분~시간** | WSL |
| **T3** | 리드필드 재생성 | **~2.6시간** | S4L + iSolve |

**T2/T3 는 브라우저 세션보다 오래 산다.** 지금 `app.py` 의 `JOBS` 는 메모리 dict 라
서버 재시작 시 날아간다 → 디스크 영속 필요.

### D4. 내용주소 캐시 — 이 프로젝트에는 선택이 아니다

`data/cache/<kind>/<hash>.npz` + 나란히 `<hash>.meta.json`.

이 프로젝트는 결론이 여러 번 뒤집혔고([[seqti-heuristic-bias]] · §6-4 풀분리 · off 정의 변경),
매번 "**어떤 입력으로 낸 숫자냐**"가 쟁점이었다. 입력 해시로 키잉하면
(a) 재실행이 공짜 (b) 출처가 자동 기록된다.

**해시에 반드시 들어가야 하는 것** — 빠뜨리면 과거의 사고가 재발한다:

| 항목 | 왜 |
|---|---|
| 리드필드 버전 + **스케일 보정계수** | §6 |
| **off-target 정의** (GM만 / GM∪WM) | ★이거 하나로 시상 몽타주가 **전부 바뀌었다**. `ANGLE_SWEEP_COVER.md` 주의1 |
| 전극 풀 · `select_k` | ★풀을 16→24 로 키우자 세 드라이브 최적 몽타주가 **전부 바뀌었다**. `MIGRATION_STATUS.md` §6-5 |
| 섬유 개체군 시드 · 개수 · 방향분포 | off 개체군이 없으면 M2·M3 가 무의미 |
| 축삭모델 · 직경 · 말단조건 · f1/f2 · 이득격자 · dt/tstop | NEURON 결과의 전부 |

### D5. 스케일 보정을 파이프라인 상수로 승격

절대 필드가 문헌 대비 **4~6배 과대**인데 원인 미확정이고 보정이 안 걸려 있다
([[seqti-scale-audit]] · `SCALE_AUDIT.md`).

지금까지는 "상대 비교만 한다"로 넘어갈 수 있었다. **NEURON 이 들어오면 못 넘어간다** —
역치 이득 `g*` 를 mA 로 환산하는 순간 4~6배가 그대로 실린다. 안전한계·용량 예측이 전부 틀어진다.

> **★2026-08-05 갱신 — 이 절의 `LEADFIELD_SCALE` 이라는 이름은 폐기한다.** 상수가
> **두 개**로 갈렸고, 문서마다 이름이 셋(`LEADFIELD_SCALE`·`FIELD_SCALE`·`LEADFIELD_AMP_FIX`)
> 돌아다녔다. 정리는 `SCALE_VALIDATION_HANDOFF.md` 의 표를 정본으로 한다.
>
> | 상수 | 무엇 | 상태 |
> |---|---|---|
> | **`config.LEADFIELD_AMP_FIX`** | **정의에서 나온 2.0배**(`El. Loss Density`=σ\|E\|²/2 를 전류로 씀) | **= 0.5, 도입 완료** |
> | `config.FIELD_SCALE` | 원인미상 **잔여분**을 흡수할 계획 상수 | **아직 없음** (확정 전엔 넣지 않는다) |
>
> 아래 본문은 둘이 갈리기 전에 쓴 것이다. `LEADFIELD_SCALE` → 위 표로 읽을 것.

조치: `tip/config.py` 에 (위 표의 상수를) **명시적으로** 두고,
값이 미확정인 동안 **절대단위가 들어간 모든 결과에 배지를 붙인다.** 무차원 지표(M2·위상·분리력)는
영향 없으므로 배지 없음.

---

## 4. 잡 계약 (데이터 스키마)

### 케이스 정의 — 프론트가 만드는 단일 객체

```jsonc
{
  "anatomy":  { "model": "MIDA_v1.0", "leadfield": "leadfieldF", "scale": 1.0 },
  "target":   { "kind": "preset|coords|mask", "name": "hippocampus_L",
                "center": [-51.7, 259.7, 28.3], "radius_mm": 6.0,
                "axis": [0.17, -0.486, -0.857] },
  "offtarget":{ "labels": ["GM", "WM"] },              // ★기본 GM∪WM
  "montage":  { "method": "classic|gevd|dual|timemux|selective|huang|fiber",
                "drive": "ve|field|af|gaf",
                "pool": { "select_k": 24, "shared": true },
                "budget_mA": 2.0, "f1": 2000, "f2": 2100 },
  "fibers":   { "target_seeds": 300, "target_length_mm": 20, "target_radius_mm": 6,
                "off_seeds": 300, "off_lines": 4, "orientation": "isotropic" },
  "physio":   { "model": "MRG_motor", "diameter_um": 10.0,
                "terminal": "absorbing",                // ★ sealed 는 §5 함정
                "gain_grid": [2, 500, 20], "tstop_ms": 110, "dt_ms": 0.005,
                "ramp_ms": 30, "window_ms": 45 }
}
```

### T1 산출 — 섬유 리드필드 (기존 규약 유지)

`fiberlead.build_fiber_leadfield()` 출력 그대로. `trajs`·`arclen` 은 **float64 필수**.

### T2 입력 — NEURON 케이스

`fiberlead.export_candidates()` / `neuron_bridge.export_ti_case()` 의 **기존 키 규약 그대로**.
맥 하네스가 이미 이 형식을 먹었으므로 WSL 러너도 같은 것을 먹으면 **1차 결과 재현 대조**가 된다.

```
coords (A,N,3) float64 · arclen (A,N) float64 · labels (A,) str
<montage>__Ve1 (A,N) mV · <montage>__Ve2 (A,N) mV · f1 · f2 · target (A,) bool
```

### T2 산출 — 역치

`MAC_FREQ_SWEEP_HANDOFF.md` §6 형식을 그대로 채택(조건 축 포함):

```
thresholds.npz
  conditions (C,) str · f1,f2 (C,) · tstop,dt (C,)
  fiber_index (F,) int · target (F,) bool · grid (C,G)
  <montage> (C,F) float   역치 g*  (미발화 = inf)
  nonmono_<montage> (C,F) bool     ★비단조(kHz 블록) 플래그
  s_initiation (C,F) float         개시 위치 [mm, 축삭중앙=0]
```

---

## 5. 프론트에 붙일 것

지금 GUI 는 "표적 고르기 → 최적화 → 3D 필드"까지다. 세 덩어리가 더 필요하다.

### 5-1. 실험 설계 패널
흩어진 입력을 §4 의 **케이스 객체 하나**로 묶는다. 프리셋 저장/불러오기.
**핵심**: off-target 정의와 전극 풀을 **사용자에게 보이게** 둔다 — 숨기면 §3-D4 의 사고가 재발한다.

### 5-2. 잡 보드
T1/T2/T3 목록 · 진행률 · 예상시간 · 취소 · 캐시 히트 표시.
브라우저를 닫아도 살아 있어야 하므로 서버 재시작 후에도 복원.

### 5-3. 생리 결과 뷰 (지금 전무)
계산은 `analyze_neuron_results.py` 에 **이미 있다.** 렌더만 붙이면 된다.

| 패널 | 무엇 | 왜 필요한가 |
|---|---|---|
| Recruitment 곡선 | 이득 vs 발화% (표적/off) | 몽타주 비교의 1차 단위 |
| 역치 분포 | g* 히스토그램, 짝지은 승률 | af>field 같은 주장의 근거 |
| **선택성 ROC + AUC** | 단일 작동점 비교의 함정 회피 | ★`MAC_FREQ_SWEEP_HANDOFF.md` §4 |
| **비단조(블록) 비율** | 개체군의 28~41% | 선형 지표로 원리적으로 못 보는 것 |
| 개시위치 히스토그램 | field↔AF 반전의 사후 확인 | `NEURON_ROLE.md` §2-④ |

---

## 6. 구현 순서

값싸고 크게 막힌 것부터.

### ①단계 — WSL NEURON 하네스 이식 ★최우선

파이프라인의 **유일한 물리적 부재**다. 이게 없으면 "생리학적 테스트"가 아예 안 된다.

할 일:
1. `conda install -n neuron_env scipy` (현재 없음)
2. MRG `.mod` 확보 — ModelDB 3810 (McIntyre 2002). `nrnivmodl` 로 컴파일
3. `nrn/runner.py` 작성 — 케이스 `.npz` → `extracellular` 구동 → 이득격자 titration →
   `thresholds.npz`. 22코어 `multiprocessing`

**판정 기준 — 이것만이 이식 성공의 증거**:
> `0804_NEURON_result/thresholds.npz` 를 **재현**해야 한다.
> 같은 입력(`fibercand_hippoL_full.npz`) → 같은 역치. 맥 결과가 정답지 역할을 한다.

예상 비용: 맥 914섬유×3몽타주 = 3.5시간(4프로세스) → 22코어면 **~40분**.

### ②단계 — 오케스트레이터 + 캐시
`tip/orch/{jobs,cache,nrn}.py`. GUI 에서 NEURON 잡을 띄우고 결과를 받는 최소 경로.
T2 부터 붙인다 (①이 끝나면 바로 쓸모가 생기므로).

### ③단계 — S4L 워커 TCP 승격 + 궤적 온디맨드
"사용자가 새 표적을 고르면 섬유 궤적이 자동 생성"이 여기서 열린다.
지금은 좌해마·좌시상만 궤적이 있다.

### ④단계 — 리드필드 재생성 자동화
§2 로 열린 경로. **첫 용도는 새 몽타주가 아니라 스케일 감사 해결**(§3-D5).
알려진 조건으로 다시 풀어 4~6배의 원인을 잡는다. 그게 잡히기 전엔 절대단위 결과를 못 낸다.

---

## 7. 함정 — 파이프라인이 자동화하면 안 되는 것

이 프로젝트가 **실제로 걸렸던** 것들이라 코드로 방지해야 한다.

| 함정 | 사고 기록 | 파이프라인의 방어 |
|---|---|---|
| **탐색공간 자동 축소** | `select_k=16` 으로 두 번 결론 뒤집힘 (Huang · 드라이브 비교) | 축소하면 **로그에 남긴다.** 드라이브 비교는 `compare_drives()` 공통풀로만 |
| **off 정의가 결과를 지배** | GM만 → GM∪WM 로 바꾸자 표적 5종 중 3종 몽타주 변경 | 케이스 해시에 포함 + UI 에 노출 |
| **단일 작동점 선택성 비교** | 격자 양자화 ±17% 에 취약, af vs gaf 결론이 뒤집힘 | ROC AUC 를 항상 동반 출력 |
| **궤적 자르기 길이 = 숨은 자유 파라미터** | sealed 말단이면 결과가 "10.4mm 지점 필드"로 결정됨 | 말단은 `absorbing` 고정. sealed 는 민감도 동반 필수 |
| **`trajs` float32 저장** | 중첩 검증이 1e-15 → 1e-5 로 깨짐 | 저장 시 dtype 단언 |
| **목적함수 아닌 축으로 승패** | M1·M2 따로 비교해 "자기 기준에서 지는" 착시 | 교차평가는 WP 로, 평가 드라이브 안에서 정규화 |
| **절대 V/m 을 그대로 보고** | 4~6배 과대 중 **2.0배는 확정·보정 완료** | `LEADFIELD_AMP_FIX`(도입됨) + 잔여분은 `FIELD_SCALE`(미도입) · 절대단위 결과에 배지 |

---

## 8. 열린 질문 (사용자 판단 필요)

1. **MRG `.mod` 를 어디서 가져오나** — 맥에 있는 것을 복사하는 게 가장 싸다(정품 ModelDB 3810,
   전도속도 54.12 m/s 검증됨). 맥 접근이 안 되면 ModelDB 에서 새로 받아 **전도속도부터 재검증**해야 한다.
2. **맥 하네스 코드(`fiber_pop.py` 등)를 가져올 수 있나** — 가져오면 ①단계가 "이식"이지만,
   없으면 "재작성"이고 검증 부담이 커진다. (단 정답지 `thresholds.npz` 는 이미 여기 있으므로
   재작성이어도 판정은 가능하다.)
3. **T-Neuro 라이선스를 계속 추진하나** — GAF 가 되면 T2 가 1000배 빨라져 파이프라인 성격이
   바뀐다(역치가 "몇 시간"에서 "몇 초"로 → 최적화 루프 안에 넣을 수 있다).
   WSL NEURON 은 그 사이의 정답지 겸 대체재다.
