# 맥에서 가져올 것 — WSL NEURON 이식 준비물

> 2026-08-05. T-Neuro 라이선스가 내년이므로 **WSL NEURON 이 당분간 유일한 생리 엔진**이다.
> 맥에만 있는 자산을 옮겨야 파이프라인의 T2 층이 선다. 배경은 `PIPELINE.md` §6-①.

---

## 0. 한 줄

**NEURON 작업 디렉토리를 통째로.** 단 컴파일 산출물(`x86_64/`·`arm64/`)은 빼고.
아래는 "통째로"가 안 될 때를 위한 항목별 목록이자, 받은 뒤 빠진 게 없는지 확인하는 체크리스트다.

---

## 1. 먼저 — 맥에서 위치 찾기

```bash
find ~ -name "*.mod" -not -path "*/nrn/*" -not -path "*/neuron/*" 2>/dev/null
find ~ -name "fiber_pop*.py" -o -name "analyze_fiber_pop*.py" 2>/dev/null
```

첫 명령이 가리키는 디렉토리가 `nrnivmodl` 을 돌린 곳이다. 둘이 같은 디렉토리면 그거 하나만 뜨면 된다.

## 2. 통째로 가져오기 (권장)

```bash
cd <위에서 찾은 디렉토리의 부모>
tar czf ~/Desktop/neuron_handoff.tgz \
  --exclude='x86_64' --exclude='arm64' --exclude='__pycache__' \
  --exclude='.git' --exclude='.DS_Store' \
  <디렉토리이름>
```

**`x86_64/`·`arm64/` 를 제외하는 이유**: 맥에서 컴파일된 `.o`/`.so` 는 리눅스에서 못 쓴다.
WSL 에서 `nrnivmodl` 로 다시 컴파일해야 하므로 **`.mod` 원본만** 있으면 된다.
(실수로 포함해도 해롭진 않지만, 있으면 "컴파일했다"고 착각하기 쉽다.)

크기가 크면 대개 결과 `.npz` 때문이다 — §5 를 보고 필요한 것만 골라도 된다.

---

## 3. 필수 — 이게 없으면 아무것도 못 한다

### 3-1. MRG 축삭 모델

| 항목 | 비고 |
|---|---|
| `.mod` 파일 **전부** | `axnode` 계열 + MYSA/FLUT/STIN 관련. 파일명은 배포본마다 다르다 |
| `.hoc` 파일 **전부** | `MRGaxon.hoc` 등 모델 정의 |
| `README` / 출처 메모 | ★**ModelDB 3810 원본 그대로인지, 손댔는지** |

> `MAC_RESULTS_HANDOFF.md` 는 "MRG(ModelDB 3810 정품)"라고 기록했지만,
> `NEURON_ROLE.md` §1-2 는 기전 이름을 `axnode`·`mysa_motor`·`flut_motor`·`stin_motor` 로 적었다.
> 뒤쪽은 **Sim4Life/Yale 배포본의 명명**이라 ModelDB 3810 원본과 다르다.
> **둘 중 어느 것을 실제로 컴파일했는지가 중요하다** — 파일을 받아보면 바로 확정된다.
> (`nrnivmodl` 을 돌린 디렉토리에 있는 게 실제로 쓴 것이다.)

### 3-2. 하네스·분석 코드

`MAC_RESULTS_HANDOFF.md` §1 이 "재현 코드"로 이름만 남긴 것들:

| 파일 | 하는 일 |
|---|---|
| `fiber_pop.py` | ★개체군 titration 본체 — 케이스 npz → 이득격자 스캔 → 역치 |
| `analyze_fiber_pop.py` | recruitment·선택성·짝지은 비교 집계 |
| `fig_fiber_pop.py` | 3패널 그림 |

**그리고 그 뒤에 자란 것들** — 문서에 결과만 있고 코드 이름이 안 적힌 것들이라 놓치기 쉽다:

| 무엇 | 어디에 결과가 인용돼 있나 |
|---|---|
| **시상 케이스를 돌린 것** | `ANGLE_SWEEP_REQUEST.md` §1-① — 시상 MRG 9 kHz `g*=862` |
| ★**동조(entrainment) 분석** | 같은 문서 §1-③ — 표적 위상일치 100.0% / off 59.6%, 분리력 40.4%p·17.8%p |
| 전기장 기준 재평가 | 같은 문서 §1-② — 표적/off `\|E\|` 비 1.23·0.71·0.90 |
| 주파수 스윕 (돌렸다면) | `MAC_FREQ_SWEEP_HANDOFF.md` |

> ★**동조 분석이 특히 중요하다.** 역치 틀에서 subthreshold 동조 틀로 넘어간 게
> 이 프로젝트의 최신 방향인데(`ANGLE_SWEEP_REQUEST.md` §1-①), **그 구현을 설명한 문서가 없다.**
> 코드가 유일한 명세다. 이것만 빠져도 각도 스윕 결과를 우리가 해석할 수 없다.

---

## 4. 환경 정보 — 재현 대조에 필요

맥에서 아래를 실행해 출력을 텍스트로 같이 보내면 된다.

```bash
python -c "import neuron; print('NEURON', neuron.__version__)"
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
uname -m                      # arm64 / x86_64
```

WSL 쪽은 **NEURON 9.0.1 · numpy 2.2.6 · scipy 없음(설치 예정) · x86_64** 다.
NEURON 버전이 같으면 재현 대조가 훨씬 깨끗하다.

추가로 알려주면 좋은 것:
- 난수 시드 (있다면). `MAC_FREQ_SWEEP_HANDOFF.md` §4 는 `20260804` 를 고정하라고 적어뒀다
- 실행 명령 원문 (프로세스 수·인자)

---

## 5. 결과물 — 이미 있는 것 / 없는 것

**이미 여기 있다** (다시 안 가져와도 된다):

```
tip/0804_NEURON_result/thresholds.npz      해마 914섬유 × 3몽타주 역치  ← ★이식 판정의 정답지
tip/0804_NEURON_result/chunk0..3.npz       격자별 발화·개시위치·비단조 플래그
tip/0804_NEURON_result/F_fiber_population.png
```

**없다 — 가져와야 한다**:

| 무엇 | 왜 필요한가 |
|---|---|
| `neuron_case_thalamusL_v2.npz` | 각도 스윕이 "이것과 동일 표적"을 전제한다. 대조군이 없으면 각도 결과를 못 읽는다 |
| **시상 케이스 NEURON 결과** | `g*=862` 의 출처. 지금은 문서에 숫자만 있고 원자료가 없다 |
| **동조 분석 결과** | 위 §3-2 의 위상 일치율·분리력 수치의 원자료 |
| 주파수 스윕 결과 (돌렸다면) | `thresholds_sweep.npz`. 안 돌렸으면 WSL 에서 우리가 돌리면 된다 |

---

## 6. 받은 뒤 할 일 (WSL 쪽)

1. `conda install -n neuron_env scipy` — 현재 없다
2. `.mod` 디렉토리에서 `nrnivmodl` — 리눅스용 재컴파일
3. **전도속도 검증** — MRG 10 µm 에서 **54.12 m/s** 가 나와야 한다
   (`MAC_RESULTS_HANDOFF.md` §2 가 원본 일치를 확인한 값)
4. ★**재현 대조** — `data/fibercand_hippoL_full.npz` 를 넣어
   `0804_NEURON_result/thresholds.npz` 와 **역치가 일치**하는지.
   이것만이 이식 성공의 증거다.

예상 시간: 맥 3.5시간(4프로세스) → WSL 22코어면 **~40분**.

---

## 7. 참고 — 지금 맥에 걸려 있는 일

`0804_NEURON_result/ANGLE_SWEEP_COVER.md` 로 넘긴 **시상 각도 스윕 12각도**가
맥에서 NEURON 대기 중이다(권장 조합 n=3: `004, 006, 009`).

하네스가 이쪽으로 넘어오면 **그 실행을 WSL 에서 우리가 직접 돌릴 수 있다.**
맥에서 이미 돌리고 있다면 결과만 받으면 되고, 아직이면 이식 후 여기서 돌리는 쪽이 빠르다.
