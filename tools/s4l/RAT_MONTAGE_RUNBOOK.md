# 랫 몽타주 — 내일 실행 순서

2026-08-19 작성. 준비는 전부 끝났고 **솔버 라이선스 좌석 하나만 기다리면 됩니다.**
좌석은 연구실 공용이고 그때 `IBY-IMR-DESKTOP` 이 쓰고 있었습니다.

---

## 0. 좌석이 비었는지 먼저 본다

```bash
"/c/Program Files/Sim4Life_9.6/lmstat.exe" -a | grep -A6 "Users of QS_SOLVER"
```

`Total of 0 licenses in use` 면 비어 있습니다. 누가 쓰고 있으면 그 줄에
`IBY-IMR-DESKTOP ... start ...` 처럼 장비 이름이 뜹니다.

> ⚠ **Sim4Life MCP 세션도 같은 QS_SOLVER 좌석을 씁니다.** 세션이 떠 있으면 `iSolve` 가
> `Licensed number of users already reached (-4,132)` 로 즉시 실패합니다.
> 클로드 세션에서 작업 중이면 `s4l_reset_session` 을 먼저 부르고 솔브하세요.

---

## A. 물리 검증 — 리드필드가 진짜 몽타주를 맞히는가 (약 4분)

**왜 하는가.** 두 머리의 리드필드가 다른 규약으로 풀렸습니다.

| | 기저 k 의 뜻 |
|---|---|
| 사람 | 전극 k 가 1 V, Cz 가 0 V, **나머지 전극은 아예 없음** |
| 랫 | 전극 k 가 1 V, **나머지 37개가 전부 0 V**(EM LF 포트 모드) |

사람은 +i/−i 를 걸면 Cz 순전류가 0 이라 기준전극이 식에서 빠져 `i·(LF[A]−LF[B])` 가 **정확**합니다.
랫은 그렇지 않습니다 — 그 식은 나머지 36개가 전류를 나눠 흘리는 상황에 해당하는데,
실제 실험에서는 그 전극들이 **떠 있습니다**. Ø0.25 mm 로 작지만 PEC 이고 두피에 닿아 있어
36개가 묶이면 머리를 가로지르는 저임피던스 경로가 생깁니다. 크기를 재야 압니다.

### 실행

입력 파일은 **이미 써져 있습니다**(`AF4` 1 V, `C3` 0 V, 나머지 36개 띄움).

```bash
cd "/c/Users/imrla/Desktop/Taemin/s4l_projects/rat_montage_test.smash_Results"
"/c/Program Files/Sim4Life_9.6/Solvers/iSolve.exe" 69562c72-*_Input.h5 > solve.log 2>&1
echo "rc=$?"; tail -3 solve.log
```

약 3.5분. 성공하면 `*_Output.h5` 가 **1.4 GB 안팎**이 됩니다(3.85 MB 면 실패한 껍데기).

```bash
cd "/c/Users/imrla/Desktop/Taemin/tip"
TIP_MODEL=rat "/c/Users/imrla/Desktop/Taemin/.venv-s4l/Scripts/python.exe" \
  tools/s4l/rat_montage_check.py --a AF4 --b C3
```

### 결과 읽는 법

- **크기비 중앙값이 1.0 근처(±5%)이고 방향 코사인이 0.99 이상** → 접지 효과가 무시할 수준.
  랫 리드필드를 지금 방식 그대로 몽타주에 쓰면 됩니다.
- **크기비가 1에서 크게 벗어남** → 리드필드가 몽타주를 과대/과소평가합니다.
  M1/M2/M3 표까지 보고 **순위가 바뀌는지**를 따로 판단하세요. 세기만 틀리고 순위가 같으면
  최적화는 유효하고 절대값만 보정 대상입니다.
- **방향 코사인이 낮음** → 공간 구조가 다른 것이라 보정으로 못 덮습니다. 그때는 랫 리드필드를
  "나머지 전극 없음" 규약으로 다시 푸는 것을 검토해야 합니다(전극당 약 200초 × 37 ≈ 2시간).

한 쌍만으로 판단하지 마세요. 최소 두세 쌍(예: `--a C6 --b CP5`, `--a AF4 --b PO3`)을 보고,
근육에 닿은 `C6·CP6·P6` 은 전도도가 1.5~2.7배라 따로 봐야 합니다.

---

## B. GUI 에서 랫 몽타주 보내기 (약 7분)

A 가 통과한 뒤에 하세요. A 가 실패하면 여기서 나오는 숫자의 의미가 달라집니다.

1. GUI 실행 — `run_gui.bat` 또는

   ```bash
   cd "/c/Users/imrla/Desktop/Taemin/tip"
   "/c/Users/imrla/miniconda3/envs/tip/python.exe" src/tip/gui/app.py
   ```

2. 헤더 아래 **머리 모델** 에서 `NeuroRat (쥐)` 선택 → 페이지가 새로 뜹니다.
3. **최대 전류** 칸에 값을 넣습니다. **랫은 비어 있고, 넣기 전에는 계산이 안 됩니다** —
   확립된 값이 없어서 일부러 그렇게 뒀습니다. 시험용으로는 `0.1` mA 가 무난합니다.
4. 표적 선택(예: `해마 (좌) · 57.2%`), 전극 선택, **계산**.
5. **Sim4Life 정밀 검증** 칸의 `▶ Sim4Life 로 보내기`.

배선은 이미 되어 있습니다:

```
rat   기준 프로젝트 rat_lf.smash        작업 슬롯 montage_gui_rat.smash
human 기준 프로젝트 mida1010_rebuild    작업 슬롯 montage_gui.smash
```

랫 프로젝트는 리드필드용이라 `Active`(포트 37개)·`Passive`(PO8) 구조인데,
`s4l_montage.set_pair` 가 **포트 모드를 끄고** 2단자로 바꿉니다.

결과는 `tip/outputs/montage/<잡ID>/` 와 캐시 `tip/outputs/cache/montage_s4l/<키>/` 에 남습니다.

> 이 경로는 아직 **한 번도 끝까지 안 돌려봤습니다**(좌석 때문에). 처음 돌릴 때는 실패해도
> 이상하지 않고, 로그는 `tip/outputs/jobs/<잡ID>.log` 에 있습니다.

---

## 정리해도 되는 것

검증이 끝나면 시험용 사본을 지워도 됩니다.

```bash
rm -rf "/c/Users/imrla/Desktop/Taemin/s4l_projects/rat_montage_test.smash"*
```

`rat_lf.smash`(원본)와 `rat_lf.smash_Results/`(포트 입력 37개, 172 MB)는 **남겨두세요** —
Sim4Life 없이 `iSolve` 만으로 리드필드를 다시 풀 수 있는 유일한 경로입니다.
