# 맥 NEURON 인계 — 섬유 개체군 역치 평가 (§8 ⑤단계)

> §7의 NEURON 검증(해마 24/24 af_opt · 피질 field_opt)을 **섬유 개체군 수준**으로 올리는 실행.
> Sim4Life GAF 경로는 보류 — 사유는 이 문서 §5.

---

## 1. 무엇이 달라졌나 (§7 대비)

| | §7 (기존) | 이번 |
|---|---|---|
| 축삭 | 합성 **직선** 축삭 3~6개 | **실제 궤적 914개** (MIDA `ViP.GenerateSplines`, 직선도 0.893) |
| 표적/off | 표적점 1곳 + off점 1곳 | 표적 **192** / off **722** (뇌 전역 60시드) |
| 몽타주 | 손으로 고른 field_opt·af_opt | **70전극 전수탐색**(275만 몽타주/비율)으로 드라이브별 최적 |
| 판정 | 몽타주 순위 | 개체군 **recruitment** (몇 %가 켜지는가) |

---

## 2. 넘길 파일

```
tip/data/fibercand_hippoL_full.npz
```

`np.load(path, allow_pickle=True)` 로 열면:

| 키 | shape / 형 | 의미 |
|---|---|---|
| `coords` (= `trajs`) | **(914, 31, 3)** float64 | 섬유별 절점 좌표 [mm] |
| `arclen` | **(914, 31)** float64 | 섬유별 누적 호길이 [mm] ★§3 주의 |
| `labels` | (914,) str | `fiber0007:target` / `fiber0500:off` |
| `target` | (914,) bool | 표적(좌해마) 통과 여부 — 192 True |
| `target_center` | (3,) | 표적 섬유 절점 평균 좌표 |
| `axis` | (3,) | 섬유 다발 평균 방향(단위벡터) |
| `f1`, `f2` | float | 반송파 2000 / 2100 Hz |
| `<이름>__Ve1`, `__Ve2` | **(914, 31)** float32 | **반송파별 세포외전위 [mV]** |
| `<이름>__montage` | str | `('AF7','C6') x ('FT9','TP9')` |
| `<이름>__montage_json` | str | 전극·비율·M1/M2/M3 전체 |

`<이름>` = `field_opt` · `af_opt` · `gaf_opt` (3종).

---

## 3. ★기존 하네스에서 고칠 곳 — 한 줄

`neuron_bridge.export_ti_case()` 는 모든 궤적이 직선·등간격이라 **`arclen` 이 `(N,)` 1차원**이었다.
이번엔 실제 궤적이라 **섬유마다 호길이가 다르므로 `(F, N)` 2차원**이다.

```python
# (구)  s = d["arclen"]                 # (N,)
# (신)  s = d["arclen"][i]              # 섬유 i 의 (N,)
```

나머지 키(`coords`·`labels`·`f1`·`f2`·`__Ve1`/`__Ve2`)는 **규약이 같으므로 그대로** 읽힌다.

---

## 4. 계산할 것

§7과 **동일한 물리·동일한 하네스**. 섬유마다:

```
e_extracellular_i(t) = g · [ Ve1(x_i)·cos(2πf₁t) + Ve2(x_i)·cos(2πf₂t) ]
```

- `Ve1`/`Ve2` 는 이미 **mV**, 좌표는 **mm** (§7과 같은 규약)
- **g\* = 이분탐색 역치** (섬유별)
- 축삭 모델은 §7에서 쓰던 것 그대로 (MRG 계열 권장 — 9.6 GAF 가 겨냥한 모델)

**돌려줄 것**: 섬유별 `g*` 3벌(몽타주 3종). 안 켜지면 `inf` 또는 `nan`.
간단히 `np.savez("thresholds.npz", field_opt=g_field, af_opt=g_af, gaf_opt=g_gaf)` 면 충분하다.

### 비용 감각
914섬유 × 몽타주 3 × 이분탐색 ~12회 ≈ **3.3만 회**. 축삭당 0.1~1초면 1~9시간, 코어 병렬로 하룻밤.
줄이려면 표적 192 + off 무작위 200 정도로 부분집합을 써도 판정에는 충분하다.

---

## 5. 판정 기준 — 무엇을 보려는 것인가

§7의 결론은 **"심부·통과섬유에서는 af_opt 가 이긴다(24/24)"** 였다. 개체군 수준에서 그대로 나오는가:

1. **표적 recruitment 곡선** — g 를 올리며 표적 섬유 중 켜진 비율. `af_opt`/`gaf_opt` 가
   `field_opt` 보다 **낮은 g 에서 더 많이** 켜져야 §7과 일치.
2. **선택성** — 같은 g 에서 (표적 켜진 비율) vs (off 켜진 비율).
   [[seqti-selectivity]] 의 "off 가 먼저 켜진다"가 개체군에서도 재현되는지.
3. **필드 지표와의 상관** — M1/M2 순위(아래)가 실제 역치 순위를 얼마나 맞히는가.
   맞히면 앞으로 NEURON 없이 필드 지표만으로 탐색해도 된다는 뜻이다.

### 최종 몽타주 (70전극 전수탐색, 풀 축소 없음 · 275만 몽타주/비율)

```
field_opt : (AF7, T8) x (FT9, P9)    r=0.767   M1=1.0773  M2= 7.684  M3=1.0%   [coarse   796위]
af_opt    : (O2,  T7) x (F10, FT9)   r=0.883   M1=0.5171  M2=14.879  M3=1.9%   [coarse     2위]
gaf_opt   : (O1,  T7) x (F10, FT9)   r=0.767   M1=0.2278  M2=18.253  M3=0.4%   [coarse    51위]
```

**수렴 확인 — 3회 독립 실행 대조** (`n_refine=30000`, coarse 순위가 2·51·796 으로 여유 충분):

| | 1차 (버그) | 2차 (버그) | **3차 (수정본)** |
|---|---|---|---|
| field | (AF7,C6)x(FT9,TP9) | (AF7,T8)x(FT9,P9) | **(AF7,T8)x(FT9,P9)** — 2차와 일치 |
| af | (O2,T7)x(F10,FT9) | 동일 | **동일** — 3회 전부 |
| gaf | (O1,T7)x(F10,FT9) | 동일 | **동일** — 3회 전부 |

> 1·2차는 coarse WP 를 **청크 안에서** 정규화하는 버그가 있어 선별이 사실상 무작위였다
> (같은 해가 2474위↔11188위). 전역 정규화로 고친 3차가 유효본이며, **1차 `field_opt` 는 폐기**.

**교차평가 (평가 드라이브 안에서 WP 정규화) — 세 드라이브 모두 자기 기준 최적이 이긴다:**

| 평가기준 | field_opt | af_opt | gaf_opt |
|---|---|---|---|
| field | **0.872** | 0.284 | 0.253 |
| af | 0.124 | **0.630** | 0.627 |
| gaf | 0.154 | 0.838 | **0.862** |

**읽는 법**: `af_opt` 와 `gaf_opt` 는 전극이 하나(O2 vs O1, 인접한 후두 전극)만 다르고
교차 점수도 사실상 동률이다 — **gaf 는 af 를 케이블커널로 평활한 것**이니 물리적으로 타당하다.
반면 `field_opt` 는 완전히 다른 몽타주이고 af/gaf 기준에서 **0.019 / 0.059** 로 무너진다.
즉 **field 와 (af, gaf) 는 서로 다른 목적함수**이고, af 와 gaf 는 사실상 같은 것이다.
→ 맥 NEURON 이 판정할 것: **실제 역치를 맞히는 쪽이 어느 것인가.**

**|Vₑ| 크기** (ITOTAL=2 mA 등총전류): field 25.1/18.4 · af 24.8/23.7 · gaf 26.6/21.9 mV.

> 결과 파일: `data/fiber_opt_fixed.json` (유효본) · `fiber_opt_full.json`(1차) ·
> `fiber_opt_confirm.json`(2차)는 기록용으로만 남겨둔다.

---

## 6. Sim4Life GAF 를 보류한 이유 (기록)

솔버(`NEURON/S4L`) 설치는 성공했고 GAF API 도 파이썬에서 직접 열린다:
`DoubleCableEstimator.compute_gaf()` · `get_threshold()`(=17.6864, 모델 상수) ·
`get_titration_prediction()`. Sim4Life 앱 초기화 없이 동작하고 라이선스 체크에도 안 걸린다.

`get_potential(model, source)` 까지 **정상 동작**(221구획)했으나,
`compute_gaf` → `StimulationPulse.__init__` 에서 **AssertionError** 로 막혔다.
`NeuronFileParser.read_sources()` 가 만든 정품 `SourceSettings` 로도 동일해 소스 구조 문제가 아니며,
Cython 컴파일이라 조건을 확인할 수 없다.

**해소 조건**: GUI 가 생성한 `<uuid>_Input.h5` 의 `Sources` 그룹을 한 번 보면 확정된다.
현재 GUI 리본에 `NEURON | Axon Model` 그룹이 나타나지 않아 그 파일을 얻지 못했다.

확보해 둔 규약(재개 시 그대로 사용): `_e_potential` 은 평평한 `{구획명: [값]}` ·
시간 단위 **ms** · HDF5 `SectionNames` 는 접두사 없이(파서가 `Neuron<id>_` 부착) ·
MRG 구획 생성은 `RwNeuronInput` → `AxonDiscretization` 로 검증 완료
(node21/MYSA40/FLUT40/STIN120). 자세한 건 메모리 `sim4life-96-mcp`.

> S4L GAF 는 **속도(1000배)** 를 주는 것이지 능력을 주는 게 아니다. 맥 NEURON 으로도
> 결론은 동일하게 나온다 — 후보를 수백 개가 아니라 3개만 검증한다는 차이뿐이다.
