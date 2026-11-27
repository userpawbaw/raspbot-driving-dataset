# Raspbot 주행 데이터 수집 및 비교 학습

라즈베리파이에서 녹화한 한 세션을 두 NPZ로 저장한다. 기존 학습기는 `legacy/`를 읽고, 새 학습기는 `mode/`에서 조작 명령과 정상·복귀 상태를 함께 학습한다.

## 수집

Pi에서 기존 Yahboom 모터·서보 모듈과 OpenCV, NumPy를 사용한다. PC 키보드 프로그램에는 Python과 Tk가 필요하다. 학습 PC에는 NumPy, PyTorch, ONNX가 필요하고 Pi에서 자율주행할 때는 ONNX Runtime이 필요하다.

곡선에서는 `--speed`를 좌우 모터 명령의 목표 평균으로 사용한다. 기본값인 `--speed 40 --inner-ratio 0.5`로 전진 좌회전하면 안쪽과 바깥쪽에 각각 27, 53을 보낸다. 바깥쪽이 255에 도달하면 좌우 비율을 유지하며 평균이 낮아진다. 실제 차체 속도는 바닥과 모터 특성에 따라 달라진다. `--motor-ids`는 `왼쪽1 왼쪽2 오른쪽1 오른쪽2` 순서로, Yahboom의 0–3 채널을 한 번씩 지정한다.

```bash
# Pi 터미널
python raspbot_receiver.py --record data --minutes 8 --start-condition normal_laps

# PC 터미널. IP는 Pi 주소로 변경.
python raspbot_keyboard.py 192.168.1.11 --record
```

| 키 | 동작 |
| --- | --- |
| WASD / 방향키 | 여러 방향을 누르면 마지막으로 누른 방향이 우선한다. 그 키를 떼면 아직 누르고 있는 키 중 마지막 방향으로 돌아간다. |
| R | 저장 ON/OFF. 조작은 계속할 수 있다. 시작은 OFF다. |
| M | NORMAL ↔ RECOVERY. 시작 모드는 NORMAL이다. 저장 중에도 바꿀 수 있다. |
| Space | 주행 정지 및 저장 OFF. |
| Esc | 키보드 프로그램 종료. |
| Pi 터미널 Ctrl+C | 녹화 종료 및 NPZ 두 개 저장. |

키보드 창에서 `confirmed`는 Pi가 해당 모드와 저장 상태가 담긴 패킷을 받았다는 뜻이다. 실제 샘플 개수는 Pi 로그의 `Samples`에서 확인한다. 키보드 창이 포커스를 잃거나 PC가 응답을 받지 못하면 저장 상태를 OFF로 돌린다.

1. **정상 주행**은 NORMAL에서 R을 켜고 여러 바퀴 연속으로 돈다.
2. **복귀 시작점 만들기**는 R을 끄고 이동한다. 저장 OFF 상태에서 M으로 RECOVERY를 고른다.
3. **복귀 시범**은 출발 위치와 자세를 잡은 뒤 R을 켜고, 트랙 안에서 정상 주행을 이어갈 수 있을 때까지 수동으로 복귀한다. 이어서 정상 주행도 수집할 경우 M을 눌러 NORMAL로 바꾼다. 다음 복귀 시범을 준비할 때는 다시 R을 끈다.
4. **라벨 전환 직후**에는 카메라에 새 라벨 이후의 프레임 5장이 쌓인 뒤 저장이 재개된다. 모드 변경이나 OFF→ON으로 시작한 구간은 새 `segment_ids` 값을 받는다.
5. **세션 종료**는 Ctrl+C 또는 제한 도달로 처리한다. 30~40분 분량은 5~10분씩 여러 세션으로 나누면 주행 전체를 단위로 학습·검증을 나누기 쉽다.

쌍으로 저장할 때는 첫 유효 녹화 샘플부터, 단일 legacy 파일로 저장할 때는 첫 유효 이동 샘플부터 `--minutes`가 흐른다. 시작 후에는 저장 OFF 시간도 포함한다. 저장 한도는 기본적으로 분당 600개다. `--samples`로 파일당 한도를 낮출 수 있다. 40분 기준 임시 프레임 공간이 최대 약 352 MiB이며, 최종 파일 두 개가 더 필요하다. 원본과 비슷한 영상에서 압축률을 확인하기 전에는 약 1.1 GiB의 여유를 잡는다. Raspberry Pi의 Linux 로컬 파일시스템에서 사용한다.

## 저장 구조

```text
data/
  legacy/run_000008.npz
  mode/run_000003.npz
```

각 폴더에서 이전 번호의 최댓값에 1을 더한다. 기존 번호를 이어가려면 해당 폴더에 기존 번호 파일을 둔다. `7.npz`, `run_000007.npz`, `dataset_7.npz` 같은 숫자 접미사를 읽는다. 위 예에서 두 파일은 같은 세션이며, 모드 파일의 `legacy_filename`이 짝을 가리킨다.

- **Legacy**는 기존 키, 배열 순서, 명령 번호 체계를 사용한다. 저장 ON 동안의 정상과 복귀 샘플을 모두 포함한다.
- **Mode**는 legacy의 모든 샘플과 필드를 동일하게 담고, `schema_version`, `modes` (0 NORMAL / 1 RECOVERY), `mode_names`, `segment_ids`, `context_segment_ids`, `session_id`, `legacy_filename`을 추가한다.
- 저장 OFF 중의 재배치 데이터는 두 파일 모두에서 제외된다. ON 중의 자발적 STOP은 NPZ에 남을 수 있고 기존 및 새 주행 학습기 모두에서 제외된다. 피니시 표시는 필요할 때만 `--finish-on-interrupt`로 활성화한다.
- 기존처럼 새 `--record DATASET.npz` 경로를 직접 지정하면 단일 legacy 기록이 가능하다. PC에서 `--record`를 붙이면 R로 수집을 켜고 끌 수 있다. M으로 선택한 모드는 단일 legacy NPZ에는 저장되지 않는다. PC에서 `--record`를 빼면 첫 이동부터 자동 수집한다.

```bash
python inspect_dataset.py data/mode/*.npz
```

## 같은 데이터로 모델 비교

아래는 같은 번호로 시작한 새 폴더의 예다. 번호가 이미 다르면 모드 파일의 `legacy_filename`으로 대응을 확인한다. 하나의 세션 전체와 그 짝은 같은 분할에 둔다.

```bash
mkdir -p models
python train_raspbot.py --train data/legacy/run_000001.npz data/legacy/run_000002.npz \
  --val data/legacy/run_000003.npz --out models/legacy.onnx --balanced
python train_raspbot.py --with-mode --train data/mode/run_000001.npz data/mode/run_000002.npz \
  --val data/mode/run_000003.npz --out models/mode.onnx --balanced
```

`--with-mode`는 기존 CNN과 64차원 공유 특징을 사용하고, 조작 출력에 더해 NORMAL/RECOVERY 로짓 2개를 출력한다. 기본 손실은 `조작 CE + 0.5 × 모드 CE`이며 `--mode-weight`로 변경할 수 있다. `--balanced`는 기존 행동 가중방식과 같은 제곱근 역빈도 가중치를 모드 손실에도 적용한다. 모드 모델은 NORMAL과 RECOVERY 양쪽에 이동 샘플이 필요하며, 검증 중 모드별 조작 평균 재현율과 모드 혼동행렬을 출력한다. 기존 학습기로도 legacy NPZ를 그대로 읽을 수 있다. 이전 NPZ를 섞는 실험은 별도로 하고 위 비교에는 두 모델에 같은 신규 세션을 사용한다.

## ONNX로 주행

```bash
# Pi
python raspbot_receiver.py --model models/mode.onnx

# PC
python raspbot_keyboard.py 192.168.1.11 --auto
```

키보드 창에서 **Space**를 한 번 누르거나 **Start** 버튼을 클릭하면 주행을 시작한다. 같은 키나 **Stop** 버튼으로 정지한다. 모델 모드에서는 WASD를 사용할 필요가 없다. 키보드 창이 포커스를 잃거나 닫히면 정지 명령을 보낸다. Pi는 0.3초 동안 주행 유지 신호를 받지 않으면 정지하고, 연결이 돌아오면 새로 시작을 눌러야 움직인다.

기존 단일 출력 ONNX도 사용할 수 있다. 두 출력 모델은 예측 모드가 바뀔 때 Pi 콘솔에 표시하며, 구동에는 조작 출력이 사용된다.

## 30~40분 데이터에서 볼 것

10 Hz에서 30~40분은 이론상 18,000~24,000 샘플이다. 저장 OFF, 구간 전환, STOP 제거로 학습에 쓰이는 수는 줄어든다. 30 fps 카메라에서는 연속 샘플의 5프레임 창이 일부 겹치고, 같은 라운드의 장면도 반복된다.

첫 수집은 유효 녹화 시간 기준 정상 60~70%, 복귀·교정 30~40%를 출발점으로 삼는다. 복귀는 여러 코너의 진입부와 출구, 좌우 경계, 차체 각도를 바꿔 30~50개의 서로 다른 시범을 초기 목표로 삼는다. 완전히 이탈한 뒤의 복귀뿐 아니라 코너 진입 전에 조향을 수정하는 시범도 모은다. 복귀 비율과 시범 수는 초기 수집 기준이며, 홀드아웃 주행 결과로 조정한다.

5×3×32×32 입력 CNN의 기존 모델은 조작 8종 기준 210,872개 파라미터다. 모드 출력 추가는 130개를 더해 211,002개가 된다. 우선 기존 크기를 사용하고, 데이터 다양성과 32×32에서 트랙 경계가 보이는지, Pi에서 추론 주기 100 ms를 맞추는지 확인한다. 훈련과 홀드아웃 모두에서 원거리 코너가 잘 보이지 않는다면 해상도나 시야를 따로 비교한다.

실제 성능은 보지 않은 세션의 검증 결과와 별개로 같은 속도·출발 조건에서 직접 비교한다. 완주율, 충돌·수동 개입 횟수, 복귀 성공률 및 복귀 시간을 기록한다. 자율주행 중 개입이 필요했던 시작 자세는 다음 수집의 복귀 시범 후보가 된다. 30~40분은 첫 모델과 이 비교를 시작할 분량으로 보고, 무충돌 여부는 실차 결과로 판정한다. [순차적 모방학습에서 주행 데이터 분포가 달라지는 이유](https://proceedings.mlr.press/v15/ross11a.html)

## 한 번 수집한 데이터로 시간 맥락 3종 비교

기존 수집 명령을 그대로 사용하고, 출력 디렉터리의 `mode/run_*.npz`를
학습 PC로 가져온다. 녹화는 한 번이면 된다. 수집기는 약 10 Hz로 최신
카메라 프레임 5장과 촬영 시각을 저장하고, 아래 조합은 학습 시점에 만든다.

| 입력 | 사용하는 프레임 | 입력 채널 |
| --- | --- | ---: |
| `short` | 최신 연속 5장 | 15 |
| `long` | 1.0, 0.75, 0.50, 0.25초 전과 현재에 가장 가까운 저장 프레임 | 15 |
| `hybrid` | 1.0, 0.8, 0.6, 0.4, 0.2초 전의 저장 프레임과 최신 연속 5장 | 30 |

과거 프레임은 각 녹화 틱에 저장된 최신 프레임에서 고른다.
새 수집기는 녹화가 켜진 동안의 정지 장면도 세션 처음부터 버퍼에 남기며,
`context_segment_ids`를 추가해 녹화를 켠 채 `M`으로 모드만 바꾼 구간을 잇는다.
`R`로 재녹화했거나 통신이 끊긴 뒤에는 새 맥락을 시작한다. 촬영 시각에
0.25초를 넘는 공백이 있어도 맥락을 끊는다. 이전 `mode` 파일은 기존
`segment_ids`를 사용하며 같은 구간 안에서만 프레임을 잇는다.

`R`을 새로 켠 뒤 첫 중요 동작까지 약 1.2초의 영상을 남기고, 벽에 붙은
상태에서 시작한다면 그동안 정지해도 된다. 비교 학습에서는 과거 프레임이
온전히 확보된 동일한 시점을 세 모델에 사용한다. STOP은 과거 프레임으로
사용할 수 있으며, 조작 정답에서는 제외한다.

```bash
# 모드별 표본과 전환 구간 수를 확인한다.
python train_raspbot_recovery.py --inspect --compare-dir models/context \
  --train data/mode/run_000001.npz data/mode/run_000002.npz \
  --val data/mode/run_000003.npz

# 동일한 훈련·검증 세션으로 세 모델을 학습한다.
python train_raspbot_recovery.py --compare-dir models/context \
  --train data/mode/run_000001.npz data/mode/run_000002.npz \
  --val data/mode/run_000003.npz
```

출력은 `short.onnx`, `long.onnx`, `hybrid.onnx`, `comparison.json`이다.
비교 파일에는 모델별 평균 재현율, 정상 주행 중 후진·우회전 오탐,
좌회전 및 복귀 전환 초반의 정답 수, 혼동행렬이 들어간다.
검증용 코너와 장애물 배치는 세션 단위로 따로 확보한다.
모델 하나만 만들 때는 `--context long --out models/long.onnx`처럼 지정한다.

새 수신기는 ONNX의 입력 종류에 맞춰 동일한 규칙으로 프레임을 선택한다.
`long`과 `hybrid`는 자동 주행을 시작한 뒤 필요한 시야가 모일 때까지
정지한 채 기다린다. 입력 형식 메타데이터가 없는 기존 ONNX는 `short`로 실행한다.

```bash
python raspbot_receiver.py --model models/context/hybrid.onnx
```
