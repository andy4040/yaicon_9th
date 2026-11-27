# 투명재질 복원을 위한 3D Gaussian Splatting

YAICON 9th 프로젝트. [GLINT](https://github.com/youngju-na/GLINT)를 바탕으로, 장면을 **일반 GS와 광학 GS 두 집합**으로 표현하는 방법을 실험했습니다.

유리 장면에서는 반사된 물체가 유리 뒤에 있는 것처럼 학습되거나, 유리 표면에 색이 붙는 문제가 생깁니다. 저희는 장면의 기하와 색은 일반 GS가 맡고, 유리 표면에서 반사·투과 방향을 정하는 역할은 광학 GS가 맡도록 나눴습니다.

## 모델 구조

GLINT는 interface, transmission, reflection Gaussian을 구분합니다. 저희 모델은 광학 GS에서 반사·투과 광선을 만들고, 두 광선 모두 같은 일반 GS에서 색을 구합니다.

| 집합 | 역할 |
|:--|:--|
| 일반 GS | 장면의 기하와 색을 표현. 직접 보이는 영상뿐 아니라 반사·투과 광선이 만나는 장면도 담당 |
| 광학 GS | 표면 위치와 법선을 공유하면서 반사·투과 경로와 재질 계수를 학습 |

### 광학 GS 시각화

3D-FRONT-T Scene 1~5에서 확정된 광학 GS입니다. 각 행은 **GT / 광학 GS 오버레이 / 광학 GS 단독 렌더링** 순서입니다. 청록색은 광학 GS가 화면을 덮는 정도(coverage)를 표시합니다.

<img src="docs/assets/confirmed-optical-gaussians.png" alt="Scene 1~5의 GT, 광학 GS 오버레이, 확정 광학 GS 단독 렌더링" width="100%">

### 광학 후보 선정

먼저 모든 Gaussian을 일반 GS로 학습합니다. 이후 최대 24개 학습 시점에서 Gaussian 중심을 투영해 diffuse albedo를 확인합니다. 화면 밖에 있거나 가려진 관측은 제외하고, 다음 조건을 만족하면 광학 후보로 옮깁니다.

- `mean(diffuse_albedo) < 0.05`인 관측이 2개 이상
- 위 조건을 만족하는 관측이 유효 관측의 60% 이상

검은 불투명 물체도 albedo가 낮을 수 있어서 바로 확정하지는 않습니다. 후보를 일정 기간 학습한 뒤, 같은 위치를 불투명 표면으로 렌더링했을 때와 광학 경로로 렌더링했을 때의 RGB L1 오차를 비교합니다. 여러 시점에서 개선이 확인된 후보만 광학 GS로 남기고, 나머지는 일반 GS로 돌려보냅니다. 기본 개선 임계값은 `0.005`입니다.

현재 버전은 한 번만 확정하며, roughness 조건이나 보류 후보 재평가는 포함하지 않습니다. 자세한 판정 조건은 [optical_selection.py](glint/optical_selection.py)에 있습니다.

### 반사와 투과

광학 표면의 색은 다음 식으로 합성합니다.

$$
C_{\mathrm{opt}}=\beta L_r+(1-\beta)\left[tL_t+(1-t)C_{\mathrm{surface}}\right]
$$

여기서 $L_r$과 $L_t$는 각각 반사·투과 광선으로 일반 GS에서 구한 색이고, $C_{\mathrm{surface}}$는 표면색입니다. 반사 계수 $\beta$와 조건부 투과 계수 $t$를 학습하며, 세 성분의 가중치 합은 1입니다.

이 방식은 코드에서 `direct_beta`로 부릅니다. Fresnel 항을 곱하지 않고 학습된 반사 계수를 직접 사용합니다. 반사 방향은 법선으로 계산하고 투과 광선은 직진합니다. 마지막으로 coverage와 전경 가림을 반영해 일반 GS 영상과 합성합니다.

### 학습 과정

제공한 Ref-DL3DV 실행 스크립트의 일정입니다.

| 학습 횟수 | 내용 |
|:--|:--|
| 0~5,000 | 일반 GS로 장면의 기하와 색 학습 |
| 5,000~10,000 | 광학 후보 시험 학습. 증식·제거와 광학 SH 업데이트는 잠시 중단 |
| 10,000 | 후보를 검증하고 광학 GS 확정 |
| 10,000~30,000 | 두 집합을 함께 학습하면서 증식·제거 |
| 30,000~60,000 | Gaussian 구성을 고정하고 나머지 파라미터 최적화 |

학습 손실은 RGB L1, SSIM, 법선 prior로 구성합니다. 일반 GS와 광학 GS의 법선 손실 가중치는 각각 `0.1`입니다. SSIM은 학습 손실에 사용하며, 후보 수락 여부는 RGB L1 개선으로 판단합니다.

## 실험 결과

### Ref-DL3DV

Ref-DL3DV 8개 씬을 각각 60,000회 학습한 실험 기록입니다. 평균은 씬별 검증 평균을 다시 평균한 값입니다.

| Scene ID | 검증뷰 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:--|--:|--:|--:|--:|
| `b65e8683` | 47 | 31.127 | 0.9388 | 0.0518 |
| `b9df30d6` | 46 | 29.594 | 0.9154 | 0.0631 |
| `52410f02` | 48 | 35.804 | 0.9645 | 0.0472 |
| `6b42314a` | 41 | 30.334 | 0.9404 | 0.0623 |
| `543b6607` | 47 | 29.193 | 0.9266 | 0.0582 |
| `194defaa` | 40 | 26.466 | 0.8758 | 0.1058 |
| `3712b8fd` | 51 | 27.520 | 0.8952 | 0.0999 |
| `5454b71d` | 38 | 35.222 | 0.9807 | 0.0152 |
| **평균** | **총 358뷰** | **30.657** | **0.9297** | **0.0629** |

Scene ID는 앞 8자리만 표기했습니다. SSIM과 LPIPS는 저장된 PNG로 계산했으며, LPIPS는 AlexNet 기준입니다.

### GLINT와 비교

| 모델 | Ref-DL3DV PSNR ↑ | SSIM ↑ | 3D-FRONT-T PSNR ↑ | SSIM ↑ |
|:--|--:|--:|--:|--:|
| GLINT | 30.21 | 0.92 | **34.50** | **0.96** |
| Ours | **30.66** | **0.93** | 34.08 | 0.95 |

YAICON 발표에 사용한 결과입니다. GLINT는 논문 보고값, Ours는 프로젝트 실험값이며 같은 환경에서 다시 측정한 비교는 아닙니다. 코드의 기반 버전과 실험 기록은 [SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md)에 정리했습니다.

### 남은 문제

반사와 투과를 분리하는 문제는 아직 남아 있습니다. RGB 오차가 줄어도 반사상이 투과 경로에 학습될 수 있고, 표면 법선이 틀어지면 반사 방향도 어긋납니다. 초기 후보에서 놓친 표면을 나중에 다시 찾는 방법도 필요합니다. 현재는 직진 투과를 사용하므로 두꺼운 유리의 굴절은 다루지 않습니다.

## 실행 방법

### 설치

Linux와 NVIDIA GPU, CUDA 개발 툴킷이 필요합니다. Python 3.11 또는 3.12를 권장합니다. 렌더링은 GLINT용 gsplat과 OptiX, 학습은 `TwoSetTrainer`를 사용합니다.

```bash
git clone --recursive https://github.com/andy4040/yaicon_9th.git
cd yaicon_9th
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

export CUDA_HOME=/usr/local/cuda
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
BUILD_2DGS=1 NUM_CHANNELS=3,4,6,7 BUILD_EXPERIMENTAL=0 \
  pip install --no-build-isolation \
  'git+https://github.com/youngju-na/gsplat.git@83a0dcd3f850cccdc7b92b50fd6a568e3260eae1'
pip install -r requirements.txt
pip install --no-deps -e .
python -m glint.install_optix
python -m glint.verify_installation
```

`CUDA_HOME`은 설치 경로에 맞게 바꿔주세요. 카메라와 설정 파일은 EasyVolCap 형식을 읽지만, EasyVolCap 학습 루프를 사용하는 버전은 아닙니다.

### 데이터 준비

[GLINT 공식 저장소](https://github.com/youngju-na/GLINT)에서 Ref-DL3DV 데이터를 받아 GPU 서버에 준비합니다. RGB, 카메라, `sparse/0/points3D.ply`, DiffusionRenderer prior가 들어 있는 폴더 구조를 그대로 사용합니다.

```text
/path/to/ref-dl3dv/
├── b65e86833c1ae29714ce881bb9d14d3ed1256a08ab944fd9e75d6b29c674346d/
├── b9df30d6e6078880acc88acb01872c65d337f84b9dba44a23fa29c9861d7e23b/
└── ...
```

전체 씬 ID와 학습·검증 분할은 [configs/datasets/ref-dl3dv](configs/datasets/ref-dl3dv)에 있습니다. 실행 스크립트는 이미지 비율 `0.5`를 사용하며, 씬마다 지정된 검증뷰 전체를 평가합니다. 데이터와 학습 체크포인트는 저장소에 포함하지 않았습니다.

### 학습

```bash
# 실행 명령 확인
python scripts/train_ref_dl3dv_two_set.py \
  --data-root /path/to/ref-dl3dv --output-root results/ref_dl3dv_two_set --dry-run

# 8개 씬 학습. 각 씬이 끝나면 검증 실행
python scripts/train_ref_dl3dv_two_set.py \
  --data-root /path/to/ref-dl3dv --output-root results/ref_dl3dv_two_set

# 체크포인트에서 재개
python scripts/train_ref_dl3dv_two_set.py \
  --data-root /path/to/ref-dl3dv --output-root results/ref_dl3dv_two_set --resume
```

한 씬만 돌리려면 `--scenes FULL_SCENE_ID`를 추가합니다. 실행 인자는 `launch_commands.json`에 저장됩니다. 체크포인트는 5,000회마다 `checkpoints/latest.pt`에 덮어쓰므로, 중간 결과를 보관하려면 따로 복사해 두어야 합니다.

우리 모델의 학습 진입점은 `glint.train_two_set`입니다. 기존 `train_ref_dl3dv_6b42.sh`는 원본 GLINT 학습기인 `glint.train`을 호출합니다.

### 평가와 렌더링

```bash
SCENE=b65e86833c1ae29714ce881bb9d14d3ed1256a08ab944fd9e75d6b29c674346d
python -m glint.evaluate_two_set \
  --config configs/exps/glint/ref-dl3dv/${SCENE}.yaml \
  --data-root /path/to/ref-dl3dv/${SCENE} \
  --checkpoint results/ref_dl3dv_two_set/${SCENE}/checkpoints/latest.pt \
  --output-dir results/ref_dl3dv_two_set/${SCENE}/eval \
  --ratio 0.5 --split val --save-components
```

`metrics.json`에 PSNR과 L1을 저장하고, 최종 RGB, GT, 일반 GS, 광학 coverage, 반사·투과 기여 영상을 출력합니다. 반사·투과 영상에는 재질 가중치와 coverage가 적용되어 있습니다. SSIM과 LPIPS는 이 명령에서 계산하지 않습니다.

## 주요 코드

| 파일 | 내용 |
|:--|:--|
| [two_set_model.py](glint/two_set_model.py) | 두 Gaussian 집합 관리와 체크포인트 |
| [optical_selection.py](glint/optical_selection.py) | 광학 후보 선정과 검증 |
| [two_set_transport.py](glint/two_set_transport.py) | 반사·투과 광선 추적과 합성 |
| [train_two_set.py](glint/train_two_set.py) | 학습 단계, 손실, optimizer |
| [two_set_refinement.py](glint/two_set_refinement.py) | 광학 GS 증식·제거 |
| [evaluate_two_set.py](glint/evaluate_two_set.py) | 평가와 성분별 렌더링 |

## 참고

[GLINT](https://github.com/youngju-na/GLINT)의 코드를 기반으로 개발했습니다. 래스터라이저는 GLINT용 gsplat, 광선 추적은 diff-surfel-tracing과 NVIDIA OptiX를 사용합니다. 원본 설명은 [upstream README](docs/UPSTREAM_README.md), 라이선스는 [LICENSE](LICENSE)와 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고하세요.
