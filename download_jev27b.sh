#!/usr/bin/env bash
# autotrust/JEV-27B 다운로드 (wget, hf CLI 불필요)
# 사용법: bash download_jev27b.sh [저장 경로]   (기본: ./JEV-27B)
# 사내 미러/프록시가 있으면: HF_BASE=https://<미러주소> bash download_jev27b.sh
set -euo pipefail

REPO="autotrust/JEV-27B"
REV="main"
HF_BASE="${HF_BASE:-https://huggingface.co}"
DEST="${1:-JEV-27B}"
URL="${HF_BASE}/${REPO}/resolve/${REV}"

# 폴더 구조를 그대로 유지해야 합니다 (vLLM이 JEV-27B/adapter_vllm 경로를 씀)
FILES=(
  # --- 백본 (필수) ---
  config.json
  model.safetensors.index.json
  $(for i in $(seq -w 1 13); do echo "model-000${i}-of-00013.safetensors"; done)
  # --- 토크나이저 (필수) ---
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  # --- vLLM 분류(System 1) 경로 (필수) ---
  adapter_vllm/adapter_config.json
  adapter_vllm/adapter_model.safetensors
  adapter_vllm/decision_head.json
  calibration.json
  # --- transformers+peft 경로 (vLLM만 쓸 거면 선택) ---
  adapter/adapter_config.json
  adapter/adapter_model.safetensors
  head.safetensors
  judge_config.json
  # --- 문서 (선택) ---
  README.md
)

mkdir -p "${DEST}/adapter" "${DEST}/adapter_vllm"
for f in "${FILES[@]}"; do
  echo ">> ${f}"
  # -c: 중단 시 이어받기
  wget -c -q --show-progress -O "${DEST}/${f}" "${URL}/${f}"
done

# 대용량 파일 무결성 검사 (HF LFS oid = SHA256)
cd "${DEST}"
cat > SHA256SUMS <<'EOF'
db2b7b40984acce0aba155c5d662c962c8be3a343c286902f04067b832eb1cb6  model-00001-of-00013.safetensors
d619ecfeb02568617e304b9e2220d8445f9133a878042400828521f5e223a9c4  model-00002-of-00013.safetensors
60adcb827a302f0e7aacccc0bd0840d894c221671248a594514745c53ad5d6ac  model-00003-of-00013.safetensors
62aef67ee25767082503d0fade7c432c1d9425206d0824e9a812af5b9148948e  model-00004-of-00013.safetensors
f2f0d54be97d1d5f852cb84c59598d6de3126be2de63587a81c4f50dac46867d  model-00005-of-00013.safetensors
e6f48774997760439d70c6bd6242c5ecf130d9edf0bac2ac558180f57f385846  model-00006-of-00013.safetensors
a5545d3e075a0c77b3d7c5538de8df0df150f0fe859b4dd7a5f708ff7d5d15a0  model-00007-of-00013.safetensors
62aee5ddd323e4390c084e46a3c20bbf98919cc52443136e6f75c099cb2b6d44  model-00008-of-00013.safetensors
9bb707f2841fef0be82f44074ed8e42922782635f9f8c4b430ccee882d51c175  model-00009-of-00013.safetensors
0f6761163164a3c6315b139e408934db4201caf89d9eed2f297e0f7b693b59d6  model-00010-of-00013.safetensors
e188b8858675dc9ed116967af43c912cc3dedabf2592851e2901e3b7f86f8938  model-00011-of-00013.safetensors
2e9dca8bb19513668aa2972e7c6453395b0145f1a3693fa17120d36e8f986f66  model-00012-of-00013.safetensors
68bbd88d686dfe82e0e98277c6e5a5f5fa0eabbbeea858eb5e8508e2fba2858a  model-00013-of-00013.safetensors
06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523  tokenizer.json
79f691c1fd127157d0fb14cb9dd0aa2f91eddd8047368fd4551348f561871682  adapter_vllm/adapter_model.safetensors
7bdb88d73895931d3cfc55ba95766f45df50f244f631d8fe55255f67ec0b6569  adapter/adapter_model.safetensors
8256776bac46e653a3e126ab4e8d5e6d0eed73e7fa6f1e55306083607e44753d  head.safetensors
EOF
echo ">> SHA256 검증 중 (수 분 소요)"
sha256sum -c SHA256SUMS
echo ">> 완료: $(du -sh . | cut -f1)"
