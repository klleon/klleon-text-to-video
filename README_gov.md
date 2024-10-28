### Setup
1. Start container with nvidia/pytorch:23.11-py3 image.
```
sudo nerdctl run --gpus all -dt --net host -v $PWD:/workspace2 --device /dev/nvidiactl --device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidia2 --device /dev/nvidia3 --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --name gov_is_t2v_daeyang nvcr.io/nvidia/pytorch:23.11-py3
```
2. install requirements.txt
```
pip install -r requirements.txt
```
3. uninstall apex due to fused_layer_norm_cuda.cpython error
```
pip uninstall apex
```
4. Run inference
```
cd inference/
python cli_demo_quantization.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-2b
```

### Gov test [241028]
1. Prepare `prompts.txt`
2. Run `1_parse_prompts.sh` to parse the whole prompts into several small prompts.
3. Run `2_run_inference.sh`
4. Run `v2f.py`
5. Run `is_score_calculate.py`

### Misc.
- `quantization_scheme` and `dtype` in `cli_demo_quantization.py` does not accelerate inference time.
- Opencv VideoCapture could not be work. Follow below steps.
    1. Check path of cv2. `python` -> `import cv2` -> `p cv2.__path__`
    2. Remove cv2 directory (e.g., /usr/local/lib/python3.10/dist-packages/cv2)
    3. Install opencv-python, `pip install opencv-python`
    4. Install required packages, `apt update && apt install -y libgl1-mesa-glx
    5. Check `cv2.VideoCaptre()`
