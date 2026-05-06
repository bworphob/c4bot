# Connect4Bot

**Medium Article:** 
[Read full explanation](https://medium.com/@woraphob.dmi/connect4bot-ai-powered-intelligent-assistant-using-reinforcement-leaning-e6af93d0a82f)


This project focuses on studying AI for playing Connect4 using Reinforcement Learning and deploying it on Jetson Nano 2gb. 

We are interetested in Reinforcement Learning, so we decide to create a Connect4 AI player follow AlphaZero Approach.

The project structure is shown below. 

```
c4bot/
├── src/
│   ├── GameBoard/ 
│   ├── GPIO/
│   ├── ImageProcess/
│   ├── Reinforcement/
│   ├── test_hardware_integration.py
│   ├── test_model.py        
│   └── main.py         
├── .gitignore
├── requirements.txt
└── README.md
```


## Prerequisites

- JetPack 4.6 (Standard for Jetson Nano 2GB)
- Python 3.6+
- TensorRT (Pre-installed with JetPack)

## Setup

1. Update system packages:
```bash
sudo apt update
sudo apt upgrade -y
```

2. Create and activate virtual environment:
```bash
cd ~/c4bot
python3 -m venv --system-site-packages myenv
source myenv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## How to use

### 1. Train the model
Run training from the reinforcement folder:
```bash
cd ~/c4bot
python3 src/Reinforcement/train_main.py --games 50
```
This script generates self-play data, trains a new model, and saves the result under `src/Reinforcement/Models/`.

### 2. Convert the `.keras` model to `.onnx`

```bash
cd ~/c4bot
python3 src/Reinforcement/convert_to_onnx.py
```
This loads `model_v4.keras` and writes `model_v4.onnx` in the current folder.

### 3. Convert `.onnx` to TensorRT `.engine`
Build the engine with TensorRT:
```bash
cd ~/c4bot
/usr/src/tensorrt/bin/trtexec --onnx=src/Reinforcement/Models/model_v4.onnx --saveEngine=src/Reinforcement/Models/model_v4.engine --fp16
```
This step requires a TensorRT-enabled system such as Jetson or a PC with TensorRT installed.

### 4. Run full integration
Run the main hardware integration entry point:
```bash
cd ~/c4bot
sudo python3 src/main.py
```
This is the full system flow with camera, GPIO, OLED, and TensorRT inference.

Note: `sudo` is required for GPIO/Hardware access

## Example test

### 5. Test game rule logic
Run the simple board logic demo:
```bash
cd ~/c4bot
python3 src/GameBoard/Example.py
```
This checks the Connect4 rule engine and valid moves without AI.

### 6. Test the Keras AI model
Run the AI vs human demo using the standard model .keras:
```bash
cd ~/c4bot
python3 src/Reinforcement/playerVsAI.py
```
Note: On some CPUs, you might see a layout error because the model is optimized for GPU (NCHW format). If you are running on Jetson Nano, we recommend using the TensorRT version (Step 7) for better performance.

### 7. Test the TensorRT AI model
Run the TensorRT AI demo:
```bash
cd ~/c4bot
python3 src/Reinforcement/playerVsAI_TRT.py
```
This loads `src/Reinforcement/Models/model_v4.engine` and runs inference through TensorRT.

### 8. Test full hardware with TensorRT
Use the hardware integration test script:
```bash
cd ~/c4bot
sudo python3 src/test_hardware_integration.py
```
This is a Jetson-oriented test that exercises GPIO, OLED, and the TensorRT engine in a hardware flow.

## Notes
- `.engine` conversion requires TensorRT and typically should be done on the target Jetson or TensorRT-capable device.


## Quick commands

```bash
# Train -> Convert -> Optimize -> Play
cd ~/c4bot
python3 src/Reinforcement/train_main.py --games 50
python3 src/Reinforcement/convert_to_onnx.py

/usr/src/tensorrt/bin/trtexec --onnx=src/Reinforcement/Models/model_v4.onnx --saveEngine=src/Reinforcement/Models/model_v4.engine --fp16

sudo python3 src/main.py
```

## Demonstration Video

[![Watch Demo](https://img.youtube.com/vi/56eyrT4iy8A/0.jpg)](https://www.youtube.com/shorts/56eyrT4iy8A)

