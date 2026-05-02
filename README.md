# connect4Bot

This project focuses on studying AI for playing Connect4 using Reinforcement Learning and deploying it on Jetson Nano 2gb. 

We are interetested in a reinforcement, so we decide to create a Connect4 AI player follow AlphaZero.

The structure of project are shown as follow. 

```
c4bot/
├── src/
│   ├── GameBoard/ 
│   ├── GPIO/
│   ├── ImageProcess/
│   ├── Reinforcement/
│   ├── test_hardware_integration.py    
│   └── main.py         
├── .gitignore
└── README.md
```

## How to use

### 1. Train the model
Run training from the reinforcement folder:
```bash
python3 src/Reinforcement/train_main.py --games 50
```
This script generates self-play data, trains a new model, and saves the result under `src/Reinforcement/Models/`.

### 2. Convert the `.keras` model to `.onnx`

```bash
python3 src/Reinforcement/convert_to_onnx.py
```
This loads `model_v8.keras` and writes `model_v8.onnx` in the current folder.

### 3. Convert `.onnx` to TensorRT `.engine`
Build the engine with TensorRT:
```bash
/usr/src/tensorrt/bin/trtexec --onnx=src/Reinforcement/Models/model_v8.onnx --saveEngine=src/Reinforcement/Models/model_v8.engine --fp16
```
This step requires a TensorRT-enabled system such as Jetson or a PC with TensorRT installed.

### 4. Run full integration
Run the main hardware integration entry point:
```bash
sudo python3 src/main.py
```
This is the full system flow with camera, GPIO, OLED, and TensorRT inference.

Note: `sudo` is required for GPIO/Hardware access

## Example test

### 5. Test game rule logic
Run the simple board logic demo:
```bash
python3 -m src.GameBoard.Example
```
This checks the Connect4 rule engine and valid moves without AI.

### 6. Test the Keras AI model
Run the AI vs human demo using the standard model .keras:
```bash
python3 -m src.Reinforcement.playerVsAI
```
This uses the existing `ZeroBrain` model logic and lets a human play against AI.

### 7. Test the TensorRT AI model
Run the TensorRT AI demo:
```bash
python3 -m src.Reinforcement.playerVsAI_TRT
```
This loads `src/Reinforcement/Models/model_v6.engine` and runs inference through TensorRT.

### 8. Test full hardware with TensorRT
Use the hardware integration test script:
```bash
sudo python3 -m src.test_hardware_integration
```
This is a Jetson-oriented test that exercises GPIO, OLED, and the TensorRT engine in a hardware flow.

## Notes
- `.engine` conversion requires TensorRT and typically should be done on the target Jetson or TensorRT-capable device.

## Recommended structure
Keep generated artifacts separate from code:
- `src/Reinforcement/Models/` for `.keras` and `.engine`
- `src/Reinforcement/convert_to_onnx.py` for model export

## Quick commands

```bash
# Train -> Convert -> Optimize -> Play
python3 src/Reinforcement/train_main.py --games 50
python3 src/Reinforcement/convert_to_onnx.py

/usr/src/tensorrt/bin/trtexec --onnx=src/Reinforcement/Models/model_v8.onnx --saveEngine=src/Reinforcement/Models/model_v8.engine --fp16

sudo python3 main.py
```