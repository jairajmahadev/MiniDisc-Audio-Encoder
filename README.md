
### Encode text to WAV
```powershell
python 2-fsk-modulation.py encode --text "Example text" --out output.wav
```

### Decode WAV to text
```powershell
python 2-fsk-modulation.py decode --wav input.wav
```

### Change bitrate
```powershell
python 2-fsk-modulation.py encode --text "Example text" --out output.wav --bitrate 20
```

## Requirements
- Python 3.7+
- numpy

Install dependencies:
```powershell
pip install numpy
```