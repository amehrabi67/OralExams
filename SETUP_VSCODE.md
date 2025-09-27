# VS Code Setup Guide for Enhanced Tutoring System

## 🚀 Quick Start in VS Code

### 1. Open the Project
1. Open VS Code
2. File → Open Folder → Select the `OralExams` folder
3. The project structure should be visible in the Explorer

### 2. Set Up Python Environment
1. Open VS Code Terminal (Ctrl+` or View → Terminal)
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Set Up Hugging Face Token (Optional)
```bash
export HF_TOKEN="your_hugging_face_token_here"
```

### 5. Test the System
```bash
# Test with simulation (no hardware needed)
python3 tutoring_cli.py run r2_interpretation_01 --simulation

# List available questions
python3 tutoring_cli.py list

# Interactive mode
python3 tutoring_cli.py
```

## 🎯 VS Code Configuration

### Recommended Extensions
- Python (Microsoft)
- Python Debugger
- Jupyter
- GitLens

### Launch Configuration
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Tutoring CLI",
            "type": "python",
            "request": "launch",
            "program": "tutoring_cli.py",
            "args": ["run", "r2_interpretation_01", "--simulation"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Interactive Mode",
            "type": "python",
            "request": "launch",
            "program": "tutoring_cli.py",
            "args": ["interactive"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

## 🧪 Testing Commands

### Basic Tests
```bash
# 1. Test question listing
python3 tutoring_cli.py list

# 2. Test simulation mode
python3 tutoring_cli.py run r2_interpretation_01 --simulation

# 3. Test search functionality
python3 tutoring_cli.py search --topic "regression"

# 4. Test configuration
python3 tutoring_cli.py config --create-default
python3 tutoring_cli.py config --show
```

### Advanced Tests
```bash
# Test with custom response
python3 examples/enhanced_simulated_turn.py --question-id r2_interpretation_01 --response "R squared means variance explained"

# Test live mode (requires camera/microphone)
python3 tutoring_cli.py run r2_interpretation_01

# Test question creation
python3 tutoring_cli.py add
```

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Make sure you're in the project root directory
2. **Module Not Found**: Run `pip install -e .` to install in development mode
3. **Camera Issues**: Use `--simulation` flag for testing without hardware
4. **Permission Issues**: Grant camera/microphone permissions on macOS

### Debug Mode
```bash
# Run with debug output
python3 -u tutoring_cli.py run r2_interpretation_01 --simulation
```

## 📁 Project Structure
```
OralExams/
├── oral_exams/
│   ├── question_manager.py      # Question management system
│   ├── tutoring_interface.py    # Main user interface
│   ├── config.py               # Configuration system
│   ├── agents/
│   │   ├── scoring.py          # Enhanced scoring agent
│   │   ├── coaching.py         # Enhanced coaching agent
│   │   └── ...                 # Other agents
│   └── ...
├── examples/
│   ├── enhanced_simulated_turn.py
│   └── enhanced_live_session.py
├── tutoring_cli.py             # Main CLI tool
├── README_ENHANCED.md          # Comprehensive documentation
└── SETUP_VSCODE.md            # This file
```

## 🎓 Next Steps
1. Try the interactive mode: `python3 tutoring_cli.py`
2. Create your own questions
3. Test with different student responses
4. Customize the configuration
5. Explore the code structure in VS Code
