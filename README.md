# Spatial Valence Solution

🧠 **Enhanced spatial valence processor for AI consciousness systems**

Transform your spatial memory systems with advanced semantic analysis, emotional intelligence, and universal DEEP mode consistency.

## 🚀 Features

- **Multi-depth Processing**: FAST/STANDARD/DEEP modes for different use cases
- **Emotional Intelligence**: Advanced sentiment analysis with intensity scoring
- **Confidence Scoring**: Every analysis includes reliability metrics
- **Temporal Detection**: Past/present/future relationship analysis
- **Universal Consistency**: Same semantic depth across all systems
- **100% Backward Compatible**: Drop-in replacement for original processor

## 📦 Installation

### From Git Repository
```bash
pip install git+https://github.com/yourusername/spatial-valence-solution.git
```

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/spatial-valence-solution.git
cd spatial-valence-solution

# Install in development mode
pip install -e .
```

## 🎯 Quick Start

```python
from spatial_valence import EnhancedSpatialValenceToCoordGeneration, SemanticDepth

# Initialize with DEEP mode for maximum consistency
processor = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)

# Process text with enhanced analysis
result = processor.process("I absolutely love this amazing AI system!")

# Access enhanced capabilities
print(f"Confidence Score: {result['confidence']:.3f}")
print(f"Emotion Score: {result['enhanced_analysis']['emotion_score']:.3f}")
print(f"Coordinate Key: {result['coordinate_key']}")
```

## 🧠 Processing Depths

### FAST Mode
- **Use case**: Real-time STM conversation processing
- **Speed**: <1ms processing
- **Analysis**: Basic semantic features

### STANDARD Mode  
- **Use case**: Balanced analysis for general processing
- **Speed**: 2-5ms processing
- **Analysis**: Enhanced semantic features

### DEEP Mode (Recommended)
- **Use case**: Maximum consistency across all systems
- **Speed**: 10-20ms processing  
- **Analysis**: Full semantic intelligence
- **Benefits**: Emotional + temporal + relational analysis

## 🔧 Integration Examples

### LTM Integration
```python
from spatial_valence import EnhancedSpatialValenceToCoordGeneration, SemanticDepth

class EngramManager:
    def __init__(self):
        # Universal DEEP mode for maximum consistency
        self.coord_system = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    def store_memory(self, text):
        result = self.coord_system.process(text)
        # Enhanced analysis available in result['enhanced_analysis']
        return result
```

### STM Integration
```python
from spatial_valence import EnhancedSpatialValenceToCoordGeneration, SemanticDepth

class SemanticSTMManager:
    def __init__(self):
        # Same processor as LTM for consistency
        self.coord_system = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    def add_conversation(self, user_input, ai_response):
        context = f"User: {user_input}\nAI: {ai_response}"
        result = self.coord_system.process(context)
        # Confidence and emotion analysis available
        return result
```

## 📊 Enhanced Analysis Features

### Emotional Intelligence
```python
result = processor.process("I'm extremely frustrated with this!")
emotion = result['enhanced_analysis']['emotion_score']
# emotion = -0.863 (strong negative)
```

### Confidence Scoring
```python
result = processor.process("Complex technical analysis...")
confidence = result['confidence']
# confidence = 0.825 (high reliability)
```

### Temporal Analysis
```python
result = processor.process("Yesterday we discovered amazing results")
temporal = result['enhanced_analysis']['temporal_indicators']
# temporal['primary_tense'] = 'past'
```

## 🎯 Universal DEEP Mode Benefits

✅ **Deterministic Consistency** - Same input = same analysis depth everywhere  
✅ **STM = LTM = Consciousness** - Identical semantic processing  
✅ **Simplified Architecture** - One processor for all systems  
✅ **Maximum Intelligence** - Full semantic capabilities everywhere  
✅ **Future-Proof** - Consistent baseline for all projects  

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=spatial_valence

# Run specific test category
pytest tests/test_enhanced.py
pytest tests/test_consistency.py
```

## 📚 Documentation

- [API Documentation](docs/API.md)
- [Integration Guide](docs/INTEGRATION_GUIDE.md)
- [Examples](examples/)

## 🔄 Migration Guide

### From Original Processor
```python
# OLD (in each repo separately)
from LTM.SpatialValenceToCoordGeneration import SpatialValenceToCoordGeneration
processor = SpatialValenceToCoordGeneration()

# NEW (same everywhere)
from spatial_valence import EnhancedSpatialValenceToCoordGeneration, SemanticDepth
processor = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
```

### Dependency Updates
```python
# requirements.txt
spatial-valence-solution>=1.0.0
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🚀 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

- Issues: [GitHub Issues](https://github.com/yourusername/spatial-valence-solution/issues)
- Documentation: [docs/](docs/)
- Examples: [examples/](examples/)

---

**Transform your AI consciousness systems with enhanced spatial valence processing!** 🧠✨ 