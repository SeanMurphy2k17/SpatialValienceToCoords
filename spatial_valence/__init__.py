"""
Spatial Valence Solution - Enhanced semantic analysis for AI consciousness systems

This package provides advanced spatial valence processing capabilities with:
- Multi-depth semantic analysis (FAST/STANDARD/DEEP)
- Emotional intelligence and confidence scoring
- Temporal relationship detection
- Universal deterministic consistency
- 100% backward compatibility

Example Usage:
    from spatial_valence import EnhancedSpatialValenceToCoordGeneration, SemanticDepth
    
    # Universal DEEP mode for maximum consistency
    processor = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    result = processor.process("I absolutely love this amazing AI system!")
    
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Emotion: {result['enhanced_analysis']['emotion_score']:.3f}")
"""

# Import SemanticDepth first
from .semantic_depth import SemanticDepth

# Import processors (original first for compatibility)
from .original_processor import SpatialValenceToCoordGeneration

# Import enhanced processor (will use the SemanticDepth from above if needed)
from .enhanced_processor import EnhancedSpatialValenceToCoordGeneration

# Import universal processor 
from .universal_processor import UniversalSpatialProcessor

__version__ = "1.0.0"
__author__ = "Sean"
__email__ = "your-email@example.com"
__description__ = "Enhanced spatial valence processor for AI consciousness systems"

__all__ = [
    "EnhancedSpatialValenceToCoordGeneration",
    "SemanticDepth",
    "SpatialValenceToCoordGeneration", 
    "UniversalSpatialProcessor",
]

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version():
    """Get the package version string"""
    return __version__

def get_info():
    """Get package information"""
    return {
        'name': 'spatial-valence-solution',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'capabilities': [
            'Multi-depth semantic processing',
            'Emotional intelligence',
            'Confidence scoring',
            'Temporal relationship detection',
            'Universal DEEP mode consistency'
        ]
    } 