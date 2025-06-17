#!/usr/bin/env python3
# Quick test for spatial valence package

import sys
sys.path.append('.')

from spatial_valence import EnhancedSpatialValenceToCoordGeneration, SemanticDepth

def quick_test():
    print("🧠 QUICK SPATIAL VALENCE PACKAGE TEST")
    print("=" * 50)
    
    # Initialize processor with DEEP mode
    processor = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    # Test processing
    result = processor.process("I absolutely love this amazing AI system!")
    
    print("✅ Processing successful!")
    print(f"📍 Coordinate: {result['coordinate_key'][:30]}...")
    print(f"🎯 Confidence: {result.get('confidence', 0):.3f}")
    
    enhanced = result.get('enhanced_analysis', {})
    if 'emotion_score' in enhanced:
        print(f"😊 Emotion: {enhanced['emotion_score']:.3f}")
    
    print()
    print("🚀 PACKAGE IMPORT TEST: SUCCESS!")
    print("✅ Enhanced processor working with DEEP mode")
    print("✅ All imports functioning correctly")

if __name__ == "__main__":
    quick_test() 