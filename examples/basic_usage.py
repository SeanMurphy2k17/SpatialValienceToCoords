#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Usage Example for Spatial Valence Solution

This example demonstrates the core functionality of the enhanced
spatial valence processor with universal DEEP mode.
"""

import sys
import os

# Add the package to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from spatial_valence import (
    EnhancedSpatialValenceToCoordGeneration,
    SemanticDepth,
    SpatialValenceToCoordGeneration
)

def basic_example():
    """Demonstrate basic usage of the enhanced processor"""
    
    print("🧠 SPATIAL VALENCE SOLUTION - BASIC EXAMPLE")
    print("=" * 50)
    
    # Initialize with universal DEEP mode
    processor = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    # Test cases
    test_texts = [
        "I absolutely love this amazing AI system!",
        "How do quantum computers work?", 
        "The research team discovered breakthrough results yesterday.",
        "I'm extremely frustrated with this broken software!",
        "Tomorrow we will build incredible things together."
    ]
    
    print("🔍 Processing with DEEP mode analysis:")
    print()
    
    for i, text in enumerate(test_texts, 1):
        result = processor.process(text)
        
        print(f"[{i}] \"{text}\"")
        print(f"    Coordinate: {result['coordinate_key'][:30]}...")
        print(f"    Confidence: {result.get('confidence', 0):.3f}")
        
        # Show enhanced analysis if available
        enhanced = result.get('enhanced_analysis', {})
        if 'emotion_score' in enhanced:
            emotion = enhanced['emotion_score']
            emotion_desc = "Positive" if emotion > 0.1 else "Negative" if emotion < -0.1 else "Neutral"
            print(f"    Emotion: {emotion:.3f} ({emotion_desc})")
        
        print()

def comparison_example():
    """Compare original vs enhanced processor"""
    
    print("🔄 ORIGINAL vs ENHANCED COMPARISON")
    print("=" * 50)
    
    # Initialize both processors
    original = SpatialValenceToCoordGeneration()
    enhanced = EnhancedSpatialValenceToCoordGeneration(SemanticDepth.DEEP)
    
    test_text = "I absolutely love this amazing AI system!"
    
    # Process with both
    original_result = original.process(test_text)
    enhanced_result = enhanced.process(test_text)
    
    print(f"Text: \"{test_text}\"")
    print()
    
    print("📊 ORIGINAL PROCESSOR:")
    print(f"  Summary: '{original_result['summary']}'")
    print(f"  Coordinate: {original_result['coordinate_key'][:40]}...")
    print(f"  Features: Basic processing only")
    print()
    
    print("🚀 ENHANCED PROCESSOR:")
    print(f"  Summary: '{enhanced_result['summary']}'")
    print(f"  Coordinate: {enhanced_result['coordinate_key'][:40]}...")
    print(f"  Confidence: {enhanced_result.get('confidence', 0):.3f}")
    
    enhanced_analysis = enhanced_result.get('enhanced_analysis', {})
    if 'emotion_score' in enhanced_analysis:
        print(f"  Emotion: {enhanced_analysis['emotion_score']:.3f}")
    
    print(f"  Features: Full semantic intelligence!")
    print()

def semantic_depth_demo():
    """Demonstrate different semantic depths"""
    
    print("🎯 SEMANTIC DEPTH DEMONSTRATION")
    print("=" * 50)
    
    test_text = "The advanced AI system processes complex information efficiently"
    
    # Test all depths
    depths = [SemanticDepth.FAST, SemanticDepth.STANDARD, SemanticDepth.DEEP]
    
    for depth in depths:
        processor = EnhancedSpatialValenceToCoordGeneration(depth)
        result = processor.process(test_text)
        
        print(f"🔍 {depth.name} Mode:")
        print(f"  Description: {depth.description}")
        print(f"  Confidence: {result.get('confidence', 0):.3f}")
        print(f"  Processing: ~{depth.typical_processing_time_ms}ms")
        print()

def main():
    """Run all examples"""
    basic_example()
    comparison_example()
    semantic_depth_demo()
    
    print("🎯 CONCLUSION:")
    print("✅ Universal DEEP mode provides maximum consistency")
    print("✅ Enhanced processor offers advanced semantic intelligence")
    print("✅ Perfect for STM, LTM, and consciousness systems")
    print("🚀 Ready for production deployment!")

if __name__ == "__main__":
    main() 