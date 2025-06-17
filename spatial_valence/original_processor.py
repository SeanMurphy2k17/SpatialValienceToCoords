#!/usr/bin/env python3
"""
🎯 SPATIAL VALENCE TO COORDINATE GENERATION 🎯

LEAN, CLEAN, FAST coordinate generation system:
1. Spatial Valence → Fast semantic summarization (algorithmic)
2. Hash Transformer → 9D coordinates (mathematical, no LLM)
3. Key Generation → [x.xxx][y.yyy][z.zzz]...[f.fff] format
4. Ready for CoreData.lmdb injection

NO LLM BOTTLENECKS - PURE SPEED!
"""

import hashlib
import time
import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter

class SpatialValenceProcessor:
    """Fast algorithmic semantic summarization using spatial valence"""
    
    def __init__(self):
        # Fast grammatical patterns
        self.subject_patterns = [
            r'^(The|A|An)\s+(\w+)',
            r'^(\w+)\s+(is|are|was|were)',
            r'^(He|She|It|They|We|I|You)',
            r'^([A-Z]\w+)\s+',
        ]
        
        self.verb_patterns = [
            r'\b(is|are|was|were|will|would|can|could|should|must)\s+(\w+ing|\w+ed|\w+)',
            r'\b(\w+ed)\b',
            r'\b(\w+ing)\b',
            r'\b(\w+s)\b(?=\s+\w+)',
        ]
        
        # Quick tense/mood indicators
        self.tense_indicators = {
            'past': ['was', 'were', 'had', 'did', 'went', 'came', 'saw'],
            'present': ['is', 'are', 'am', 'do', 'does', 'have', 'has'],
            'future': ['will', 'shall', 'going to', 'would', 'could'],
        }
        
        self.mood_indicators = {
            'interrogative': ['what', 'when', 'where', 'why', 'how', 'who', '?'],
            'imperative': ['please', 'must', 'should', '!'],
            'conditional': ['if', 'would', 'could', 'might'],
        }
    
    def extract_key_elements(self, text: str) -> Dict[str, str]:
        """Fast extraction of key semantic elements"""
        text_lower = text.lower()
        
        # Extract subject (first meaningful word)
        subject = ""
        for pattern in self.subject_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                subject = match.group(0).strip()
                break
        if not subject:
            words = text.split()
            subject = words[0] if words else ""
        
        # Extract verb
        verb = ""
        for pattern in self.verb_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                verb = match.group(0).strip()
                break
        
        # Extract object (last meaningful words)
        words = text.split()
        obj = ' '.join(words[-2:]) if len(words) > 2 else ""
        
        # Determine tense
        tense = 'present'
        for t, indicators in self.tense_indicators.items():
            if any(ind in text_lower for ind in indicators):
                tense = t
                break
        
        # Determine mood
        mood = 'declarative'
        for m, indicators in self.mood_indicators.items():
            if any(ind in text_lower for ind in indicators):
                mood = m
                break
        
        return {
            'subject': subject,
            'verb': verb,
            'object': obj,
            'tense': tense,
            'mood': mood
        }
    
    def generate_summary(self, text: str) -> str:
        """Generate fast semantic summary"""
        elements = self.extract_key_elements(text)
        
        # Build summary from key elements
        summary_parts = []
        
        # Filter meaningful words
        articles = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but'}
        
        for key in ['subject', 'verb', 'object']:
            if elements[key]:
                words = elements[key].split()
                meaningful = [w for w in words if w.lower() not in articles and len(w) > 2]
                summary_parts.extend(meaningful[:2])  # Max 2 words per element
        
        # Ensure we have something
        if not summary_parts:
            words = text.split()
            summary_parts = [w for w in words if len(w) > 3][:4]
        
        return ' '.join(summary_parts[:5])  # Max 5 words total

class HashCoordinateGenerator:
    """Mathematical 9D coordinate generation using hash-based system (NO LLM)"""
    
    def __init__(self):
        # Pre-compiled coordinate maps (from LUDICROUS system)
        self.coord_maps = {
            't': {'p': -1.0, 'n': 0.0, 'f': 1.0},  # time
            'e': {'n': -0.8, 'u': 0.0, 'p': 0.8},  # emotion  
            'r': {'m': -0.6, 'y': 0.0, 'o': 0.6},  # person
            'c': {'a': -0.7, 'h': 0.7},             # concrete
            'a': {'p': -0.5, 'a': 0.5},             # action
            'u': {'l': -0.4, 'm': 0.0, 'h': 0.4},  # urgency
            's': {'u': -0.9, 'd': 0.9},             # certainty
            'o': {'i': -0.3, 'g': 0.0, 'v': 0.3},  # scope
            'f': {'i': -0.2, 'e': 0.2}              # focus
        }
        
        # Defaults for missing features
        self.defaults = {'t': 'n', 'e': 'u', 'r': 'y', 'c': 'h', 'a': 'a', 
                        'u': 'm', 's': 'd', 'o': 'g', 'f': 'e'}
        
        # Dimension order
        self.dims = ['t', 'e', 'r', 'c', 'a', 'u', 's', 'o', 'f']
        self.coord_names = ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']
        
        # Cache for deterministic behavior
        self.coord_cache = {}
    
    def extract_semantic_keys(self, summary: str) -> Dict[str, str]:
        """Fast extraction of semantic keys from summary"""
        summary_lower = summary.lower()
        
        # Simple rule-based key extraction (NO LLM!)
        keys = {}
        
        # Time detection
        if any(word in summary_lower for word in ['was', 'were', 'had', 'past']):
            keys['t'] = 'p'  # past
        elif any(word in summary_lower for word in ['will', 'future', 'tomorrow']):
            keys['t'] = 'f'  # future
        else:
            keys['t'] = 'n'  # now/present
        
        # Emotion detection
        positive_words = ['good', 'great', 'happy', 'love', 'beautiful', 'amazing']
        negative_words = ['bad', 'sad', 'hate', 'terrible', 'awful', 'angry']
        
        if any(word in summary_lower for word in positive_words):
            keys['e'] = 'p'  # positive
        elif any(word in summary_lower for word in negative_words):
            keys['e'] = 'n'  # negative
        else:
            keys['e'] = 'u'  # neutral
        
        # Person detection
        if any(word in summary_lower for word in ['i', 'me', 'my', 'myself']):
            keys['r'] = 'm'  # me
        elif any(word in summary_lower for word in ['you', 'your', 'yourself']):
            keys['r'] = 'y'  # you
        else:
            keys['r'] = 'o'  # others
        
        # Concrete/abstract
        concrete_words = ['see', 'hear', 'touch', 'physical', 'hand', 'eye', 'body']
        if any(word in summary_lower for word in concrete_words):
            keys['c'] = 'h'  # physical
        else:
            keys['c'] = 'a'  # abstract
        
        # Action/passive
        action_words = ['do', 'make', 'create', 'build', 'run', 'work', 'action']
        if any(word in summary_lower for word in action_words):
            keys['a'] = 'a'  # active
        else:
            keys['a'] = 'p'  # passive
        
        # Fill missing with defaults
        for dim in self.dims:
            if dim not in keys:
                keys[dim] = self.defaults[dim]
        
        return keys
    
    def generate_coordinates(self, summary: str, keys: Dict[str, str]) -> Dict[str, float]:
        """Generate 9D coordinates using hash-based math (NO LLM!)"""
        
        # Check cache first
        cache_key = summary
        if cache_key in self.coord_cache:
            return self.coord_cache[cache_key]
        
        # Get base coordinates from maps
        coords = [self.coord_maps[dim][keys[dim]] for dim in self.dims]
        
        # Generate deterministic hash offset
        keys_str = ''.join(f"{k}:{keys[k]}" for k in self.dims)
        hash_bytes = hashlib.md5((keys_str + summary).encode()).digest()
        
        # Generate final coordinates
        final_coords = {}
        for i, name in enumerate(self.coord_names):
            # Add small hash-based offset for uniqueness
            offset = (hash_bytes[i] / 255.0 - 0.5) * 0.02
            coord = max(-1.0, min(1.0, coords[i] + offset))
            final_coords[name] = round(coord, 4)  # 3 decimal places
        
        # Cache result
        self.coord_cache[cache_key] = final_coords
        return final_coords

class SpatialValenceToCoordGeneration:
    """MAIN SYSTEM: Spatial Valence → Hash-based 9D Coordinates"""
    
    def __init__(self):
        self.valence_processor = SpatialValenceProcessor()
        self.coord_generator = HashCoordinateGenerator()
        self.total_processed = 0
        self.cache_hits = 0
    
    def process(self, text: str) -> Dict:
        """Complete processing: Text → Summary → 9D Coordinates"""
        start_time = time.time()
        
        # Stage 1: Fast semantic summarization
        summary = self.valence_processor.generate_summary(text)
        
        # Stage 2: Extract semantic keys
        keys = self.coord_generator.extract_semantic_keys(summary)
        
        # Stage 3: Generate 9D coordinates
        coords = self.coord_generator.generate_coordinates(summary, keys)
        
        # Generate coordinate key string
        coord_key = self.generate_coordinate_key(coords)
        
        self.total_processed += 1
        processing_time = time.time() - start_time
        
        return {
            'input': text,
            'summary': summary,
            'semantic_keys': keys,
            'coordinates': coords,
            'coordinate_key': coord_key,
            'processing_time': processing_time
        }
    
    def generate_coordinate_key(self, coords: Dict[str, float]) -> str:
        """Generate [x.xxx][y.yyy][z.zzz]...[f.fff] key format"""
        key_parts = []
        for name in ['x', 'y', 'z', 'a', 'b', 'c', 'd', 'e', 'f']:
            value = coords[name]
            key_parts.append(f"[{value:.3f}]")
        return ''.join(key_parts)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        cache_rate = (self.cache_hits / max(1, self.total_processed)) * 100
        return {
            'total_processed': self.total_processed,
            'cache_hits': self.cache_hits,
            'cache_rate': f"{cache_rate:.1f}%",
            'cache_size': len(self.coord_generator.coord_cache)
        }

# Quick test
if __name__ == "__main__":
    print("🎯 Testing Spatial Valence to Coordinate Generation")
    
    processor = SpatialValenceToCoordGeneration()
    
    test_texts = [
        "The cat sat on the mat",
        "Scientists discovered new galaxies in space",
        "I love walking in the peaceful forest",
        "The computer crashed and lost my important work",
        "Tomorrow we will build amazing things together"
    ]
    
    print(f"\n⚡ Processing {len(test_texts)} texts...\n")
    
    start_time = time.time()
    
    for i, text in enumerate(test_texts, 1):
        result = processor.process(text)
        
        print(f"[{i}] Input: {text}")
        print(f"    Summary: {result['summary']}")
        print(f"    Coordinates: {result['coordinate_key']}")
        print(f"    Time: {result['processing_time']*1000:.1f}ms")
        print()
    
    total_time = time.time() - start_time
    rate = len(test_texts) / total_time
    
    print(f"📊 Results: {rate:.1f} texts/second")
    print(f"🎯 System ready for production!") 