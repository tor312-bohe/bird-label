#!/usr/bin/env python3

# Quick test to verify the encyclopedia display works
# This will help us implement a simple fix

from app import recognize_bird_hholove
import json

print("🔍 Testing HHOLOVE Encyclopedia Data Display")
print("="*60)

result = recognize_bird_hholove('uploads/DSC04440.jpg')

if result.get('encyclopedia_data'):
    enc_data = result['encyclopedia_data']
    
    print(f"✅ Bird: {result.get('chinese_name', 'N/A')}")
    print(f"📚 Encyclopedia sections available:")
    
    sections = [
        ('overview', '📋 综述'),
        ('physical_features', '🔍 外形特征'),
        ('identification', '🔎 区别辨识'),
        ('behavior', '🦅 生活习性'),
        ('distribution', '🌍 地理分布'),
        ('breeding', '🥚 生长繁殖'),
        ('vocalizations', '🎵 鸣叫特征'),
        ('conservation_status', '🛡️ 保护现状')
    ]
    
    for key, title in sections:
        if enc_data.get(key) and enc_data[key].strip():
            print(f"   ✓ {title}: {len(enc_data[key])} chars")
        else:
            print(f"   ✗ {title}: Missing")
    
    print("\n📋 Sample content:")
    if enc_data.get('overview'):
        print(f"概述: {enc_data['overview'][:100]}...")
        
    if enc_data.get('physical_features'):
        print(f"外形特征: {enc_data['physical_features'][:100]}...")
    
else:
    print("❌ No encyclopedia data found")
    
print("="*60)
